# /// script
# dependencies = [
#   "torch",
#   "torch-einops-utils>=0.1.24",
#   "einops",
#   "ema-pytorch",
#   "hl-gauss-pytorch",
#   "assoc-scan",
#   "mean-conc-beta>=0.1.5",
#   "gymnasium[box2d,other]",
#   "moviepy",
#   "memmap-replay-buffer",
#   "numpy",
#   "fire",
#   "tqdm"
# ]
# ///

from __future__ import annotations

import os
import fire
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from functools import partial
from collections import deque, namedtuple
from random import randrange

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, cat, stack, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import TensorDataset, DataLoader

from mean_conc_beta import Beta as MeanConcBeta

from einops import reduce, repeat, einsum, rearrange, pack

from ema_pytorch import EMA

from torch.optim import Adam

from hl_gauss_pytorch import HLGaussLoss
from torch_einops_utils import temp_eval

from x_ppo import ppo_actor_loss, spo_actor_loss, calc_gae

import gymnasium as gym

from memmap_replay_buffer import ReplayBuffer

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def update_network_(loss, optimizer, params = None, max_grad_norm = None):
    optimizer.zero_grad()
    loss.mean().backward()

    grad_norm = None
    if exists(max_grad_norm) and exists(params):
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

    optimizer.step()
    return grad_norm

# RSM Norm (not to be confused with RMSNorm from transformers)
# this was proposed by SimBa https://arxiv.org/abs/2410.09754
# experiments show this to outperform other types of normalization

class RSMNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        # equation (3) in https://arxiv.org/abs/2410.09754
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_variance', torch.ones(dim))

    def forward(
        self,
        x
    ):
        assert x.shape[-1] == self.dim, f'expected feature dimension of {self.dim} but received {x.shape[-1]}'

        time = self.step.item()
        mean = self.running_mean
        variance = self.running_variance

        normed = (x - mean) / variance.sqrt().clamp(min = self.eps)

        if not self.training:
            return normed

        # update running mean and variance

        with torch.no_grad():

            new_obs_mean = reduce(x, '... d -> d', 'mean')
            delta = new_obs_mean - mean

            new_mean = mean + delta / time
            new_variance = (time - 1) / time * (variance + (delta ** 2) / time)

            self.step.add_(1)
            self.running_mean.copy_(new_mean)
            self.running_variance.copy_(new_variance)

# attention residual - enformer pooling with pre-rmsnorm on keys

LinearNoBias = partial(nn.Linear, bias = False)

# LoRA - https://arxiv.org/abs/2106.09685

class LoRA(Module):
    def __init__(
        self,
        dim,
        r = 16
    ):
        super().__init__()
        self.down = LinearNoBias(dim, r)
        self.up = LinearNoBias(r, dim)

    def forward(self, x):
        return self.up(self.down(x))

# uncompetitive sigmoid attention residual

class AttentionResidual(Module):
    def __init__(
        self,
        dim,
        lora_rank = 16
    ):
        super().__init__()
        self.scale = dim ** -0.5

        self.pseudo_query = nn.Parameter(torch.zeros(dim))
        self.to_keys = nn.Sequential(
            nn.RMSNorm(dim),
            LoRA(dim, r = lora_rank),
            nn.RMSNorm(dim)
        )

    def forward(
        self,
        past_deltas: list[Tensor] | Tensor
    ):
        stacked = torch.stack(list(past_deltas), dim = 0)
        keys = self.to_keys(stacked)

        logits = einsum(self.pseudo_query, keys, 'd, l b ... d -> l b ...') * self.scale

        weights = 2.0 * torch.sigmoid(logits) # uncompetitive sigmoid attention

        return einsum(weights, stacked, 'l b ..., l b ... d -> b ... d')

# SimBa - Kaist + SonyAI

class ReluSquared(Module):
    def forward(self, x):
        return x.sign() * F.relu(x) ** 2

class SimBa(Module):

    def __init__(
        self,
        dim,
        dim_hidden = None,
        depth = 3,
        dropout = 0.,
        expansion_factor = 2,
        use_attn_residual = True,
        lora_rank = 16
    ):
        super().__init__()
        """
        following the design of SimBa https://arxiv.org/abs/2410.09754v1
        """

        self.use_attn_residual = use_attn_residual

        dim_hidden = default(dim_hidden, dim * expansion_factor)

        layers = []

        self.proj_in = nn.Linear(dim, dim_hidden)

        dim_inner = dim_hidden * expansion_factor

        for ind in range(depth):

            block = nn.Sequential(
                nn.RMSNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_inner),
                ReluSquared(),
                nn.Linear(dim_inner, dim_hidden),
                nn.Dropout(dropout)
            )

            attn_residual = AttentionResidual(dim_hidden, lora_rank = lora_rank) if use_attn_residual else None

            layers.append(ModuleList([block, attn_residual]) if use_attn_residual else block)

        self.layers = ModuleList(layers)

        self.final_attn = AttentionResidual(dim_hidden, lora_rank = lora_rank) if use_attn_residual else None

        self.final_norm = nn.RMSNorm(dim_hidden)

    def forward(self, x):
        no_batch = x.ndim == 1

        if no_batch:
            x = rearrange(x, '... -> 1 ...')

        x = self.proj_in(x)

        if not self.use_attn_residual:
            for block in self.layers:
                x = block(x) + x
        else:
            deltas = [x]

            for block, attn_residual in self.layers:
                h = attn_residual(deltas)
                res = block(h)
                deltas.append(res)

            x = self.final_attn(deltas)

        out = self.final_norm(x)

        if no_batch:
            out = rearrange(out, '1 ... -> ...')

        return out

# networks

class Actor(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        bounds = (-1., 1.),
        mlp_depth = 2,
        dropout = 0.1,
        rsmnorm_input = True,  # use the RSMNorm for inputs proposed by KAIST + SonyAI
        use_attn_residual = True,
        beta_eps = 1e-5
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim,
            dim_hidden = hidden_dim * 2,
            depth = mlp_depth,
            dropout = dropout,
            use_attn_residual = use_attn_residual
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            ReluSquared(),
            nn.Linear(hidden_dim, action_dim * 2)
        )

        # beta distribution parameterized by mean and concentration - https://github.com/lucidrains/mean-conc-beta

        self.dist_module = MeanConcBeta(bounds = bounds, init_conc = 2., unimodal = True, eps = beta_eps)

    @temp_eval
    def forward_eval(self, x):
        with torch.no_grad():
            return self.forward(x)

    def forward(self, x):
        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        hidden = self.net(x)

        out = self.action_head(hidden)

        params = rearrange(out, '... (a c) -> ... a c', c = 2)

        return self.dist_module(params)

class Critic(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        dim_pred = 1,
        mlp_depth = 6, # recent paper has findings that show scaling critic is more important than scaling actor
        dropout = 0.1,
        rsmnorm_input = True,
        use_attn_residual = True
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim + action_dim,
            dim_hidden = hidden_dim,
            depth = mlp_depth,
            dropout = dropout,
            use_attn_residual = use_attn_residual
        )

        self.value_head = nn.Linear(hidden_dim, dim_pred)

    @temp_eval
    def forward_eval(self, x, past_action):
        with torch.no_grad():
            return self.forward(x, past_action)

    def forward(self, x, past_action):

        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        x = torch.cat((x, past_action), dim = -1)
        hidden = self.net(x)

        value = self.value_head(hidden)
        return value

# agent

class PPO(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        bounds,
        reward_range: tuple[float, float],
        epochs,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        ema_decay,
        max_grad_norm = 0.5,
        use_spo = False,
        asymmetric_spo = False,
        actor_depth = 2,
        critic_depth = 6,
        use_attn_residual = True,
        beta_eps = 1e-5,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1000
        ),
        save_path = './ppo.pt'
    ):
        super().__init__()

        self.actor = Actor(
            state_dim,
            actor_hidden_dim,
            action_dim,
            bounds = bounds,
            mlp_depth = actor_depth,
            use_attn_residual = use_attn_residual,
            beta_eps = beta_eps
        )

        self.critic = Critic(
            state_dim,
            critic_hidden_dim,
            action_dim,
            dim_pred = critic_pred_num_bins,
            mlp_depth = critic_depth,
            use_attn_residual = use_attn_residual
        )

        self.last_actor_grad_norm = 0.
        self.last_critic_grad_norm = 0.

        # weight tie rsmnorm

        self.rsmnorm = self.actor.rsmnorm
        self.critic.rsmnorm = self.rsmnorm

        # https://arxiv.org/abs/2403.03950

        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = reward_range[0],
            max_value = reward_range[1],
            num_bins = critic_pred_num_bins,
            clamp_to_range = True
        )

        self.ema_actor = EMA(self.actor, beta = ema_decay, include_online_model = False, **ema_kwargs)
        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False, **ema_kwargs)

        self.opt_actor = Adam(self.actor.parameters(), lr = lr, betas = betas)
        self.opt_critic = Adam(self.critic.parameters(), lr = lr, betas = betas)

        self.ema_actor.add_to_optimizer_post_step_hook(self.opt_actor)
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

        # learning hparams

        self.minibatch_size = minibatch_size

        self.epochs = epochs

        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm

        self.use_spo = use_spo
        self.asymmetric_spo = asymmetric_spo # https://arxiv.org/abs/2510.06062v1

        self.save_path = Path(save_path)

    def save(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path), weights_only = True)

        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

    def learn(self, memories: ReplayBuffer, device = None):

        hl_gauss = self.critic_hl_gauss_loss

        # calculate generalized advantage estimate

        dl = memories.dataloader(
            batch_size = 4,
            return_indices = True,
            to_named_tuple = ('_index', 'is_boundary', 'value', 'reward', '_lens'),
            device = device
        )

        for indices, is_boundaries, values, rewards, lens in dl:

            with torch.no_grad():

                masks = (1. - is_boundaries.float())
                scalar_values = hl_gauss(values)

                returns = calc_gae(
                    rewards = rewards,
                    masks = masks,
                    lam = self.lam,
                    gamma = self.gamma,
                    values = scalar_values,
                    lens = lens,
                    use_accelerated = False
                )

                memories.data['returns'][indices, :returns.shape[-1]] = returns.cpu().numpy()
                memories.flush()

        # get data

        dl = memories.dataloader(
            batch_size = self.minibatch_size,
            shuffle = True,
            filter_fields = dict(
                learnable = True
            ),
            to_named_tuple = ('state', 'action', 'action_log_prob', 'returns', 'value', 'past_action'),
            timestep_level = True,
            device = device
        )

        # policy phase training, similar to original PPO

        self.actor.train()
        self.critic.train()

        for _ in range(self.epochs):
            for _, (states, actions, old_log_probs, returns, old_values, past_action) in enumerate(dl):

                dist = self.actor(states)

                action_log_probs = dist.log_prob(actions).sum(dim = -1)
                entropy = dist.entropy().sum(dim = -1)

                scalar_old_values = hl_gauss(old_values)

                # calculate clipped surrogate objective, classic PPO loss

                advantages = returns - scalar_old_values.detach()

                if self.use_spo or self.asymmetric_spo:
                    policy_loss = spo_actor_loss(action_log_probs, old_log_probs, advantages, self.eps_clip, normalize_advantages = True, asymmetric = self.asymmetric_spo)
                else:
                    policy_loss = ppo_actor_loss(action_log_probs, old_log_probs, advantages, self.eps_clip, normalize_advantages = True)

                policy_loss = policy_loss - self.beta_s * entropy

                actor_grad_norm = update_network_(policy_loss, self.opt_actor, params = list(self.actor.parameters()), max_grad_norm = self.max_grad_norm)

                values = self.critic(states, past_action)
                value_loss = hl_gauss(values, returns).mean()
                critic_grad_norm = update_network_(value_loss, self.opt_critic, params = list(self.critic.parameters()), max_grad_norm = self.max_grad_norm)

                if exists(actor_grad_norm):
                    self.last_actor_grad_norm = actor_grad_norm.item()
                if exists(critic_grad_norm):
                    self.last_critic_grad_norm = critic_grad_norm.item()

        # update the state normalization with rsmnorm for 1 epoch after actor critic are updated

        self.rsmnorm.train()

        for states, *_ in dl:
            self.rsmnorm(states)

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 1000,
    max_timesteps = None,
    actor_hidden_dim = 64,
    critic_hidden_dim = 256,
    actor_depth = 2,
    critic_depth = 6,
    use_attn_residual = True,
    beta_eps = 1e-5,
    update_timesteps = 2048,
    buffer_episodes = 40,
    critic_pred_num_bins = 250,
    reward_range = (-300., 300.),
    minibatch_size = 64,
    lr = 0.0005,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    max_grad_norm = 0.5,
    beta_s = 0.005,
    ema_decay = 0.9,
    use_spo = False,
    asymmetric_spo = False,
    epochs = 4,
    seed = None,
    render = False,
    render_every_eps = 50,
    log_every = 5,
    save_every = 1000,
    clear_videos = True,
    video_folder = './lunar-recording',
    load = False,
    save_path = './ppo.pt',
    rolling_window_size = 100,
    stop_at_reward = 200
):
    if env_name.startswith('LunarLander'):
        env = gym.make(env_name, render_mode = 'rgb_array', continuous = True)
    else:
        env = gym.make(env_name, render_mode = 'rgb_array')

    if not exists(max_timesteps):
        max_timesteps = 1000 if 'InvertedPendulum' in env_name else 500

    if render:
        if clear_videos:
            rmtree(video_folder, ignore_errors = True)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = f"{env_name.lower().replace('-v3', '')}-video",
            episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
            disable_logger = True
        )

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    # action bounds - the beta distribution is defined directly on this range

    action_bounds = np.stack([env.action_space.low, env.action_space.high], axis = -1)

    memories = ReplayBuffer(
        f"./{env_name.lower().replace('-v3', '')}-memories/past-action",
        max_episodes = buffer_episodes,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            learnable = 'bool',
            state = ('float', state_dim),
            action = ('float', action_dim),
            action_log_prob = 'float',
            reward = 'float',
            is_boundary = 'bool',
            value = ('float', critic_pred_num_bins),
            returns = 'float',
            past_action = ('float', action_dim)
        ),
        circular = True,
        overwrite = True
    )

    agent = PPO(
        state_dim,
        action_dim,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        action_bounds,
        reward_range,
        epochs,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        ema_decay,
        max_grad_norm = max_grad_norm,
        use_spo = use_spo,
        asymmetric_spo = asymmetric_spo,
        actor_depth = actor_depth,
        critic_depth = critic_depth,
        use_attn_residual = use_attn_residual,
        beta_eps = beta_eps,
        save_path = save_path
    ).to(device)

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0

    rolling_reward = deque(maxlen = rolling_window_size)
    rolling_steps = deque(maxlen = rolling_window_size)

    pbar = tqdm(range(num_episodes), desc = f'episodes ({env_name})')
    for eps in pbar:

        state, _ = env.reset(seed = seed)
        state = torch.from_numpy(state).float().to(device)

        past_action = torch.zeros(action_dim).to(device)

        eps_reward = 0.
        eps_steps = 0

        with memories.one_episode():
            for timestep in range(max_timesteps):
                time += 1

                dist = agent.ema_actor.forward_eval(state)

                action = dist.sample()
                action_log_prob = dist.log_prob(action).sum(dim = -1)

                env_action = action.clamp(action_bounds[:, 0].min(), action_bounds[:, 1].max())
                env_action_item = env_action.cpu().numpy()

                next_state, reward, terminated, truncated, _ = env.step(env_action_item)

                next_state = torch.from_numpy(next_state).float().to(device)

                # custom reward scaling for LunarLander to help learning stability

                reward = float(reward)
                eps_reward += reward
                eps_steps += 1
                value = agent.ema_critic.forward_eval(state, past_action)

                # determine if truncating, either from environment or learning phase of the agent

                updating_agent = divisible_by(time, update_timesteps)
                done = terminated or truncated or updating_agent

                # take care of truncated by bootstrapping the next value for GAE

                if done and not terminated:
                    next_value = agent.ema_critic.forward_eval(next_state, env_action)
                    scalar_next_value = agent.critic_hl_gauss_loss(next_value).item()
                    reward += agent.gamma * scalar_next_value

                memory = memories.store(
                    learnable = True,
                    state = state,
                    action = action,
                    action_log_prob = action_log_prob,
                    reward = reward,
                    is_boundary = done,
                    value = value,
                    past_action = past_action
                )

                state = next_state
                past_action = env_action

                # updating of the agent

                if updating_agent:
                    agent.learn(memories, device)
                    num_policy_updates += 1
                    memories.clear()

                # break if done

                if done:
                    break

        rolling_reward.append(eps_reward)
        rolling_steps.append(eps_steps)

        if divisible_by(eps, log_every):
            print(f"Episode {eps:4d} | Avg reward: {np.mean(rolling_reward):+7.2f} | Ep reward: {eps_reward:+7.2f} | Ep steps: {eps_steps:3d} | a_grad: {agent.last_actor_grad_norm:.2f} | c_grad: {agent.last_critic_grad_norm:.2f}", flush = True)

        pbar.set_postfix(
            reward = f'{np.mean(rolling_reward):.2f}',
            steps = f'{np.mean(rolling_steps):.1f}',
            a_grad = f'{agent.last_actor_grad_norm:.2f}',
            c_grad = f'{agent.last_critic_grad_norm:.2f}'
        )

        if divisible_by(eps, save_every):
            agent.save()

        if exists(stop_at_reward) and len(rolling_reward) >= rolling_window_size and np.mean(rolling_reward) >= stop_at_reward:
            print(f"Rolling reward reached {stop_at_reward}, stopping training.")
            break

    agent.save()

if __name__ == '__main__':
    fire.Fire(main)
