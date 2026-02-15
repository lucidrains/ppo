# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "tqdm",
#   "einops",
#   "ema-pytorch",
#   "adam-atan2-pytorch",
#   "hl-gauss-pytorch",
#   "assoc-scan",
#   "gymnasium[box2d]",
#   "pygame",
#   "memmap-replay-buffer",
#   "moviepy",
#   "hyper-connections",
#   "x-evolution",
#   "accelerate",
#   "wandb",
#   "fire",
# ]
# ///

from __future__ import annotations

import wandb

import fire
from accelerate import Accelerator
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from functools import partial
from collections import deque, namedtuple
from random import randrange

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical

from einops import reduce, repeat, einsum, rearrange, pack

from ema_pytorch import EMA

from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from hl_gauss_pytorch import HLGaussLoss

from hyper_connections import ManifoldConstrainedHyperConnections

from assoc_scan import AssocScan

import gymnasium as gym

from memmap_replay_buffer import ReplayBuffer

# constants (removed manual device logic, handled by Accelerator)

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

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

        return normed

# gradient dropout

class GradientDropout(Module):
    def __init__(
        self,
        strength = 0. # increase for more dropout
    ):
        super().__init__()
        self.strength = strength

    def forward(self, x):
        if not x.requires_grad or not self.training:
            return x

        logit = torch.randn_like(x)
        logit = logit + self.strength
        mask = logit.sigmoid()

        return x * (1. - mask) + x.detach() * mask

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
        num_residual_streams = 4
    ):
        super().__init__()
        """
        following the design of SimBa https://arxiv.org/abs/2410.09754v1
        """

        self.num_residual_streams = num_residual_streams

        dim_hidden = default(dim_hidden, dim * expansion_factor)

        layers = []

        self.proj_in = nn.Linear(dim, dim_hidden)

        dim_inner = dim_hidden * expansion_factor

        # hyper connections

        init_hyper_conn, self.expand_stream, self.reduce_stream = ManifoldConstrainedHyperConnections.get_init_and_expand_reduce_stream_functions(1, num_fracs = num_residual_streams, sinkhorn_iters = 2)

        for ind in range(depth):

            layer = nn.Sequential(
                nn.RMSNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_inner),
                ReluSquared(),
                nn.Linear(dim_inner, dim_hidden),
                nn.Dropout(dropout),
            )

            layer = init_hyper_conn(dim = dim_hidden, layer_index = ind, branch = layer)
            layers.append(layer)

        # final layer out

        self.layers = ModuleList(layers)

        self.final_norm = nn.RMSNorm(dim_hidden)

    def forward(self, x):
        no_batch = x.ndim == 1

        if no_batch:
            x = rearrange(x, '... -> 1 ...')

        x = self.proj_in(x)

        x = self.expand_stream(x)

        for layer in self.layers:
            x = layer(x)

        x = self.reduce_stream(x)

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
        num_actions,
        mlp_depth = 2,
        dropout = 0.1,
        rsmnorm_input = True,  # use the RSMNorm for inputs proposed by KAIST + SonyAI
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim,
            dim_hidden = hidden_dim * 2,
            depth = mlp_depth,
            dropout = dropout
        )

        self.grad_dropout = GradientDropout()

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            ReluSquared(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        hidden = self.net(x)

        hidden = self.grad_dropout(hidden)

        action_probs = self.action_head(hidden).softmax(dim = -1)
        return action_probs

class Critic(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_actions,
        dim_pred = 1,
        mlp_depth = 6, # recent paper has findings that show scaling critic is more important than scaling actor
        dropout = 0.1,
        rsmnorm_input = True,
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim + num_actions,
            dim_hidden = hidden_dim,
            depth = mlp_depth,
            dropout = dropout
        )

        self.grad_dropout = GradientDropout()

        self.value_head = nn.Linear(hidden_dim, dim_pred)

    def forward(self, x, past_action):

        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        x = torch.cat((x, past_action), dim = -1)
        hidden = self.net(x)

        hidden = self.grad_dropout(hidden)

        value = self.value_head(hidden)
        return value

# GAE

def calc_gae(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    assert values.shape[-1] == rewards.shape[-1]
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    values = F.pad(values, (0, 1), value = 0.)

    values, values_next = values[..., :-1], values[..., 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    returns = gae + values

    return returns

# agent

class PPO(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        reward_range: tuple[float, float],
        epochs,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        regen_reg_rate,
        cautious_factor,
        eps_clip,
        value_clip,
        ema_decay,
        use_spo = False,
        asymmetric_spo = False,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1000
        ),
        save_path = './ppo.pt'
    ):
        super().__init__()

        self.actor = Actor(state_dim, actor_hidden_dim, num_actions)

        self.critic = Critic(state_dim, critic_hidden_dim, num_actions, dim_pred = critic_pred_num_bins)

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

        self.opt_actor = AdoptAtan2(self.actor.parameters(), lr = lr, betas = betas, regen_reg_rate = regen_reg_rate, cautious_factor = cautious_factor)
        self.opt_critic = AdoptAtan2(self.critic.parameters(), lr = lr, betas = betas, regen_reg_rate = regen_reg_rate, cautious_factor = cautious_factor)

        self.ema_actor.add_to_optimizer_post_step_hook(self.opt_actor)
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

        # learning hparams

        self.minibatch_size = minibatch_size

        self.epochs = epochs

        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.use_spo = use_spo
        self.asymmetric_spo = asymmetric_spo # https://arxiv.org/abs/2510.06062v1

        self.save_path = Path(save_path)

    def save(self):
        torch.save(dict(
            actor = self.actor.state_dict(),
            critic = self.critic.state_dict(),
            ema_actor = self.ema_actor.state_dict(),
            ema_critic = self.ema_critic.state_dict(),
            rsmnorm = self.rsmnorm.state_dict()
        ), str(self.save_path))

    def load(self, device = None):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path), weights_only = True, map_location = device)

        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

        if 'ema_actor' in data:
            self.ema_actor.load_state_dict(data['ema_actor'])

        if 'ema_critic' in data:
            self.ema_critic.load_state_dict(data['ema_critic'])

        if 'rsmnorm' in data:
            self.rsmnorm.load_state_dict(data['rsmnorm'])

    def learn(self, memories: ReplayBuffer, device = None):
        hl_gauss = self.critic_hl_gauss_loss

        policy_losses = []
        value_losses = []
        entropies = []

        # calculate generalized advantage estimate

        dl = memories.dataloader(
            batch_size = 4,
            return_indices = True,
            to_named_tuple = ('_index', 'is_boundary', 'value', 'reward'),
            device = device
        )

        for indices, is_boundaries, values, rewards in dl:

            with torch.no_grad():

                masks = (1. - is_boundaries.float())
                scalar_values = hl_gauss(values)

                returns = calc_gae(
                    rewards = rewards,
                    masks = masks,
                    lam = self.lam,
                    gamma = self.gamma,
                    values = scalar_values,
                    use_accelerated = False
                )

                memories.data['returns'][indices.cpu().numpy(), :returns.shape[-1]] = returns.cpu().numpy()
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

                action_probs = self.actor(states)
                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                scalar_old_values = hl_gauss(old_values)

                # calculate clipped surrogate objective, classic PPO loss

                ratios = (action_log_probs - old_log_probs).exp()

                advantages = normalize(returns - scalar_old_values.detach())

                if self.use_spo or self.asymmetric_spo:
                    # Xie et al. https://arxiv.org/abs/2401.16025v9 line 14 of Algorithm 1
                    spo_policy_loss = -(
                        ratios * advantages -
                        (advantages.abs() * (ratios - 1.).square()) / (2 * self.eps_clip)
                    )

                if not self.use_spo or self.asymmetric_spo:
                    surr1 = ratios * advantages
                    surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    ppo_policy_loss = - torch.min(surr1, surr2)

                if self.asymmetric_spo:
                    # https://arxiv.org/abs/2510.06062v1
                    policy_loss = torch.where(advantages > 0, ppo_policy_loss, spo_policy_loss)
                elif self.use_spo:
                    policy_loss = spo_policy_loss
                else:
                    policy_loss = ppo_policy_loss

                policy_loss = policy_loss - self.beta_s * entropy

                policy_losses.append(policy_loss.mean().item())
                entropies.append(entropy.mean().item())

                update_network_(policy_loss, self.opt_actor)

                clip = self.value_clip

                def update_critic(critic, scalar_old_values, opt_critic):
                    # calculate clipped value loss and update value network separate from policy network

                    values = critic(states, past_action)

                    scalar_values = hl_gauss(values)

                    # using the proposal from https://www.authorea.com/users/855021/articles/1240083-on-analysis-of-clipped-critic-loss-in-proximal-policy-gradient

                    clipped_returns = returns.clamp(scalar_old_values - clip, scalar_old_values + clip)

                    clipped_loss = hl_gauss(values, clipped_returns, reduction = 'none')
                    loss = hl_gauss(values, returns, reduction = 'none')

                    old_values_lo = scalar_old_values - clip
                    old_values_hi = scalar_old_values + clip

                    def is_between(mid, lo, hi):
                        return (lo < mid) & (mid < hi)

                    value_loss = torch.where(
                        is_between(scalar_values, returns, old_values_lo) |
                        is_between(scalar_values, old_values_hi, returns),
                        0.,
                        torch.min(loss, clipped_loss)
                    )

                    value_loss = value_loss.mean()
                    value_losses.append(value_loss.item())

                    update_network_(value_loss, opt_critic)

                update_critic(self.critic, scalar_old_values, self.opt_critic)

        # update the state normalization with rsmnorm for 1 epoch after actor critic are updated

        self.rsmnorm.train()

        for states, *_ in dl:
            self.rsmnorm(states)

        return dict(
            policy_loss = np.mean(policy_losses),
            value_loss = np.mean(value_losses),
            entropy = np.mean(entropies)
        )

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 500,
    actor_hidden_dim = 64,
    critic_hidden_dim = 256,
    update_timesteps = 5000,
    buffer_episodes = 40,
    critic_pred_num_bins = 250,
    minibatch_size = 64,
    lr = 0.0008,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    value_clip = 0.4,
    beta_s = .01,
    regen_reg_rate = 1e-4,
    use_spo = False,
    asymmetric_spo = False,
    cautious_factor = 0.1,
    ema_decay = 0.9,
    epochs = 2,
    seed = None,
    render = True,
    render_every_eps = 250,
    save_every = 1000,
    clear_videos = True,
    video_folder = './lunar-recording',
    load = False,
    use_wandb = False,
    wandb_project = 'ppo-evolution',
    wandb_run_name = None,
    reward_range = (-300., 300.),
    cpu = False,
):
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    if use_wandb:
        wandb.init(project = wandb_project, name = wandb_run_name, config = locals())

    env = gym.make(env_name, render_mode = 'rgb_array')

    if render:
        if clear_videos:
            rmtree(video_folder, ignore_errors = True)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = 'lunar-video',
            episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
            disable_logger = True
        )

    state_dim = int(env.observation_space.shape[0])
    num_actions = int(env.action_space.n)

    memories = ReplayBuffer(
        './lunar-lander-memories/past-action',
        max_episodes = buffer_episodes,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            learnable = 'bool',
            state = ('float', state_dim),
            action = 'int',
            action_log_prob = 'float',
            reward = 'float',
            is_boundary = 'bool',
            value = ('float', critic_pred_num_bins),
            returns = 'float',
            past_action = ('int', num_actions)
        ),
        circular = True,
        overwrite = True
    )

    agent = PPO(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        reward_range,
        epochs,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        regen_reg_rate,
        cautious_factor,
        eps_clip,
        value_clip,
        ema_decay,
        use_spo,
        asymmetric_spo
    ).to(device)

    if load:
        agent.load(device = device)

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc = 'episodes'):

        state, _ = env.reset()
        state = torch.from_numpy(state).to(device)

        cumulative_reward = 0
        past_action = torch.zeros(num_actions).to(device)

        with memories.one_episode():
            for timestep in range(max_timesteps):
                time += 1

                action_probs = agent.ema_actor.forward_eval(state)
                value = agent.ema_critic.forward_eval(state, past_action)

                dist = Categorical(action_probs)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
                action_item = action.item()

                next_state, reward, terminated, truncated, _ = env.step(action_item)

                next_state = torch.from_numpy(next_state).to(device)

                reward = float(reward)
                cumulative_reward += reward

                memory = memories.store(
                    learnable = True,
                    state = state,
                    action = action,
                    action_log_prob = action_log_prob,
                    reward = reward,
                    is_boundary = terminated,
                    value = value,
                    past_action = past_action
                )

                state = next_state
                past_action = F.one_hot(action, num_classes = num_actions)

                # determine if truncating, either from environment or learning phase of the agent

                updating_agent = divisible_by(time, update_timesteps)
                done = terminated or truncated or updating_agent

                # take care of truncated by adding a non-learnable memory storing the next value for GAE

                if done and not terminated:
                    next_value = agent.ema_critic.forward_eval(state, past_action)

                    bootstrap_value_memory = memory._replace(
                        state = state,
                        learnable = False,
                        is_boundary = True,
                        value = next_value,
                        past_action = past_action
                    )

                    memories.store(**bootstrap_value_memory._asdict())

                # updating of the agent

                if updating_agent:
                    metrics = agent.learn(memories, device)
                    num_policy_updates += 1

                    if use_wandb:
                        wandb.log(dict(
                            **metrics,
                            num_policy_updates = num_policy_updates
                        ))

                # break if done

                if done:
                    break

        if use_wandb:
            wandb.log(dict(
                cumulative_reward = cumulative_reward,
                steps_per_episode = timestep + 1
            ))

        if render and use_wandb and divisible_by(eps, render_every_eps):
            videos = list(Path(video_folder).glob('*.mp4'))
            if len(videos) > 0:
                latest_video = max(videos, key = lambda p: p.stat().st_mtime)
                wandb.log(dict(video = wandb.Video(str(latest_video), format = "mp4")))

        if divisible_by(eps, save_every):
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
