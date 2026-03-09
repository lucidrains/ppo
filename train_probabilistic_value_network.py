# /// script
# dependencies = [
#   "torch",
#   "einops",
#   "ema-pytorch",
#   "adam-atan2-pytorch",
#   "hl-gauss-pytorch",
#   "hyper-connections",
#   "assoc-scan",
#   "gymnasium[box2d,other]",
#   "moviepy",
#   "memmap-replay-buffer",
#   "fire",
#   "tqdm"
# ]
# ///

from __future__ import annotations

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
from torch import nn, tensor, cat, stack, Tensor, exp
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical, Normal

from einops import reduce, repeat, einsum, rearrange, pack

from ema_pytorch import EMA

from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from hyper_connections import ManifoldConstrainedHyperConnections

from assoc_scan import AssocScan

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

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def calculate_uncertainty_exploration_bonus(
    mean: Tensor,
    std: Tensor,
    eps: float = 1e-6
):
    return std / (mean.abs() + eps)

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

        action_probs = self.action_head(hidden).softmax(dim = -1)
        return action_probs

class Critic(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_actions,
        mlp_depth = 6, # recent paper has findings that show scaling critic is more important than scaling actor
        dropout = 0.1,
        rsmnorm_input = True,
        min_log_var = -6,
        max_log_var = 2
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim + num_actions,
            dim_hidden = hidden_dim,
            depth = mlp_depth,
            dropout = dropout
        )

        self.value_head = nn.Linear(hidden_dim, 2)

        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

    def forward(self, x, past_action):

        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        x = torch.cat((x, past_action), dim = -1)
        hidden = self.net(x)

        out = self.value_head(hidden)

        mean, log_var = out.chunk(2, dim = -1)

        # log_var is clamped for stability
        log_var = log_var.clamp(min = self.min_log_var, max = self.max_log_var)
        std = (log_var / 2).exp()

        return mean, std

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
        downweight_advantages_with_value_uncertainty = False,
        use_relative_value_uncertainty = False,
        beta = 0.,
        eps = 1e-6,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1000
        ),
        save_path = './ppo.pt'
    ):
        super().__init__()

        self.actor = Actor(state_dim, actor_hidden_dim, num_actions)

        self.critic = Critic(state_dim, critic_hidden_dim, num_actions)

        # weight tie rsmnorm

        self.rsmnorm = self.actor.rsmnorm
        self.critic.rsmnorm = self.rsmnorm

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

        self.downweight_advantages_with_value_uncertainty = downweight_advantages_with_value_uncertainty
        self.use_relative_value_uncertainty = use_relative_value_uncertainty
        self.beta = beta
        self.eps = eps

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

                returns = calc_gae(
                    rewards = rewards,
                    masks = masks,
                    lam = self.lam,
                    gamma = self.gamma,
                    values = values,
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
            to_named_tuple = ('state', 'action', 'action_log_prob', 'returns', 'value', 'value_std', 'past_action'),
            timestep_level = True,
            device = device
        )

        # policy phase training, similar to original PPO

        self.actor.train()
        self.critic.train()

        for _ in range(self.epochs):
            for _, (states, actions, old_log_probs, returns, old_values, old_value_stds, past_action) in enumerate(dl):

                action_probs = self.actor(states)
                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                # calculate clipped surrogate objective, classic PPO loss

                ratios = (action_log_probs - old_log_probs).exp()

                advantages = returns - old_values.detach()

                if self.downweight_advantages_with_value_uncertainty:
                    uncertainty = old_value_stds.detach()
                    if self.use_relative_value_uncertainty:
                        uncertainty = uncertainty / (old_values.abs().detach() + self.eps)

                    factor = 2 * torch.sigmoid(-uncertainty)
                    advantages = advantages * factor

                advantages = normalize(advantages)

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

                update_network_(policy_loss, self.opt_actor)

                def update_critic(critic, old_values, opt_critic):
                    # calculate Gaussian NLL loss and update value network separate from policy network

                    mean, std = critic(states, past_action)

                    dist = Normal(mean, std)

                    # negative log likelihood of simple gaussian

                    log_prob = dist.log_prob(returns)

                    if self.beta > 0:
                        # Beta-NLL (Seitzer et al. 2022)
                        # weight the NLL loss by (variance)^beta to mitigate the effect of outliers
                        log_prob = log_prob * (std.detach() ** 2).pow(self.beta)

                    value_loss = -log_prob.mean()

                    update_network_(value_loss, opt_critic)

                update_critic(self.critic, old_values, self.opt_critic)

        # update the state normalization with rsmnorm for 1 epoch after actor critic are updated

        self.rsmnorm.train()

        for states, *_ in dl:
            self.rsmnorm(states)

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 500,
    actor_hidden_dim = 64,
    critic_hidden_dim = 256,
    update_timesteps = 5000,
    buffer_episodes = 100,
    num_envs = 8,
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
    video_folder = './probabilistic-value-recording-uncertainty',
    downweight_advantages_with_value_uncertainty = True,
    use_relative_value_uncertainty = True,
    beta = 0.,
    uncertainty_exploration_bonus_weight = 0.,
    eps = 1e-6,
    load = False
):
    def make_env(rank):
        def _init():
            env = gym.make(env_name, render_mode = 'rgb_array')
            if render and rank == 0:
                if clear_videos:
                    rmtree(video_folder, ignore_errors = True)

                env = gym.wrappers.RecordVideo(
                    env = env,
                    video_folder = video_folder,
                    name_prefix = 'lunar-video',
                    episode_trigger = lambda eps_num: divisible_by(eps_num, max(1, render_every_eps // num_envs)),
                    disable_logger = True
                )
            return env
        return _init

    env = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])

    state_dim = int(env.single_observation_space.shape[0])
    num_actions = int(env.single_action_space.n)

    memories = ReplayBuffer(
        './lunar-lander-memories/probabilistic-value',
        max_episodes = buffer_episodes,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            learnable = 'bool',
            state = ('float', state_dim),
            action = 'int',
            action_log_prob = 'float',
            reward = 'float',
            is_boundary = 'bool',
            value = 'float',
            value_std = 'float',
            returns = 'float',
            past_action = ('float', num_actions)
        ),
        circular = True,
        overwrite = True
    )

    agent = PPO(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
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
        asymmetric_spo,
        downweight_advantages_with_value_uncertainty,
        use_relative_value_uncertainty,
        beta = beta,
        eps = eps
    ).to(device)

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0
    total_eps = 0

    rolling_reward = deque(maxlen = 100)
    rolling_steps = deque(maxlen = 100)
    rolling_uncertainty = deque(maxlen = 100)

    states, _ = env.reset(seed = seed)
    states = torch.from_numpy(states).to(device)

    past_actions = torch.zeros(num_envs, num_actions).to(device)

    eps_rewards = np.zeros(num_envs)
    eps_steps = np.zeros(num_envs)

    pbar = tqdm(total = num_episodes, desc = 'episodes')

    while total_eps < num_episodes:

        with memories.batched_episode(batch_size = num_envs):
            for timestep in range(max_timesteps):
                time += num_envs

                action_probs = agent.ema_actor.forward_eval(states)
                means, stds = agent.ema_critic.forward_eval(states, past_actions)

                dist = Categorical(action_probs)
                actions = dist.sample()
                action_log_probs = dist.log_prob(actions)

                next_states, environment_rewards, terminateds, truncateds, infos = env.step(actions.cpu().numpy())

                rewards = torch.from_numpy(environment_rewards).float()

                if uncertainty_exploration_bonus_weight > 0:
                    bonus = calculate_uncertainty_exploration_bonus(
                        means.squeeze(),
                        stds.squeeze(),
                        eps = eps
                    )
                    rewards = rewards + uncertainty_exploration_bonus_weight * bonus.cpu()

                next_states = torch.from_numpy(next_states).to(device)

                eps_rewards += environment_rewards
                eps_steps += 1

                rolling_uncertainty.extend(stds.squeeze().cpu().tolist())

                memories.store_batch(
                    learnable = torch.ones(num_envs, dtype = torch.bool),
                    state = states,
                    action = actions,
                    action_log_prob = action_log_probs,
                    reward = rewards,
                    is_boundary = torch.from_numpy(terminateds),
                    value = means.squeeze(),
                    value_std = stds.squeeze(),
                    past_action = past_actions
                )

                states = next_states
                past_actions = F.one_hot(actions, num_classes = num_actions).float()

                # check for completed episodes in the vector

                completed_indices = np.where(terminateds | truncateds)[0]

                if completed_indices.size > 0:
                    rolling_reward.extend(eps_rewards[completed_indices].tolist())
                    rolling_steps.extend(eps_steps[completed_indices].tolist())

                    eps_rewards[completed_indices] = 0.
                    eps_steps[completed_indices] = 0

                    total_eps += completed_indices.size
                    pbar.update(completed_indices.size)

                    pbar.set_postfix(
                        reward = f'{np.mean(rolling_reward):.2f}',
                        steps = f'{np.mean(rolling_steps):.1f}',
                        uncertainty = f'{np.mean(rolling_uncertainty):.4f}'
                    )

                # determine if updating agent
                updating_agent = time >= (num_policy_updates + 1) * update_timesteps

                if updating_agent:
                    # bootstrap for all envs
                    # but only if we have space in the buffer for one more step
                    if memories.timestep_index < max_timesteps:
                        next_means, next_stds = agent.ema_critic.forward_eval(states, past_actions)

                        memories.store_batch(
                            learnable = torch.zeros(num_envs, dtype = torch.bool),
                            state = states,
                            action = actions, # dummy
                            action_log_prob = action_log_probs, # dummy
                            reward = torch.zeros(num_envs), # dummy
                            is_boundary = torch.ones(num_envs, dtype = torch.bool),
                            value = next_means.squeeze(),
                            value_std = next_stds.squeeze(),
                            past_action = past_actions
                        )

                    break

        if updating_agent:
            agent.learn(memories, device)
            num_policy_updates += 1
            memories.clear()

        if divisible_by(total_eps, save_every) and total_eps > 0:
            agent.save()

        if total_eps >= num_episodes:
            break

if __name__ == '__main__':
    fire.Fire(main)
