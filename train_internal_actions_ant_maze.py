# /// script
# dependencies = [
#   "torch",
#   "einops",
#   "ema-pytorch",
#   "adam-atan2-pytorch",
#   "hl-gauss-pytorch",
#   "assoc-scan",
#   "gymnasium[mujoco]>=1.0.0",
#   "gymnasium-robotics",
#   "memmap-replay-buffer",
#   "fire",
#   "tqdm",
#   "termcolor"
# ]
# ///

from __future__ import annotations
import re

import fire
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from itertools import zip_longest, cycle
from collections import deque

import numpy as np
from tqdm import tqdm
from termcolor import colored

import torch
from torch import nn, tensor, cat, stack, is_tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.distributions import Categorical, Normal

from einops import reduce, repeat, rearrange, pack

from ema_pytorch import EMA

from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from hl_gauss_pytorch import HLGaussLoss

from assoc_scan import AssocScan

import gymnasium as gym
import gymnasium_robotics

from memmap_replay_buffer import ReplayBuffer

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def exists(val):
    return val is not None

def softclamp(t, value):
    return (t / value).tanh() * value

def default(v, d):
    return v if exists(v) else d

def maybe(fn):
    return lambda v: fn(v) if exists(v) else v

def module_device(module):
    if is_tensor(module): return module.device
    return next(module.parameters()).device

def divisible_by(num, den):
    return (num % den) == 0

def normalize(t, mask = None, eps = 1e-5):
    if not exists(mask):
        return (t - t.mean()) / t.std(unbiased = False).clamp(min = eps)
    masked_t = t[mask]
    if masked_t.numel() == 0: return t
    mean = masked_t.mean()
    std = masked_t.std(unbiased = False).clamp(min = eps)
    return (t - mean) / std

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

# phase schedule parsing

def parse_string_schedule(schedule_str):
    def expand_match(m):
        inner_content = m.group(1)
        repeats = int(m.group(2))
        return " ".join([inner_content] * repeats)

    schedule_str = re.sub(r'\(([^)]+)\)\s*\*\s*(\d+)', expand_match, schedule_str)

    matches = re.findall(r'(\d+)\s*(both|inner|outer)', schedule_str.lower())
    if not matches:
        raise ValueError(f"Could not parse phase schedule string: {schedule_str}")
    return [(phase, int(duration)) for duration, phase in matches]

def get_phase_generator(schedule, num_envs = 1):
    for phase, duration in cycle(schedule):
        if duration % num_envs != 0:
            raise ValueError(f"Phase '{phase}' duration {duration} is not a multiple of num_envs ({num_envs}). "
                             f"Vectorized rollouts cannot have mixed phases.")
        for _ in range(duration):
            yield phase

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

class SoLU(Module):
    def __init__(
        self,
        dim,
        softmax_segments = 3,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias = False)
        self.softmax_segments = softmax_segments

    def forward(self, x, actions = None):

        logits = rearrange(x, '... (segments d) -> ... segments d', segments = self.softmax_segments)
        prob = logits.softmax(dim = -1)
        prob = rearrange(prob, '... segments d -> ... (segments d)')

        return self.norm(x * prob), (None, None)

class GumbelSoLU(Module):
    def __init__(
        self,
        dim,
        softmax_segments = 3,
    ):
        super().__init__()
        self.softmax_segments = softmax_segments
        self.norm = nn.LayerNorm(dim, bias = False)

    def forward(self, x, actions = None):

        logits = rearrange(x, '... (segments d) -> ... segments d', segments = self.softmax_segments)

        if not exists(actions):
            if self.training:
                dist = Categorical(logits = logits)
                actions = dist.sample()
            else:
                actions = logits.argmax(dim = -1)

        prob = logits.softmax(dim = -1)
        one_hot = F.one_hot(actions.long(), prob.shape[-1]).float()

        # straight through estimator
        one_hot = one_hot + prob - prob.detach()
        one_hot = rearrange(one_hot, '... segments d -> ... (segments d)')

        return self.norm(x * one_hot), (logits, actions)

class FeedForward(Module):
    def __init__(
        self,
        dim,
        expansion_factor,
        gumbel_sample = False,
        dropout = 0.,
        softmax_segments = 3
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.proj_gelu = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim_inner),
        )

        self.proj_solu = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim_inner),
        )

        self.activation = GumbelSoLU(dim_inner, softmax_segments) if gumbel_sample else SoLU(dim_inner, softmax_segments)

        self.values = nn.Sequential(
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
        actions = None
    ):
        gate, hidden = self.proj_gelu(x), self.proj_solu(x)
        hidden, logits_and_actions = self.activation(hidden, actions)

        out = F.gelu(gate) + hidden
        return self.values(out), logits_and_actions

class SimBa(Module):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        depth = 3,
        dropout = 0.,
        expansion_factor = 3,
        ff_solu_gumbel_sample = False,
        num_internal_actions = 4,
        internal_action_dim = 3,
        num_time_embeds = 8
    ):
        super().__init__()
        """
        following the design of SimBa https://arxiv.org/abs/2410.09754v1
        """

        dim_hidden = default(dim_hidden, dim * expansion_factor)

        self.num_time_embeds = num_time_embeds
        self.time_emb = nn.Embedding(num_time_embeds, dim_hidden)

        layers = []

        # networks

        self.proj_in = nn.Linear(dim, dim_hidden)

        for _ in range(depth):
            layer = FeedForward(
                dim_hidden,
                expansion_factor,
                gumbel_sample = ff_solu_gumbel_sample,
                softmax_segments = internal_action_dim
            )
            layers.append(layer)

        # final layer out

        self.layers = ModuleList(layers)

        self.final_norm = nn.RMSNorm(dim_hidden)

    def forward(
        self,
        x,
        actions = None,
        time_embed_index = None
    ):
        no_batch = x.ndim == 1

        if no_batch:
            x = rearrange(x, '... -> 1 ...')

        x = self.proj_in(x)

        if exists(time_embed_index):
            assert (time_embed_index < self.num_time_embeds).all(), 'time indices passed in must be never greater than or equal to num time embeds'
            x = x + self.time_emb(time_embed_index)

        logits_and_actions = []

        if not exists(actions):
            actions = tuple()
        else:
            actions = actions.unbind(dim = 1)

        for layer, layer_actions in zip_longest(self.layers, actions):
            x, one_logits_and_actions = layer(x, layer_actions)

            logits_and_actions.append(one_logits_and_actions)

        out = self.final_norm(x)

        if no_batch:
            out = rearrange(out, '1 ... -> ...')

        return out, logits_and_actions

# networks

class Actor(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        mlp_depth = 6,
        dropout = 0.1,
        rsmnorm_input = True,
        num_internal_actions = 4,
        internal_action_dim = 3,
        num_time_embeds = 8
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim,
            dim_hidden = hidden_dim * 2,
            depth = mlp_depth,
            dropout = dropout,
            ff_solu_gumbel_sample = True,
            num_internal_actions = num_internal_actions,
            internal_action_dim = internal_action_dim,
            num_time_embeds = num_time_embeds
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim * 2)
        )

    def forward(
        self,
        x,
        actions = None,
        time_embed_index = None
    ):
        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        hidden, logits_and_actions = self.net(x, actions = actions, time_embed_index = time_embed_index)
        action_logits = self.action_head(hidden)

        mean, log_var = action_logits.chunk(2, dim = -1)
        std = (0.5 * softclamp(log_var, 5.)).exp()

        return mean, std, logits_and_actions

class Critic(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        dim_pred = 1,
        mlp_depth = 6, # recent paper has findings that show scaling critic is more important than scaling actor
        dropout = 0.1,
        rsmnorm_input = True,
        num_internal_actions = 4,
        internal_action_dim = 3,
        num_time_embeds = 8
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim,
            dim_hidden = hidden_dim,
            depth = mlp_depth,
            dropout = dropout,
            ff_solu_gumbel_sample = False,
            num_internal_actions = num_internal_actions,
            internal_action_dim = internal_action_dim,
            num_time_embeds = num_time_embeds
        )

        self.value_head = nn.Linear(hidden_dim, dim_pred)
        self.internal_value_head = nn.Linear(hidden_dim, dim_pred)

    def forward(self, x, time_embed_index = None):

        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        hidden, _ = self.net(x, time_embed_index = time_embed_index)
        return self.value_head(hidden), self.internal_value_head(hidden)

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

def calc_internal_gae(
    rewards,
    values,
    masks,
    is_sampled,
    learnable,
    gamma,
    lam,
    num_time_embeds,
    use_accelerated = None
):
    b, seq_len = rewards.shape
    device = rewards.device
    use_accelerated = default(use_accelerated, rewards.is_cuda)
    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    is_boundary = (masks == 0)
    is_truncated = is_boundary & (~learnable)
    step_rewards = rewards + torch.where(is_truncated, values, torch.zeros_like(values))

    is_sampled_next = is_sampled.roll(-1, dims = 1)
    is_sampled_next[:, -1] = True

    # bucket rewards

    bucket_reward_gates = gamma * masks * (~is_sampled_next).float()
    bucket_rewards = scan(bucket_reward_gates, step_rewards)

    # bucket masks

    bucket_mask_gates = masks * (~is_sampled_next).float()
    bucket_mask_values = masks * is_sampled_next.float()
    bucket_masks = scan(bucket_mask_gates, bucket_mask_values)

    # next internal values

    next_internal_val_gates = (~is_sampled).float()
    next_internal_val_values = values * is_sampled.float()
    next_internal_values = scan(next_internal_val_gates, next_internal_val_values)

    next_internal_values = F.pad(next_internal_values[:, 1:], (0, 1), value = 0.0)

    # internal gae

    deltas = bucket_rewards + (gamma ** num_time_embeds) * bucket_masks * next_internal_values - values

    next_internal_gae_gates = (gamma ** num_time_embeds) * lam * bucket_masks
    next_internal_gaes = scan(next_internal_gae_gates, deltas)

    internal_returns = next_internal_gaes + values
    return internal_returns

# agent

class PPO(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        reward_range: tuple[float, float],
        epochs,
        minibatch_size,
        gae_batch_size,
        lr,
        betas,
        lam,
        gamma,
        gamma_inner,
        beta_s,
        regen_reg_rate,
        cautious_factor,
        eps_clip,
        value_clip,
        ema_decay,
        internal_policy_loss_weight = 1.0,
        num_internal_actions = 4,
        internal_action_dim = 3,
        num_time_embeds = 8,
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
            mlp_depth = num_internal_actions,
            internal_action_dim = internal_action_dim,
            num_time_embeds = num_time_embeds
        )

        self.critic = Critic(
            state_dim,
            critic_hidden_dim,
            dim_pred = critic_pred_num_bins,
            mlp_depth = num_internal_actions,
            internal_action_dim = internal_action_dim,
            num_time_embeds = num_time_embeds
        )

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
        self.gae_batch_size = gae_batch_size

        self.epochs = epochs

        self.lam = lam
        self.gamma = gamma
        self.gamma_inner = gamma_inner
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.internal_policy_loss_weight = internal_policy_loss_weight
        self.save_path = Path(save_path)

    def save(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path), weights_only = True)

        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

    def learn(self, memories: ReplayBuffer, *, phase = 'both', device = device):
        hl_gauss = self.critic_hl_gauss_loss

        # calculate generalized advantage estimate
        # Since this is a circular buffer, we compute GAE for all valid trajectories.
        # This is very fast when done periodically (e.g. every 5000 steps) rather than every timestep.

        dl = memories.dataloader(
            batch_size = self.gae_batch_size,
            return_indices = True,
            to_named_tuple = ('_index', 'is_boundary', 'value', 'internal_value', 'reward', 'is_internal_action_sampled', 'learnable'),
            device = device
        )

        for indices, is_boundaries, values, internal_values, rewards, is_sampled, learnable in tqdm(dl, desc='calculating GAE', leave=True):
            with torch.no_grad():
                masks = (1. - is_boundaries.float())
                scalar_values = hl_gauss(values.to(device))
                scalar_internal_values = hl_gauss(internal_values.to(device))

                returns = calc_gae(
                    rewards = rewards,
                    masks = masks,
                    lam = self.lam,
                    gamma = self.gamma,
                    values = scalar_values.cpu(),
                    use_accelerated = False
                )

                internal_returns = calc_internal_gae(
                    rewards = rewards.cpu(),
                    values = scalar_internal_values.cpu(),
                    masks = masks.cpu(),
                    is_sampled = is_sampled.cpu(),
                    learnable = learnable.cpu(),
                    gamma = self.gamma_inner,
                    lam = self.lam,
                    num_time_embeds = self.actor.net.num_time_embeds
                )

                memories.data['returns'][indices, :returns.shape[-1]] = returns.cpu().numpy()
                memories.data['internal_returns'][indices, :internal_returns.shape[-1]] = internal_returns.cpu().numpy()

        memories.flush()

        # get data

        dl = memories.dataloader(
            batch_size = self.minibatch_size,
            shuffle = True,
            filter_fields = dict(
                learnable = True
            ),
            to_named_tuple = (
                'state',
                'action',
                'action_log_prob',
                'returns',
                'internal_returns',
                'value',
                'internal_value',
                'internal_actions',
                'internal_action_logits',
                'time_embed_index',
                'is_internal_action_sampled'
            ),
            timestep_level = True,
            device = device
        )

        self.actor.train()
        self.critic.train()

        # policy phase training, similar to original PPO

        for epoch in range(self.epochs):
            for i, (
                states,
                actions,
                old_log_probs,
                returns,
                internal_returns,
                old_values,
                old_internal_values,
                internal_actions,
                old_internal_action_logits,
                time_embed_indices,
                is_internal_action_sampled
            ) in enumerate(tqdm(dl, desc=f'ppo epoch {epoch}', leave=True)):

                mean, std, internals = self.actor(states, internal_actions, time_embed_index = time_embed_indices)
                internal_action_logits = stack(tuple(t[0] for t in internals), dim = 1)

                dist = Normal(mean, std)
                action_log_probs = dist.log_prob(actions).sum(dim = -1)
                entropy = dist.entropy().sum(dim = -1)

                internal_dist = Categorical(logits = internal_action_logits)
                internal_log_probs = internal_dist.log_prob(internal_actions)
                internal_entropy = internal_dist.entropy()

                old_internal_dist = Categorical(logits = old_internal_action_logits)
                old_internal_log_probs = old_internal_dist.log_prob(internal_actions)

                action_log_probs, _ = pack([action_log_probs, internal_log_probs], 'b *')
                old_log_probs, _ = pack([old_log_probs, old_internal_log_probs], 'b *')
                entropy, _ = pack([entropy, internal_entropy], 'b * ')

                # calculate clipped surrogate objective, classic PPO loss

                ratios = (action_log_probs - old_log_probs).exp()

                scalar_old_values = hl_gauss(old_values)
                advantages_main = normalize(returns - scalar_old_values.detach())
                advantages_main = rearrange(advantages_main, '... -> ... 1')

                scalar_old_internal_values = hl_gauss(old_internal_values)
                advantages_inner = normalize(internal_returns - scalar_old_internal_values.detach(), mask = is_internal_action_sampled)
                advantages_inner = rearrange(advantages_inner, '... -> ... 1')
                advantages_inner = repeat(advantages_inner, 'b 1 -> b num', num = internal_log_probs[0].numel())

                advantages, _ = pack([advantages_main, advantages_inner], 'b *')

                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - torch.min(surr1, surr2)

                policy_loss = policy_loss - self.beta_s * entropy

                # weigh internal a bit less

                if phase == 'outer':
                    main_loss_weight = torch.ones((1,), device = device)
                    internal_loss_weight = 0.
                elif phase == 'inner':
                    main_loss_weight = torch.zeros((1,), device = device)
                    internal_loss_weight = self.internal_policy_loss_weight * self.actor.net.num_time_embeds
                elif phase == 'both':
                    main_loss_weight = torch.ones((1,), device = device)
                    internal_loss_weight = self.internal_policy_loss_weight * self.actor.net.num_time_embeds
                else:
                    raise ValueError(f'unknown phase {phase}')

                batch_size = states.shape[0]
                num_packed = policy_loss.shape[-1]
                num_internal_packed = num_packed - 1

                is_sampled = is_internal_action_sampled.float()
                is_sampled = rearrange(is_sampled, 'b -> b 1')

                internal_weight = internal_loss_weight * is_sampled
                internal_weight = repeat(internal_weight, 'b 1 -> b i', i = num_internal_packed)

                main_weight = repeat(main_loss_weight, '1 -> b 1', b = batch_size)

                loss_weight, _ = pack([main_weight, internal_weight], 'b *')

                policy_loss = (policy_loss * loss_weight).sum(dim = -1)

                update_network_(policy_loss, self.opt_actor)

                # calculate clipped value loss and update value network separate from policy network

                clip = self.value_clip

                values, internal_values = self.critic(states, time_embed_index = time_embed_indices)

                scalar_values = hl_gauss(values)
                scalar_internal_values = hl_gauss(internal_values)

                def is_between(mid, lo, hi):
                    return (lo < mid) & (mid < hi)

                def calc_value_loss(scalar_values, scalar_old_values, returns, values):
                    clipped_returns = returns.clamp(scalar_old_values - clip, scalar_old_values + clip)
                    clipped_loss = hl_gauss(values, clipped_returns, reduction = 'none')
                    loss = hl_gauss(values, returns, reduction = 'none')

                    old_values_lo = scalar_old_values - clip
                    old_values_hi = scalar_old_values + clip

                    return torch.where(
                        is_between(scalar_values, returns, old_values_lo) |
                        is_between(scalar_values, old_values_hi, returns),
                        0.,
                        torch.min(loss, clipped_loss)
                    )

                value_loss_main = calc_value_loss(scalar_values, scalar_old_values, returns, values)
                value_loss_inner = calc_value_loss(scalar_internal_values, scalar_old_internal_values, internal_returns, internal_values)

                if phase == 'outer':
                    main_v_weight = 1.
                    inner_v_weight = 0.
                elif phase == 'inner':
                    main_v_weight = 0.
                    inner_v_weight = 1. * self.actor.net.num_time_embeds
                elif phase == 'both':
                    main_v_weight = 1.
                    inner_v_weight = 1. * self.actor.net.num_time_embeds

                is_sampled_float = is_internal_action_sampled.float()

                value_loss = value_loss_main.mean() * main_v_weight + (value_loss_inner * is_sampled_float).mean() * inner_v_weight

                update_network_(value_loss, self.opt_critic)

        # update the state normalization with rsmnorm for 1 epoch after actor critic are updated

        self.rsmnorm.train()

        for states, *_ in dl:
            self.rsmnorm(states)

# helpers for vectorized rollout

def concat_goal_obs(state_dict):
    """Concatenate observation and desired_goal from dict-style gym obs."""
    return np.concatenate([state_dict['observation'], state_dict['desired_goal']], axis = -1)

def record_eval_episode(
    env_name,
    agent,
    max_timesteps,
    num_time_embeds,
    video_folder,
    episode_id,
    clear_previous = False
):
    """Spin up a temporary sync env, run one greedy episode, record video, tear down."""

    if clear_previous:
        rmtree(video_folder, ignore_errors = True)

    eval_env = gym.make(env_name, render_mode = 'rgb_array')

    eval_env = gym.wrappers.RecordVideo(
        eval_env,
        video_folder = video_folder,
        name_prefix = f'antmaze-eval-eps-{episode_id}',
        episode_trigger = lambda _: True,
        disable_logger = True
    )

    state_dict, _ = eval_env.reset()
    state = torch.from_numpy(concat_goal_obs(state_dict)).float().to(device)

    last_internal_actions = None

    for timestep in range(max_timesteps):
        time_idx = timestep % num_time_embeds
        is_sampled = (time_idx == 0)

        with torch.no_grad():
            time_index = tensor([time_idx], device = device, dtype = torch.long)
            state_batch = rearrange(state, 'd -> 1 d')

            prev_actions = last_internal_actions if not is_sampled else None
            mean, std, internals = agent.ema_actor.forward_eval(state_batch, actions = prev_actions, time_embed_index = time_index)

        internal_action_logits, internal_actions = tuple(rearrange(stack(t), 'l b ... -> b l ...') for t in zip(*internals))

        if is_sampled or not exists(last_internal_actions):
            last_internal_actions = internal_actions.clone()

        action = mean[0].tanh().cpu().numpy()
        next_state_dict, _, terminated, truncated, _ = eval_env.step(action)

        if terminated or truncated:
            break

        state = torch.from_numpy(concat_goal_obs(next_state_dict)).float().to(device)

    eval_env.close()

def extract_final_obs(infos, idx, num_envs):
    """Extract the final observation dict for a truncated env from vectorized infos."""
    if 'final_observation' not in infos:
        return None

    final = infos['final_observation']

    if not isinstance(final, (list, tuple, np.ndarray)):
        return None

    entry = final[idx]

    if not isinstance(entry, dict) or 'observation' not in entry:
        return None

    return entry

def collect_vectorized_rollouts(
    env,
    agent,
    num_envs,
    max_timesteps,
    num_time_embeds,
    memories,
    seed = None
):
    """Run one batch of num_envs episodes in parallel, return (cum_rewards, steps)."""

    # reset

    state_dict, _ = env.reset(seed = seed)
    state = torch.from_numpy(concat_goal_obs(state_dict)).float().to(device)

    # per-env bookkeeping

    env_active = np.ones(num_envs, dtype = bool)
    cum_rewards = np.zeros(num_envs)
    env_steps = np.zeros(num_envs, dtype = int)

    episode_fields = (
        'learnable', 'state', 'action', 'action_log_prob', 'reward',
        'is_boundary', 'value', 'internal_value', 'internal_actions', 'internal_action_logits',
        'time_embed_index', 'is_internal_action_sampled'
    )

    collected = [{field: [] for field in episode_fields} for _ in range(num_envs)]

    last_internal_actions = None
    last_internal_logits = None
    time_elapsed = 0

    # rollout loop

    for timestep in range(max_timesteps):
        if not env_active.any():
            break

        time_elapsed += 1
        time_idx = timestep % num_time_embeds
        is_sampled = (time_idx == 0)

        # batched forward pass

        with torch.no_grad():
            time_index = tensor([time_idx] * num_envs, device = device, dtype = torch.long)

            prev_actions = last_internal_actions if not is_sampled else None
            mean, std, internals = agent.ema_actor.forward_eval(state, actions = prev_actions, time_embed_index = time_index)
            value, internal_value = agent.ema_critic.forward_eval(state, time_embed_index = time_index)

        dist = Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim = -1)

        internal_action_logits_stacked, internal_actions_stacked = tuple(rearrange(stack(t), 'l b ... -> b l ...') for t in zip(*internals))

        if is_sampled or not exists(last_internal_actions):
            last_internal_actions = internal_actions_stacked.clone()
            last_internal_logits = internal_action_logits_stacked.clone()

        # step environment

        env_action = action.tanh().cpu().numpy()

        next_state_dict, reward_np, terminated, truncated, infos = env.step(env_action)
        next_state = torch.from_numpy(concat_goal_obs(next_state_dict)).float().to(device)

        # reward shaping

        exp_bonus = (std.mean(dim = -1) * 0.01).cpu().numpy()
        extreme_penalty = ((mean.abs() > 1.).float().mean(dim = -1) * 0.01).cpu().numpy()
        total_reward = reward_np.astype(float) + exp_bonus - extreme_penalty

        done = terminated | truncated

        # scatter into per-env episode buffers

        for i in range(num_envs):
            if not env_active[i]:
                continue

            cum_rewards[i] += total_reward[i]
            env_steps[i] += 1

            d = collected[i]
            d['learnable'].append(True)
            d['state'].append(state[i].cpu())
            d['action'].append(action[i].cpu())
            d['action_log_prob'].append(action_log_prob[i].item())
            d['reward'].append(total_reward[i])
            d['is_boundary'].append(terminated[i])
            d['value'].append(value[i].cpu())
            d['internal_value'].append(internal_value[i].cpu())
            d['internal_actions'].append(last_internal_actions[i].cpu())
            d['internal_action_logits'].append(last_internal_logits[i].cpu())
            d['time_embed_index'].append(time_idx)
            d['is_internal_action_sampled'].append(is_sampled)

            if not done[i]:
                continue

            env_active[i] = False

            # for truncated (not terminated), store a bootstrap value

            if terminated[i] or len(d['state']) > memories.max_timesteps:
                continue

            final_obs = extract_final_obs(infos, i, num_envs)

            if final_obs is None:
                continue

            final_state_np = concat_goal_obs({'observation': final_obs['observation'], 'desired_goal': final_obs['desired_goal']})
            final_state_t = torch.from_numpy(final_state_np).float().to(device).unsqueeze(0)
            next_time_idx = (timestep + 1) % num_time_embeds

            with torch.no_grad():
                next_value, next_internal_value = agent.ema_critic.forward_eval(
                    final_state_t,
                    time_embed_index = tensor([next_time_idx], device = device, dtype = torch.long)
                )
                next_value, next_internal_value = next_value[0], next_internal_value[0]

            d['learnable'].append(False)
            d['state'].append(final_state_t[0].cpu())
            d['action'].append(action[i].cpu())
            d['action_log_prob'].append(0.)
            d['reward'].append(0.)
            d['is_boundary'].append(True)
            d['value'].append(next_value.cpu())
            d['internal_value'].append(next_internal_value.cpu())
            d['internal_actions'].append(last_internal_actions[i].cpu())
            d['internal_action_logits'].append(last_internal_logits[i].cpu())
            d['time_embed_index'].append(next_time_idx)
            d['is_internal_action_sampled'].append(False)

        state = next_state

    # flush collected episodes into the replay buffer

    for i in range(num_envs):
        d = collected[i]
        with memories.one_episode():
            for t in range(len(d['state'])):
                memories.store(
                    learnable = d['learnable'][t],
                    state = d['state'][t],
                    action = d['action'][t],
                    action_log_prob = d['action_log_prob'][t],
                    reward = d['reward'][t],
                    is_boundary = bool(d['is_boundary'][t]),
                    value = d['value'][t],
                    internal_value = d['internal_value'][t],
                    internal_actions = d['internal_actions'][t],
                    internal_action_logits = d['internal_action_logits'][t],
                    time_embed_index = int(d['time_embed_index'][t]),
                    is_internal_action_sampled = bool(d['is_internal_action_sampled'][t])
                )

    return cum_rewards, env_steps

# main

def main(
    num_envs = 8,
    env_name = 'AntMaze_UMaze-v5',
    phase_schedule = "256both (128inner 128outer)*500",
    max_timesteps = 700,
    actor_hidden_dim = 64,
    critic_hidden_dim = 256,
    critic_pred_num_bins = 100,
    reward_range = (-50., 50.),
    update_episodes = 50,
    buffer_episodes = 100,
    minibatch_size = 1024,
    gae_batch_size = 256,
    lr = 0.0008,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    gamma_inner = 0.99,
    eps_clip = 0.2,
    value_clip = 0.4,
    beta_s = .01,
    regen_reg_rate = 1e-4,
    cautious_factor = 0.1,
    ema_decay = 0.9,
    epochs = 2,
    num_internal_actions = 8,
    internal_action_dim = 3,
    num_time_embeds = 4,
    seed = None,
    render = True,
    save_every = 1000,
    video_folder = './antmaze-recording',
    load = False
):
    # environment setup - always async for training

    env = gym.make_vec(env_name, num_envs = num_envs, vectorization_mode = 'async')

    temp_env = gym.make(env_name)
    state_dim = temp_env.observation_space['observation'].shape[0] + temp_env.observation_space['desired_goal'].shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()

    # replay buffer

    memories = ReplayBuffer(
        './antmaze-memories/internal-actions',
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
            internal_value = ('float', critic_pred_num_bins),
            returns = 'float',
            internal_returns = 'float',
            internal_actions = ('int', (num_internal_actions, internal_action_dim)),
            internal_action_logits = ('float', (num_internal_actions, internal_action_dim, actor_hidden_dim * 2)),
            time_embed_index = 'int',
            is_internal_action_sampled = 'bool'
        ),
        circular = True,
        overwrite = True
    )

    # agent

    agent = PPO(
        state_dim, action_dim,
        actor_hidden_dim, critic_hidden_dim,
        critic_pred_num_bins, reward_range,
        epochs, minibatch_size, gae_batch_size,
        lr, betas, lam, gamma, gamma_inner, beta_s,
        regen_reg_rate, cautious_factor,
        eps_clip, value_clip, ema_decay,
        num_internal_actions = num_internal_actions,
        internal_action_dim = internal_action_dim,
        num_time_embeds = num_time_embeds
    ).to(device)

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    # training state

    num_policy_updates = 0

    running_rewards = deque(maxlen = 100)
    running_steps = deque(maxlen = 100)

    if isinstance(phase_schedule, str):
        phase_schedule_tuples = parse_string_schedule(phase_schedule)
    else:
        phase_schedule_tuples = phase_schedule

    num_episodes = sum(duration for _, duration in phase_schedule_tuples)
    phase_generator = get_phase_generator(phase_schedule_tuples, num_envs=num_envs)

    curr_phase = None
    phase_update_count = 0
    total_eps = 0

    print(colored(f'\nLearning Frequency set to 1 update every {update_episodes} episodes', 'cyan', attrs=['bold']))
    print(colored(f'Latent Actions emitted every {num_time_embeds} steps\n', 'cyan', attrs=['bold']))

    pbar = tqdm(total = num_episodes, desc = 'episodes')

    # training loop
    prev_eps = 0

    while total_eps < num_episodes:
        for _ in range(num_envs):
            next_phase = next(phase_generator)

            if next_phase != curr_phase:
                if exists(curr_phase):
                    print(f'\n[Phase Change] Now switching to optimization phase: {next_phase}')
                curr_phase = next_phase
                phase_update_count = 0
                pbar.set_description(f'episodes (phase: {curr_phase})')

        # collect num_envs episodes in parallel

        cum_rewards, steps = collect_vectorized_rollouts(
            env, agent, num_envs, max_timesteps,
            num_time_embeds, memories, seed
        )

        running_rewards.extend(cum_rewards)
        running_steps.extend(steps)

        total_eps += num_envs
        pbar.update(num_envs)

        # periodic learning

        updating_agent = (total_eps // update_episodes) > (prev_eps // update_episodes)

        if len(running_rewards) > 0:
            avg_reward = sum(running_rewards) / len(running_rewards)
            avg_steps = sum(running_steps) / len(running_steps)
            pbar.set_postfix({
                'reward': round(avg_reward, 2),
                'phase_upd': phase_update_count,
                'steps': round(avg_steps, 2)
            })

        if updating_agent:

            if render:
                record_eval_episode(
                    env_name, agent, max_timesteps,
                    num_time_embeds, video_folder,
                    total_eps,
                    clear_previous = (num_policy_updates == 0)
                )

            agent.learn(memories, phase = curr_phase)
            num_policy_updates += 1
            phase_update_count += 1

        if divisible_by(total_eps, save_every) and total_eps > 0:
            agent.save()

        prev_eps = total_eps

if __name__ == '__main__':
    fire.Fire(main)
