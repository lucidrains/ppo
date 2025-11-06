from __future__ import annotations

import fire
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from functools import partial
from itertools import zip_longest
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

from assoc_scan import AssocScan

import gymnasium as gym

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# memory tuple

Memory = namedtuple('Memory', [
    'learnable',
    'state',
    'action',
    'action_log_prob',
    'reward',
    'is_boundary',
    'value',
    'internal_actions',
    'internal_action_logits'
])

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
            one_hot = F.gumbel_softmax(logits, hard = True, dim = -1)
            actions = one_hot.argmax(dim = -1)
        else:
            prob = logits.softmax(dim = -1)
            one_hot = F.one_hot(actions, prob.shape[-1]).float()
            one_hot = one_hot + prob - prob.detach()

        one_hot = rearrange(one_hot, '... segments d -> ... (segments d)')
        output = self.norm(x * one_hot)

        return output, (logits, actions)

class FeedForward(Module):
    def __init__(
        self,
        dim,
        expansion_factor,
        gumbel_sample = False,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.keys = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim_inner),
        )

        self.activation = GumbelSoLU(dim_inner, expansion_factor) if gumbel_sample else SoLU(dim_inner, expansion_factor)

        self.values = nn.Sequential(
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries,
        actions = None
    ):
        
        sim = self.keys(queries)

        attn, logits_and_actions = self.activation(sim, actions)

        return self.values(attn), logits_and_actions

class SimBa(Module):

    def __init__(
        self,
        dim,
        dim_hidden = None,
        depth = 3,
        dropout = 0.,
        expansion_factor = 3,
        ff_solu_gumbel_sample = False
    ):
        super().__init__()
        """
        following the design of SimBa https://arxiv.org/abs/2410.09754v1
        """

        dim_hidden = default(dim_hidden, dim * expansion_factor)

        layers = []

        self.proj_in = nn.Linear(dim, dim_hidden)

        for ind in range(depth):

            layer = FeedForward(dim_hidden, expansion_factor, gumbel_sample = ff_solu_gumbel_sample)
            layers.append(layer)

        # final layer out

        self.layers = ModuleList(layers)

        self.final_norm = nn.RMSNorm(dim_hidden)

    def forward(
        self,
        x,
        actions = None
    ):
        no_batch = x.ndim == 1

        if no_batch:
            x = rearrange(x, '... -> 1 ...')

        x = self.proj_in(x)

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
        num_actions,
        mlp_depth = 4,
        dropout = 0.1,
        rsmnorm_input = True  # use the RSMNorm for inputs proposed by KAIST + SonyAI
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim,
            dim_hidden = hidden_dim * 2,
            depth = mlp_depth,
            dropout = dropout,
            ff_solu_gumbel_sample = True
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(
        self,
        x,
        actions = None
    ):
        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        hidden, logits_and_actions = self.net(x, actions = actions)

        action_probs = self.action_head(hidden).softmax(dim = -1)

        return action_probs, logits_and_actions

class Critic(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        dim_pred = 1,
        mlp_depth = 6, # recent paper has findings that show scaling critic is more important than scaling actor
        dropout = 0.1,
        rsmnorm_input = True
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim,
            dim_hidden = hidden_dim,
            depth = mlp_depth,
            dropout = dropout,
            ff_solu_gumbel_sample = False
        )

        self.value_head = nn.Linear(hidden_dim, dim_pred)

    def forward(self, x):

        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        hidden, _ = self.net(x)

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
    values, values_next = values[:-1], values[1:]

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
        internal_policy_loss_weight = 0.1,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1000
        ),
        save_path = './ppo.pt'
    ):
        super().__init__()

        self.actor = Actor(state_dim, actor_hidden_dim, num_actions)

        self.critic = Critic(state_dim, critic_hidden_dim, dim_pred = critic_pred_num_bins)

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

    def learn(self, memories):
        hl_gauss = self.critic_hl_gauss_loss

        # retrieve and prepare data from memory for training

        (
            learnable,
            states,
            actions,
            old_log_probs,
            rewards,
            is_boundaries,
            values,
            internal_actions,
            internal_action_logits
        ) = zip(*memories)
        
        actions = [tensor(action) for action in actions]
        masks = [(1. - float(is_boundary)) for is_boundary in is_boundaries]

        # calculate generalized advantage estimate

        scalar_values = hl_gauss(stack(values))

        with torch.no_grad():
            calc_gae_from_values = partial(calc_gae,
                rewards = tensor(rewards).to(device),
                masks = tensor(masks).to(device),
                lam = self.lam,
                gamma = self.gamma,
                use_accelerated = False
            )

            returns = calc_gae_from_values(values = scalar_values)

        # convert values to torch tensors

        to_torch_tensor = lambda t: stack(t).to(device).detach()

        states = to_torch_tensor(states)
        actions = to_torch_tensor(actions)
        old_values = to_torch_tensor(values)
        old_log_probs = to_torch_tensor(old_log_probs)        

        # internal choices

        internal_actions = to_torch_tensor(internal_actions)
        old_internal_action_logits = to_torch_tensor(internal_action_logits)

        # prepare dataloader for policy phase training

        learnable = tensor(learnable).to(device)
        data = (states, actions, old_log_probs, returns, old_values, internal_actions, old_internal_action_logits)
        data = tuple(t[learnable] for t in data)

        dataset = TensorDataset(*data)

        dl = DataLoader(dataset, batch_size = self.minibatch_size, shuffle = True)

        # policy phase training, similar to original PPO

        for _ in range(self.epochs):
            for i, (states, actions, old_log_probs, returns, old_values, internal_actions, old_internal_action_logits) in enumerate(dl):

                action_probs, internals = self.actor(states, internal_actions)
                internal_action_logits = stack(tuple(t[0] for t in internals), dim = 1)

                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                internal_action_probs = internal_action_logits.softmax(dim = -1)
                internal_dist = Categorical(internal_action_probs)
                internal_log_probs = internal_dist.log_prob(internal_actions)
                internal_entropy = internal_dist.entropy()

                old_internal_action_probs = old_internal_action_logits.softmax(dim = -1)
                old_internal_dist = Categorical(old_internal_action_probs)
                old_internal_log_probs = old_internal_dist.log_prob(internal_actions)

                action_log_probs, _ = pack([action_log_probs, internal_log_probs], 'b *')
                old_log_probs, _ = pack([old_log_probs, old_internal_log_probs], 'b *')
                entropy, _ = pack([entropy, internal_entropy], 'b * ')

                # calculate clipped surrogate objective, classic PPO loss

                ratios = (action_log_probs - old_log_probs).exp()

                scalar_old_values = hl_gauss(old_values)
                advantages = normalize(returns - scalar_old_values.detach())
                advantages = rearrange(advantages, '... -> ... 1')

                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - torch.min(surr1, surr2)

                policy_loss = policy_loss - self.beta_s * entropy

                # weigh internal a bit less

                main_loss_weight = torch.ones((1,), device = device)
                loss_weight = F.pad(main_loss_weight, (0, policy_loss.shape[-1] - 1), value = self.internal_policy_loss_weight)

                policy_loss = (policy_loss * loss_weight).sum(dim = -1)

                update_network_(policy_loss, self.opt_actor)

                # calculate clipped value loss and update value network separate from policy network

                clip = self.value_clip

                values = self.critic(states)

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

                update_network_(value_loss, self.opt_critic)

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
    critic_pred_num_bins = 250,
    reward_range = (-100., 100.),
    minibatch_size = 64,
    lr = 0.0008,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    value_clip = 0.4,
    beta_s = .01,
    regen_reg_rate = 1e-4,
    cautious_factor = 0.1,
    ema_decay = 0.9,
    update_timesteps = 5000,
    epochs = 2,
    seed = None,
    render = True,
    render_every_eps = 250,
    save_every = 1000,
    clear_videos = True,
    video_folder = './lunar-recording',
    load = False
):
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

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    memories = deque([])

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

    ).to(device)

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc = 'episodes'):

        state, _ = env.reset(seed = seed)
        state = torch.from_numpy(state).to(device)

        for timestep in range(max_timesteps):
            time += 1

            action_probs, internals = agent.ema_actor.forward_eval(state)
            value = agent.ema_critic.forward_eval(state)

            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()

            next_state, reward, terminated, truncated, _ = env.step(action)

            next_state = torch.from_numpy(next_state).to(device)

            reward = float(reward)

            internal_action_logits, internal_actions = tuple(rearrange(stack(t), 'l 1 ... -> l ...') for t in zip(*internals))

            memory = Memory(True, state, action, action_log_prob, reward, terminated, value, internal_actions, internal_action_logits)

            memories.append(memory)

            state = next_state

            # determine if truncating, either from environment or learning phase of the agent

            updating_agent = divisible_by(time, update_timesteps)
            done = terminated or truncated or updating_agent

            # take care of truncated by adding a non-learnable memory storing the next value for GAE

            if done and not terminated:
                next_value = agent.ema_critic.forward_eval(state)

                bootstrap_value_memory = memory._replace(
                    state = state,
                    learnable = False,
                    is_boundary = True,
                    value = next_value,
                )

                memories.append(bootstrap_value_memory)

            # updating of the agent

            if updating_agent:
                agent.learn(memories)
                num_policy_updates += 1
                memories.clear()

            # break if done

            if done:
                break

        if divisible_by(eps, save_every):
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
