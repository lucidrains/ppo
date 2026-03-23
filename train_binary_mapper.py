# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "numpy",
#     "tqdm",
#     "einops",
#     "fire",
#     "gymnasium[box2d]",
#     "gymnasium[other]",
#     "pygame",
#     "assoc-scan",
#     "vector-quantize-pytorch>=1.28.0",
# ]
# ///

from __future__ import annotations

import fire
import numpy as np
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree
from collections import deque, namedtuple

import torch
from torch import nn, tensor, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

from einops import reduce, rearrange
from assoc_scan import AssocScan
import gymnasium as gym

from vector_quantize_pytorch import BinaryMapper

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'is_boundary', 'value'])

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

# modules

class RSMNorm(Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_variance', torch.ones(dim))

    def forward(self, x):
        time = self.step.item()
        mean, var = self.running_mean, self.running_variance

        normed = (x - mean) / var.sqrt().clamp(min = self.eps)

        if not self.training:
            return normed

        with torch.no_grad():
            new_obs_mean = reduce(x, '... d -> d', 'mean')
            delta = new_obs_mean - mean

            new_mean = mean + delta / time
            new_var = (time - 1) / time * (var + (delta ** 2) / time)

            self.step.add_(1)
            self.running_mean.copy_(new_mean)
            self.running_variance.copy_(new_var)

        return normed

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
        expansion_factor = 2
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim * expansion_factor)
        dim_inner = dim_hidden * expansion_factor

        self.proj_in = nn.Linear(dim, dim_hidden)

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.RMSNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_inner),
                ReluSquared(),
                nn.Linear(dim_inner, dim_hidden),
                nn.Dropout(dropout)
            ))

        self.final_norm = nn.RMSNorm(dim_hidden)

    def forward(self, x):
        no_batch = x.ndim == 1
        x = rearrange(x, 'd -> 1 d') if no_batch else x

        x = self.proj_in(x)
        for layer in self.layers:
            x = x + layer(x)

        out = self.final_norm(x)
        return rearrange(out, '1 d -> d') if no_batch else out

# networks

class Actor(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_actions,
        mlp_depth = 2,
        dropout = 0.1
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim)
        self.net = SimBa(state_dim, dim_hidden = hidden_dim * 2, depth = mlp_depth, dropout = dropout)

        self.bits = int(np.log2(num_actions))
        self.mapper = BinaryMapper(bits = self.bits)

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            ReluSquared(),
            nn.Linear(hidden_dim, self.bits)
        )

    def forward(self, x):
        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        hidden = self.net(x)
        return self.action_head(hidden)

class Critic(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        mlp_depth = 3,
        dropout = 0.1
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim)
        self.net = SimBa(state_dim, dim_hidden = hidden_dim, depth = mlp_depth, dropout = dropout)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        hidden = self.net(x)
        return self.value_head(hidden).squeeze(-1)

# ppo trainer

def calc_gae(rewards, values, masks, gamma = 0.99, lam = 0.95):
    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[:-1], values[1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = False)
    return scan(gates, delta) + values

class PPO(Module):
    def __init__(
        self,
        *,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        minibatch_size,
        lr,
        lam,
        gamma,
        beta_s,
        eps_clip
    ):
        super().__init__()
        self.actor = Actor(state_dim, actor_hidden_dim, num_actions)
        self.critic = Critic(state_dim, critic_hidden_dim)

        # weight tie rsmnorm
        self.critic.rsmnorm = self.actor.rsmnorm

        self.opt_actor = AdamW(self.actor.parameters(), lr = lr)
        self.opt_critic = AdamW(self.critic.parameters(), lr = lr)

        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s
        self.eps_clip = eps_clip

    def learn(self, memories):
        states, actions, old_log_probs, rewards, is_boundaries, values = zip(*memories)

        # data prep

        states = stack(states).to(device).detach()
        actions = tensor(actions, device = device).detach()
        old_log_probs = stack(old_log_probs).to(device).detach()

        rewards = tensor(rewards, device = device)
        masks = tensor([(1. - float(is_term)) for is_term in is_boundaries], device = device)
        values = stack(values).to(device)

        with torch.no_grad():
            returns = calc_gae(rewards, values, masks, self.gamma, self.lam)

        old_values = values.detach()

        # dataloader

        dataset = TensorDataset(states, actions, old_log_probs, returns, old_values)
        dl = DataLoader(dataset, batch_size = self.minibatch_size, shuffle = True)

        # optimize

        for _ in range(self.epochs):
            for states_b, actions_b, old_log_probs_b, returns_b, old_values_b in dl:

                # actor update

                logits = self.actor(states_b)
                action_log_probs = self.actor.mapper.log_prob(logits, indices = actions_b, sum_bits = True)
                entropy = self.actor.mapper.binary_entropy(logits).mean()

                ratios = (action_log_probs - old_log_probs_b).exp()
                advantages = normalize(returns_b - old_values_b.detach())

                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.beta_s * entropy

                self.opt_actor.zero_grad()
                policy_loss.backward()
                self.opt_actor.step()

                # critic update

                values_pred = self.critic(states_b)
                critic_loss = F.mse_loss(values_pred, returns_b)

                self.opt_critic.zero_grad()
                critic_loss.backward()
                self.opt_critic.step()

        # update rsmnorm state

        self.actor.rsmnorm.train()
        for states_b, *_ in dl:
            self.actor.rsmnorm(states_b)

# main rollout script

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 5000,
    max_timesteps = 500,
    actor_hidden_dim = 64,
    critic_hidden_dim = 256,
    minibatch_size = 64,
    lr = 3e-4,
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    beta_s = 0.01,
    update_timesteps = 4000,
    epochs = 10,
    seed = None,
    render = True,
    render_every_eps = 250,
    clear_videos = True,
    video_folder = './lunar-recording'
):
    # env setup

    env = gym.make(env_name, render_mode = 'rgb_array' if render else None)

    if render:
        if clear_videos:
            rmtree(video_folder, ignore_errors = True)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = 'lunar-video',
            episode_trigger = lambda eps: divisible_by(eps, render_every_eps),
            disable_logger = True
        )

    # init agent

    agent = PPO(
        state_dim = env.observation_space.shape[0],
        num_actions = env.action_space.n,
        actor_hidden_dim = actor_hidden_dim,
        critic_hidden_dim = critic_hidden_dim,
        epochs = epochs,
        minibatch_size = minibatch_size,
        lr = lr,
        lam = lam,
        gamma = gamma,
        beta_s = beta_s,
        eps_clip = eps_clip
    ).to(device)

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    # training loop

    time = 0
    memories = []
    recent_rewards = deque(maxlen = 100)

    pbar = tqdm(range(num_episodes), desc = 'episodes')

    for eps in pbar:
        state, _ = env.reset(seed = seed)
        state_t = tensor(state, dtype = torch.float32, device = device)
        eps_reward = 0.

        for _ in range(max_timesteps):
            time += 1

            agent.eval()

            with torch.no_grad():
                logits = agent.actor(state_t)
                value = agent.critic(state_t)

                _, action_tensor, _ = agent.actor.mapper(logits, deterministic = False, return_indices = True)
                action_log_prob = agent.actor.mapper.log_prob(logits, indices = action_tensor, sum_bits = True)

            action = action_tensor.item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_t = tensor(next_state, dtype = torch.float32, device = device)

            eps_reward += reward
            done = terminated or truncated

            memories.append(Memory(state_t, action, action_log_prob, float(reward), done, value))
            state_t = next_state_t

            if divisible_by(time, update_timesteps):
                agent.train()
                agent.learn(memories)
                memories.clear()

            if done:
                break

        recent_rewards.append(eps_reward)
        pbar.set_postfix(reward = f"{sum(recent_rewards) / len(recent_rewards):.2f}")

if __name__ == '__main__':
    fire.Fire(main)
