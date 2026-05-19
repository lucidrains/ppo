# /// script
# dependencies = [
#   "accelerate",
#   "adam-atan2-pytorch",
#   "assoc-scan",
#   "ema-pytorch",
#   "fire",
#   "gymnasium[box2d,other]",
#   "hl-gauss-pytorch>=0.1.7",
#   "memmap-replay-buffer>=0.1.2",
#   "numpy",
#   "torch",
#   "tqdm",
#   "x-transformers",
#   "einops"
# ]
# ///

# transformer critic as proposed in T-SAC - Dong Tian et al. https://arxiv.org/abs/2503.03660
# action chunking PPO - Sanghyun Hahn et al. https://openreview.net/forum?id=WFQnqY1c39
# adaptive action chunking - Yongjae Shin et al. https://arxiv.org/abs/2605.10044

from __future__ import annotations

import fire
from pathlib import Path
from shutil import rmtree
from collections import deque
import random

import wandb

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import Categorical

from accelerate import Accelerator

from ema_pytorch import EMA
from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2
from hl_gauss_pytorch import HLGaussLoss

from assoc_scan import AssocScan
from x_transformers import Decoder
from einops import rearrange, repeat
from torch_einops_utils import lens_to_mask

import gymnasium as gym
from memmap_replay_buffer import ReplayBuffer

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def bernoulli(p):
    return random.random() < p

def sample_categorical(logits, low = 1):
    sampled = Categorical(logits = logits).sample()
    return (sampled + 1).clamp(min = low, max = logits.shape[-1]).item()

def normalize(t: Tensor, eps = 1e-5):
    if t.numel() <= 1:
        return torch.zeros_like(t)
    return (t - t.mean()) / (t.std() + eps)

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

# networks

class TransformerPolicy(nn.Module):
    def __init__(
        self,
        *,
        dim_state,
        num_actions,
        dim_hidden = 256,
        depth = 3,
        heads = 8,
        max_chunk_size = 64
    ):
        super().__init__()
        self.state_proj = nn.Linear(dim_state, dim_hidden)
        self.action_proj = nn.Linear(num_actions, dim_hidden)
        self.chunk_queries = nn.Parameter(torch.randn(max_chunk_size, dim_hidden))

        self.decoder = Decoder(
            dim = dim_hidden,
            depth = depth,
            heads = heads,
            attn_dim_head = 32,
            polar_pos_emb = True,
            rotary_pos_emb = False,
            rotary_emb_dim = 32
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, num_actions)
        )

    def forward(
        self,
        state: Tensor,
        past_action: Tensor,
        chunk_size = 1
    ):
        past_action = past_action.float()

        state_tokens = self.state_proj(state)
        action_tokens = self.action_proj(past_action)

        context = torch.stack((state_tokens, action_tokens), dim = 1)

        queries = repeat(self.chunk_queries[:chunk_size], 'n d -> b n d', b = context.shape[0])

        tokens = torch.cat((context, queries), dim = 1)
        out = self.decoder(tokens)

        logits = self.to_logits(out[:, 2:])
        return logits

class TransformerCritic(nn.Module):
    def __init__(
        self,
        *,
        dim_state,
        num_actions,
        dim_hidden = 256,
        depth = 3,
        heads = 8,
        dim_out = 1,
        max_seq_len = 512
    ):
        super().__init__()
        self.dim_out = dim_out
        self.max_seq_len = max_seq_len

        self.state_proj = nn.Linear(dim_state, dim_hidden)
        self.action_proj = nn.Linear(num_actions, dim_hidden)

        self.decoder = Decoder(
            dim = dim_hidden,
            depth = depth,
            heads = heads,
            attn_dim_head = 32,
            polar_pos_emb = True,
            rotary_pos_emb = False,
            rotary_emb_dim = 32
        )

        self.to_values = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(
        self,
        state: Tensor,
        past_action: Tensor
    ):
        past_action = past_action.float()
        has_time = state.ndim == 3

        if not has_time:
            state = rearrange(state, 'b d -> b 1 d')
            past_action = rearrange(past_action, 'b d -> b 1 d')

        b, seq, _ = state.shape
        assert seq <= self.max_seq_len, f'{seq} needs to be less than max seq len {self.max_seq_len}'

        tokens = self.state_proj(state) + self.action_proj(past_action)

        out = self.decoder(tokens)
        values = self.to_values(out)

        if self.dim_out == 1:
            values = rearrange(values, '... 1 -> ...')

        if not has_time:
            values = rearrange(values, 'b 1 ... -> b ...')

        return values

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

    return gae + values

# agent

class PPO(nn.Module):
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
        ema_decay,
        action_chunk_size = 4,
        critic_chunk_size = 16,
        actor_depth = 3,
        critic_depth = 4,
        ema_kwargs: dict = dict(update_every = 10),
        save_path = './ppo.pt'
    ):
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.critic_chunk_size = critic_chunk_size

        self.actor = TransformerPolicy(
            dim_state = state_dim,
            num_actions = num_actions,
            dim_hidden = actor_hidden_dim,
            depth = actor_depth
        )

        self.critic = TransformerCritic(
            dim_state = state_dim,
            num_actions = num_actions,
            dim_hidden = critic_hidden_dim,
            dim_out = critic_pred_num_bins,
            depth = critic_depth,
            max_seq_len = critic_chunk_size + 1
        )

        # hl gauss

        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = reward_range[0],
            max_value = reward_range[1],
            num_bins = critic_pred_num_bins,
            clamp_to_range = True
        )

        # ema

        self.ema_actor = EMA(self.actor, beta = ema_decay, include_online_model = False, **ema_kwargs)
        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False, **ema_kwargs)

        # optimizer

        self.opt_actor = AdoptAtan2(self.actor.parameters(), lr = lr, betas = betas, regen_reg_rate = regen_reg_rate, cautious_factor = cautious_factor)
        self.opt_critic = AdoptAtan2(self.critic.parameters(), lr = lr, betas = betas, regen_reg_rate = regen_reg_rate, cautious_factor = cautious_factor)

        self.ema_actor.add_to_optimizer_post_step_hook(self.opt_actor)
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

        # hparams

        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.save_path = Path(save_path)

    def save(self):
        torch.save(dict(
            actor = self.actor.state_dict(),
            critic = self.critic.state_dict()
        ), str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return
        data = torch.load(str(self.save_path), weights_only = True)
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

    def learn(
        self,
        memories: ReplayBuffer,
        device = None
    ):
        hl_gauss = self.critic_hl_gauss_loss

        # phase 1 - compute GAE across full episodes

        gae_dl = memories.dataloader(
            batch_size = 4,
            return_indices = True,
            return_mask = True,
            to_named_tuple = ('_index', '_mask', 'is_boundary', 'reward', 'state', 'past_action'),
            device = device
        )

        for indices, valid_mask, is_boundaries, rewards, states, past_actions in gae_dl:
            indices = indices.cpu().numpy()

            with torch.no_grad():
                seq = states.shape[1]

                values = torch.cat([
                    self.critic(states[:, i:i + self.critic_chunk_size], past_actions[:, i:i + self.critic_chunk_size])
                    for i in range(0, seq, self.critic_chunk_size)
                ], dim = 1)

                scalar_values = hl_gauss(values)
                masks = (1. - is_boundaries.float()) * valid_mask.float()

                returns = calc_gae(
                    rewards  = rewards * valid_mask.float(),
                    masks    = masks,
                    lam      = self.lam,
                    gamma    = self.gamma,
                    values   = scalar_values * valid_mask.float(),
                    use_accelerated = False
                )

                memories.update(indices, returns = returns, value = values)

        # phase 2 - train on chunks starting at chunk boundaries

        actor_dl = memories.dataloader(
            batch_size      = self.minibatch_size,
            n_steps         = self.action_chunk_size,
            shuffle         = True,
            filter_fields   = dict(chunk_start = True),
            sequence_fields = ('state', 'action', 'action_log_prob', 'returns', 'value', 'past_action'),
            to_named_tuple  = ('seq_state', 'seq_action', 'seq_action_log_prob', 'seq_returns', 'seq_value', 'seq_past_action', 'sampled_chunk_len', 'n_step_lens'),
            device          = device
        )

        critic_dl = memories.dataloader(
            batch_size      = max(1, self.minibatch_size // 4),
            n_steps         = self.critic_chunk_size,
            shuffle         = True,
            sequence_fields = ('state', 'past_action', 'returns'),
            to_named_tuple  = ('seq_state', 'seq_past_action', 'seq_returns', 'n_step_lens'),
            device          = device
        )

        # train

        self.actor.train()
        self.critic.train()

        for _ in range(self.epochs):
            for batch in actor_dl:
                seq = batch.seq_returns.shape[-1]

                valid = lens_to_mask(batch.sampled_chunk_len, seq) & lens_to_mask(batch.n_step_lens, seq)

                action_logits = self.actor(batch.seq_state[:, 0], batch.seq_past_action[:, 0], chunk_size = seq)

                dist = Categorical(logits = action_logits[valid])
                action_log_probs = dist.log_prob(batch.seq_action[valid])
                entropy = dist.entropy()

                advantages = normalize(batch.seq_returns[valid] - hl_gauss(batch.seq_value[valid]).detach())

                ratios = (action_log_probs - batch.seq_action_log_prob[valid]).exp()
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages

                policy_loss = (-torch.min(surr1, surr2) - self.beta_s * entropy).mean()

                update_network_(policy_loss, self.opt_actor)

            for batch in critic_dl:
                valid = lens_to_mask(batch.n_step_lens, batch.seq_returns.shape[-1])

                values = self.critic(batch.seq_state, batch.seq_past_action)
                value_loss = hl_gauss(values[valid], batch.seq_returns[valid]).mean()

                update_network_(value_loss, self.opt_critic)

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 500,
    actor_hidden_dim = 256,
    critic_hidden_dim = 256,
    actor_depth = 3,
    critic_depth = 4,
    action_chunk_size = 4,
    critic_chunk_size = 16,
    update_timesteps = 5000,
    buffer_episodes = 40,
    critic_pred_num_bins = 250,
    reward_range = (-400., 400.),
    minibatch_size = 64,
    lr = 0.0008,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    beta_s = .01,
    regen_reg_rate = 1e-4,
    cautious_factor = 0.1,
    ema_decay = 0.9,
    epochs = 2,
    seed = None,
    dynamic_chunk = False,
    dynamic_chunk_eps = 0.05,
    dynamic_chunk_min_len = 1,
    render = True,
    render_every_eps = 250,
    save_every = 1000,
    clear_videos = True,
    video_folder = './lunar-recording',
    load = False,
    train_cpu = False,
    rollout_cpu = True,
    use_wandb = False,
    wandb_project = 'ppo-action-chunking',
    log_window_size = 20
):
    print('\n' + '=' * 40)
    print(f' PPO Action Chunking')
    print(f' Critic Chunk Size:  {critic_chunk_size}')
    print(f' Action Chunk Size:  {action_chunk_size}')
    print(f' Dynamic Chunk Size: {dynamic_chunk}')
    print(f' Video Directory:    {video_folder}')
    print('=' * 40 + '\n')

    if use_wandb:
        wandb.init(project = wandb_project)

    train_accelerator = Accelerator(cpu = train_cpu)
    train_device = train_accelerator.device
    rollout_device = torch.device('cpu') if rollout_cpu else train_device

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
        './lunar-lander-memories/action-chunking',
        max_episodes = buffer_episodes,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            chunk_start = 'bool',
            state = ('float', state_dim),
            action = 'int',
            action_log_prob = 'float',
            reward = 'float',
            is_boundary = 'bool',
            value = ('float', critic_pred_num_bins),
            returns = 'float',
            past_action = ('float', num_actions),
            sampled_chunk_len = 'int'
        ),
        circular = True,
        overwrite = True
    )

    agent = PPO(
        state_dim = state_dim,
        num_actions = num_actions,
        actor_hidden_dim = actor_hidden_dim,
        critic_hidden_dim = critic_hidden_dim,
        critic_pred_num_bins = critic_pred_num_bins,
        reward_range = reward_range,
        epochs = epochs,
        minibatch_size = minibatch_size,
        lr = lr,
        betas = betas,
        lam = lam,
        gamma = gamma,
        beta_s = beta_s,
        regen_reg_rate = regen_reg_rate,
        cautious_factor = cautious_factor,
        eps_clip = eps_clip,
        ema_decay = ema_decay,
        action_chunk_size = action_chunk_size,
        critic_chunk_size = critic_chunk_size,
        actor_depth = actor_depth,
        critic_depth = critic_depth
    ).to(train_device)

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0

    reward_window = deque(maxlen = log_window_size)
    steps_window = deque(maxlen = log_window_size)

    pbar = tqdm(range(num_episodes), desc = 'episodes')

    max_hist_len = agent.critic.max_seq_len - 1
    dummy_value = torch.zeros(critic_pred_num_bins)

    for eps in pbar:
        agent.eval()
        episode_reward = 0.

        state, _ = env.reset(seed = seed)
        state = torch.from_numpy(state).to(rollout_device)
        past_action = torch.zeros(num_actions).to(rollout_device)

        state_history = deque(maxlen = max_hist_len)
        past_action_history = deque(maxlen = max_hist_len)
        action_queue = deque()
        ep_sampled_chunk_lengths = []
        chunk_len = 0

        with memories.one_episode():
            for timestep in range(max_timesteps):
                time += 1
                state_history.append(state)
                past_action_history.append(past_action)

                is_chunk_start = len(action_queue) == 0

                if is_chunk_start:
                    # propose a chunk of actions from current state

                    with torch.no_grad():
                        chunk_logits = agent.actor(
                            rearrange(state, '... -> 1 ...').to(train_device),
                            rearrange(past_action, '... -> 1 ...').to(train_device),
                            chunk_size = action_chunk_size
                        )

                    chunk_logits = chunk_logits[0]  # (chunk_size, num_actions)

                    if dynamic_chunk:
                        with torch.no_grad():
                            sampled_actions = Categorical(logits = chunk_logits).sample()
                            action_seq = F.one_hot(sampled_actions, num_classes = num_actions).float()
                            action_seq = rearrange(action_seq, 'c d -> 1 c d').to(train_device)
                            state_seq = repeat(state, 'd -> 1 c d', c = action_chunk_size).to(train_device)

                            q_values = agent.critic_hl_gauss_loss(agent.critic(state_seq, action_seq))[0]

                        if bernoulli(dynamic_chunk_eps):
                            chunk_len = random.randint(dynamic_chunk_min_len, action_chunk_size)
                        else:
                            chunk_len = sample_categorical(q_values, low = dynamic_chunk_min_len)
                            ep_sampled_chunk_lengths.append(chunk_len)
                    else:
                        chunk_len = action_chunk_size

                    for logits in chunk_logits[:chunk_len]:
                        action_queue.append(logits.to(rollout_device))

                # dequeue action

                action_logits = action_queue.popleft()
                dist = Categorical(logits = action_logits)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                next_state = torch.from_numpy(next_state).to(rollout_device)
                next_past_action = F.one_hot(action, num_classes = num_actions).float().to(rollout_device)

                if timestep == max_timesteps - 1:
                    truncated = True

                reward = float(reward)

                updating_agent = divisible_by(time, update_timesteps)
                done = terminated or truncated or updating_agent

                if done and not terminated:
                    hist_states = rearrange(torch.stack([*state_history, next_state]), '... -> 1 ...').to(train_device)
                    hist_actions = rearrange(torch.stack([*past_action_history, next_past_action]), '... -> 1 ...').to(train_device)

                    with torch.no_grad():
                        bootstrap_value = agent.critic(hist_states, hist_actions)[0, -1]

                    reward += agent.gamma * agent.critic_hl_gauss_loss(rearrange(bootstrap_value, '... -> 1 ...')).item()

                episode_reward += reward

                memories.store(
                    chunk_start = is_chunk_start,
                    state = state,
                    action = action,
                    action_log_prob = action_log_prob,
                    reward = reward,
                    is_boundary = done,
                    value = dummy_value,
                    past_action = past_action,
                    sampled_chunk_len = chunk_len
                )

                state = next_state
                past_action = next_past_action

                if updating_agent:
                    agent.train()
                    action_queue.clear()
                    agent.learn(memories, train_device)
                    memories.clear()
                    agent.eval()

                if done:
                    break

        reward_window.append(episode_reward)
        steps_window.append(timestep + 1)

        avg_reward = sum(reward_window) / len(reward_window)
        avg_steps = sum(steps_window) / len(steps_window)

        postfix_kwargs = dict(reward = f"{avg_reward:.2f}", steps = f"{avg_steps:.1f}")
        log_dict = dict(
            reward_avg = avg_reward,
            steps_avg = avg_steps,
            reward_raw = episode_reward,
            steps_raw = timestep + 1,
        )

        if dynamic_chunk and len(ep_sampled_chunk_lengths) > 0:
            avg_chunk_len = sum(ep_sampled_chunk_lengths) / len(ep_sampled_chunk_lengths)
            postfix_kwargs.update(chunk_len = f"{avg_chunk_len:.1f}")
            log_dict.update(avg_chunk_len = avg_chunk_len)

        pbar.set_postfix(**postfix_kwargs)

        if use_wandb:
            wandb.log(log_dict, step = eps)

        if divisible_by(eps, save_every):
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
