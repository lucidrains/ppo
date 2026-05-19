# /// script
# dependencies = [
#   "adam-atan2-pytorch",
#   "assoc-scan",
#   "discrete-continuous-embed-readout",
#   "ema-pytorch",
#   "fire",
#   "gymnasium[mujoco,moviepy]>=1.0.0",
#   "gymnasium-robotics",
#   "hl-gauss-pytorch>=0.1.7",
#   "memmap-replay-buffer>=0.1.1",
#   "numpy",
#   "torch",
#   "tqdm",
#   "x-transformers",
#   "einops",
#   "wandb"
# ]
# ///

# transformer critic as proposed in T-SAC - Dong Tian et al. https://arxiv.org/abs/2503.03660
# action chunking PPO - Sanghyun Hahn et al. https://openreview.net/forum?id=WFQnqY1c39
# adaptive action chunking - Yongjae Shin et al. https://arxiv.org/abs/2605.10044
# adapted for AntMaze continuous control with Beta distribution readout

from __future__ import annotations

import os
import shutil
import random
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import fire
from pathlib import Path
from collections import deque

import wandb

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import Categorical

from ema_pytorch import EMA
from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2
from hl_gauss_pytorch import HLGaussLoss

from assoc_scan import AssocScan
from x_transformers import Decoder
from einops import rearrange, repeat
from torch_einops_utils import lens_to_mask

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import gymnasium_robotics
from memmap_replay_buffer import ReplayBuffer

from discrete_continuous_embed_readout import Readout

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

def concat_goal_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([obs['observation'], obs['desired_goal']], axis = -1)
    return obs

def extract_final_obs(infos, idx):
    if 'final_observation' not in infos:
        return None

    final = infos['final_observation']

    if not isinstance(final, (list, tuple, np.ndarray)):
        return None

    entry = final[idx]

    if not isinstance(entry, dict) or 'observation' not in entry:
        return None

    return entry

# networks

class TransformerPolicy(nn.Module):
    def __init__(
        self,
        *,
        dim_state,
        action_dim,
        dim_hidden = 256,
        depth = 3,
        heads = 8,
        max_chunk_size = 64
    ):
        super().__init__()
        self.state_proj = nn.Linear(dim_state, dim_hidden)
        self.action_proj = nn.Linear(action_dim, dim_hidden)
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

        self.readout = Readout(
            dim = dim_hidden,
            num_continuous = action_dim,
            continuous_dist_type = 'beta'
        )

    def forward(
        self,
        state: Tensor,
        past_action: Tensor,
        chunk_size = 1
    ):
        state_tokens = self.state_proj(state)
        action_tokens = self.action_proj(past_action.float())

        context = torch.stack((state_tokens, action_tokens), dim = 1)
        queries = repeat(self.chunk_queries[:chunk_size], 'n d -> b n d', b = state.shape[0])

        tokens = torch.cat((context, queries), dim = 1)
        out = self.decoder(tokens)

        params = self.readout(out[:, 2:])
        return self.readout.continuous_dist.dist(params)

class TransformerCritic(nn.Module):
    def __init__(
        self,
        *,
        dim_state,
        action_dim,
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
        self.action_proj = nn.Linear(action_dim, dim_hidden)

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
        has_time = state.ndim == 3

        if not has_time:
            state = rearrange(state, 'b d -> b 1 d')
            past_action = rearrange(past_action, 'b d -> b 1 d')

        b, seq, _ = state.shape
        assert seq <= self.max_seq_len, f'{seq} needs to be less than max seq len {self.max_seq_len}'

        tokens = self.state_proj(state) + self.action_proj(past_action.float())

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
        action_dim,
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
        critic_value_chunk_seq_len = 8,
        actor_depth = 3,
        critic_depth = 4,
        ema_kwargs: dict = dict(update_every = 10),
        save_path = './ppo.pt'
    ):
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.critic_value_chunk_seq_len = critic_value_chunk_seq_len

        self.actor = TransformerPolicy(
            dim_state = state_dim,
            action_dim = action_dim,
            dim_hidden = actor_hidden_dim,
            depth = actor_depth
        )

        self.critic = TransformerCritic(
            dim_state = state_dim,
            action_dim = action_dim,
            dim_hidden = critic_hidden_dim,
            dim_out = critic_pred_num_bins,
            depth = critic_depth,
            max_seq_len = critic_value_chunk_seq_len + 1
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
                    self.critic(states[:, i:i + self.critic_value_chunk_seq_len], past_actions[:, i:i + self.critic_value_chunk_seq_len])
                    for i in range(0, seq, self.critic_value_chunk_seq_len)
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

                memories.data['returns'][indices, :returns.shape[-1]] = returns.cpu().numpy()
                memories.data['value'][indices, :values.shape[-2]] = values.cpu().numpy()
                memories.flush()

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
            n_steps         = self.critic_value_chunk_seq_len,
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

                dist = self.actor(batch.seq_state[:, 0], batch.seq_past_action[:, 0], chunk_size = seq)

                safe_action = batch.seq_action.clamp(1e-5, 1 - 1e-5)
                action_log_probs = dist.log_prob(safe_action).sum(dim = -1)
                entropy = dist.entropy().sum(dim = -1)

                advantages = normalize(batch.seq_returns[valid] - hl_gauss(batch.seq_value[valid]).detach())

                ratios = (action_log_probs[valid] - batch.seq_action_log_prob[valid]).exp()
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages

                policy_loss = (-torch.min(surr1, surr2) - self.beta_s * entropy[valid]).mean()

                update_network_(policy_loss, self.opt_actor)

            for batch in critic_dl:
                valid = lens_to_mask(batch.n_step_lens, batch.seq_returns.shape[-1])

                values = self.critic(batch.seq_state, batch.seq_past_action)
                value_loss = hl_gauss(values[valid], batch.seq_returns[valid]).mean()

                update_network_(value_loss, self.opt_critic)

# vectorized rollout

def collect_vectorized_rollouts(
    env,
    agent,
    num_envs,
    action_dim,
    max_timesteps,
    action_chunk_size,
    memories,
    device,
    dynamic_chunk = False,
    dynamic_chunk_min_len = 1,
    dynamic_chunk_eps = 0.05
):
    state_dict, _ = env.reset()
    state = torch.from_numpy(concat_goal_obs(state_dict)).float().to(device)

    env_active = np.ones(num_envs, dtype = bool)
    cum_rewards = np.zeros(num_envs)
    env_steps = np.zeros(num_envs, dtype = int)

    fields = (
        'chunk_start', 'state', 'action', 'action_log_prob', 'reward',
        'is_boundary', 'value', 'returns', 'past_action', 'sampled_chunk_len'
    )

    collected = [{f: [] for f in fields} for _ in range(num_envs)]

    action_queues = [deque() for _ in range(num_envs)]
    chunk_lengths_tracker = [[] for _ in range(num_envs)]
    past_action = torch.zeros(num_envs, action_dim, device = device)
    dummy_value = torch.zeros(agent.critic.dim_out)

    for timestep in range(max_timesteps):
        if not env_active.any():
            break

        # propose chunks for envs that need them

        needs_chunk = [i for i in range(num_envs) if env_active[i] and len(action_queues[i]) == 0]

        if len(needs_chunk) > 0:
            with torch.no_grad():
                chunk_dist = agent.actor(state[needs_chunk], past_action[needs_chunk], chunk_size = action_chunk_size)
                chunk_actions = chunk_dist.sample()
                chunk_log_probs = chunk_dist.log_prob(chunk_actions).sum(dim = -1)
                
                if dynamic_chunk:
                    state_seq = repeat(state[needs_chunk], 'b d -> b c d', c = action_chunk_size)
                    q_values = agent.critic_hl_gauss_loss(agent.critic(state_seq, chunk_actions))

            for idx, env_idx in enumerate(needs_chunk):
                chunk_len = action_chunk_size
                if dynamic_chunk:
                    if bernoulli(dynamic_chunk_eps):
                        chunk_len = random.randint(dynamic_chunk_min_len, action_chunk_size)
                    else:
                        chunk_len = sample_categorical(q_values[idx], low = dynamic_chunk_min_len)

                chunk_lengths_tracker[env_idx].append(chunk_len)

                for step in range(chunk_len):
                    action_queues[env_idx].append((chunk_actions[idx, step], chunk_log_probs[idx, step], chunk_len))

        # dequeue one action per env

        step_actions = torch.zeros(num_envs, action_dim, device = device)
        step_log_probs = torch.zeros(num_envs, device = device)
        is_chunk_start = np.zeros(num_envs, dtype = bool)

        step_chunk_lens = np.zeros(num_envs, dtype = int)
        for i in range(num_envs):
            if not env_active[i]:
                continue

            action, log_prob, chunk_len = action_queues[i].popleft()
            is_chunk_start[i] = len(action_queues[i]) == chunk_len - 1
            step_actions[i] = action
            step_log_probs[i] = log_prob
            step_chunk_lens[i] = chunk_len

        # step environment - rescale from Beta [0, 1] to action space [-1, 1]

        env_action = (step_actions * 2 - 1).clamp(-1., 1.).cpu().numpy()

        next_state_dict, reward_np, terminated, truncated, infos = env.step(env_action)
        next_state = torch.from_numpy(concat_goal_obs(next_state_dict)).float().to(device)
        done = terminated | truncated

        # store transitions

        for i in range(num_envs):
            if not env_active[i]:
                continue

            reward = float(reward_np[i])
            cum_rewards[i] += reward
            env_steps[i] += 1

            env_data = collected[i]
            env_data['chunk_start'].append(is_chunk_start[i])
            env_data['state'].append(state[i].cpu())
            env_data['action'].append(step_actions[i].cpu())
            env_data['action_log_prob'].append(step_log_probs[i].item())
            env_data['reward'].append(reward)
            env_data['is_boundary'].append(terminated[i])
            env_data['value'].append(dummy_value)
            env_data['returns'].append(0.)
            env_data['past_action'].append(past_action[i].cpu())
            env_data['sampled_chunk_len'].append(step_chunk_lens[i])

            if not done[i]:
                continue

            env_active[i] = False

            # bootstrap truncated episodes

            if terminated[i] or len(env_data['state']) > max_timesteps:
                continue

            final_obs = extract_final_obs(infos, i)

            if final_obs is None:
                continue

            final_state_np = concat_goal_obs({
                'observation': final_obs['observation'],
                'desired_goal': final_obs['desired_goal']
            })

            final_state = torch.from_numpy(final_state_np).float().to(device)
            final_action = step_actions[i]

            with torch.no_grad():
                bootstrap_value = agent.critic(
                    rearrange(final_state, 'd -> 1 d'),
                    rearrange(final_action, 'd -> 1 d')
                )

            bootstrap_scalar = agent.critic_hl_gauss_loss(rearrange(bootstrap_value, '... -> 1 ...')).item()
            env_data['reward'][-1] += agent.gamma * bootstrap_scalar

        state = next_state
        past_action = step_actions

    # flush collected episodes into replay buffer

    for i in range(num_envs):
        env_data = collected[i]
        with memories.one_episode():
            for t in range(len(env_data['state'])):
                memories.store(**{field: env_data[field][t] for field in fields})

    return cum_rewards, env_steps, chunk_lengths_tracker

# evaluation

def evaluate_and_log_video(env_name, agent, device, step, max_timesteps, action_chunk_size):
    eval_env = gym.make(env_name, render_mode = 'rgb_array')

    video_folder = f'./antmaze-recordings/eval_{step}'
    eval_env = RecordVideo(eval_env, video_folder = video_folder, episode_trigger = lambda x: True, disable_logger = True)

    obs, _ = eval_env.reset()
    state = torch.from_numpy(concat_goal_obs(obs)).float().to(device)

    action_dim = eval_env.action_space.shape[0]
    past_action = torch.zeros(action_dim, device = device)

    action_queue = deque()
    total_reward = 0.

    for _ in range(max_timesteps):
        if len(action_queue) == 0:
            with torch.no_grad():
                dist = agent.actor(rearrange(state, 'd -> 1 d'), rearrange(past_action, 'd -> 1 d'), chunk_size = action_chunk_size)
                chunk_actions = dist.mean[0] if hasattr(dist, 'mean') else dist.sample()[0]

                for i in range(action_chunk_size):
                    action_queue.append(chunk_actions[i])

        action = action_queue.popleft()
        env_action = (action * 2 - 1).clamp(-1., 1.).cpu().numpy()

        obs, reward, terminated, truncated, _ = eval_env.step(env_action)
        state = torch.from_numpy(concat_goal_obs(obs)).float().to(device)
        past_action = action
        total_reward += reward

        if terminated or truncated:
            break

    eval_env.close()

    video_files = list(Path(video_folder).glob('*.mp4'))

    if len(video_files) > 0 and wandb.run is not None:
        wandb.log({'eval/video': wandb.Video(str(video_files[0]), fps = 4, format = 'gif')}, step = step)

    return total_reward

# main

def main(
    num_envs = 8,
    env_name = 'AntMaze_UMaze-v5',
    num_episodes = 6000,
    max_timesteps = 700,
    actor_hidden_dim = 256,
    critic_hidden_dim = 256,
    actor_depth = 3,
    critic_depth = 4,
    action_chunk_size = 4,
    critic_value_chunk_seq_len = 8,
    update_episodes = 8,
    buffer_episodes = 40,
    critic_pred_num_bins = 100,
    reward_range = (-50., 50.),
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
    dynamic_chunk = False,
    dynamic_chunk_min_len = 1,
    dynamic_chunk_eps = 0.05,
    seed = None,
    save_every = 1000,
    eval_every = 500,
    load = False,
    use_wandb = True,
    wandb_project = 'ppo-action-chunking-antmaze',
    log_window_size = 20
):
    if use_wandb:
        wandb.init(project = wandb_project, config = locals())

    video_dir = Path('./antmaze-recordings')

    if video_dir.exists():
        shutil.rmtree(video_dir)

    video_dir.mkdir(parents = True, exist_ok = True)
    print(f'evaluation videos will be logged to {video_dir.resolve()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # environment

    env = gym.make_vec(env_name, num_envs = num_envs, vectorization_mode = 'async')

    temp_env = gym.make(env_name)
    state_dim = temp_env.observation_space['observation'].shape[0] + temp_env.observation_space['desired_goal'].shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()

    # replay buffer

    memories = ReplayBuffer(
        './antmaze-memories/action-chunking',
        max_episodes = buffer_episodes,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            chunk_start = 'bool',
            state = ('float', state_dim),
            action = ('float', action_dim),
            action_log_prob = 'float',
            reward = 'float',
            is_boundary = 'bool',
            value = ('float', critic_pred_num_bins),
            returns = 'float',
            past_action = ('float', action_dim),
            sampled_chunk_len = 'int'
        ),
        circular = True,
        overwrite = True
    )

    # agent

    agent = PPO(
        state_dim = state_dim,
        action_dim = action_dim,
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
        critic_value_chunk_seq_len = critic_value_chunk_seq_len,
        actor_depth = actor_depth,
        critic_depth = critic_depth
    ).to(device)

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    # training loop

    reward_window = deque(maxlen = log_window_size)
    steps_window = deque(maxlen = log_window_size)

    pbar = tqdm(total = num_episodes, desc = 'episodes')

    total_eps = 0
    prev_eps = 0

    while total_eps < num_episodes:
        agent.eval()

        cum_rewards, steps, chunk_lengths = collect_vectorized_rollouts(
            env, agent, num_envs, action_dim, max_timesteps,
            action_chunk_size, memories, device,
            dynamic_chunk = dynamic_chunk,
            dynamic_chunk_min_len = dynamic_chunk_min_len,
            dynamic_chunk_eps = dynamic_chunk_eps
        )

        total_eps += num_envs
        pbar.update(num_envs)

        reward_window.extend(cum_rewards)
        steps_window.extend(steps)

        avg_reward = sum(reward_window) / len(reward_window)
        avg_steps = sum(steps_window) / len(steps_window)

        all_chunk_lengths = [length for env_lengths in chunk_lengths for length in env_lengths]
        avg_chunk_len = sum(all_chunk_lengths) / max(len(all_chunk_lengths), 1)

        pbar.set_postfix(reward = f'{avg_reward:.2f}', steps = f'{avg_steps:.1f}', chunk_len = f'{avg_chunk_len:.1f}')

        if use_wandb:
            for i, (cum_reward, step_count) in enumerate(zip(cum_rewards, steps)):
                log_dict = dict(
                    episode_reward = cum_reward,
                    episode_steps = step_count,
                    reward_avg = avg_reward,
                    steps_avg = avg_steps,
                    avg_chunk_len = avg_chunk_len,
                    total_eps = total_eps
                )

                wandb.log(log_dict)

        # periodic learning

        if (total_eps // update_episodes) > (prev_eps // update_episodes):
            agent.train()
            agent.learn(memories, device)
            memories.clear()

        if divisible_by(total_eps, save_every):
            agent.save()

        # periodic evaluation

        if (total_eps // eval_every) > (prev_eps // eval_every) and use_wandb:
            eval_reward = evaluate_and_log_video(env_name, agent, device, total_eps, max_timesteps, action_chunk_size)
            wandb.log({'eval/reward': eval_reward}, step = total_eps)

        prev_eps = total_eps

    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    fire.Fire(main)
