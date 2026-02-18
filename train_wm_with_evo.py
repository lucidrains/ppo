# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "tqdm",
#   "einops",
#   "einx",
#   "ema-pytorch",
#   "adam-atan2-pytorch",
#   "hl-gauss-pytorch",
#   "assoc-scan",
#   "gymnasium[box2d]",
#   "pygame",
#   "moviepy",
#   "x-transformers>=2.16.1",
#   "x-evolution>=0.1.30",
#   "accelerate",
#   "wandb",
#   "fire",
#   "memmap-replay-buffer",
# ]
# ///

from __future__ import annotations

import wandb
import fire
from accelerate import Accelerator
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from functools import partial, wraps
from collections import deque, namedtuple
from random import randrange

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, is_tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical
from torch.utils._pytree import tree_map

from torch.nn.utils.rnn import pad_sequence
pad_sequence = partial(pad_sequence, batch_first = True)

import einx
from einops import reduce, repeat, einsum, rearrange, pack
from einops.layers.torch import Rearrange

from ema_pytorch import EMA

from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from hl_gauss_pytorch import HLGaussLoss

from x_transformers import (
    Decoder,
    ContinuousTransformerWrapper
)

from x_evolution import EvoStrategy
from memmap_replay_buffer import ReplayBuffer

from assoc_scan import AssocScan

import gymnasium as gym

# memory tuple

Memory = namedtuple('Memory', [
    'eps',
    'state',
    'action',
    'action_log_prob',
    'reward',
    'is_boundary',
    'value',
    'dones'
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

def frac_gradient(t, frac = 1.):
    assert 0 <= frac <= 1.
    return t.detach() * (1. - frac) + t * frac

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

# world model + actor / critic in one

class WorldModelActorCritic(Module):
    def __init__(
        self,
        backbone_transformer: Module,
        actor_transformer: Module,
        critic_transformer: Module,
        num_actions,
        critic_dim_pred,
        critic_min_max_value: tuple[float, float],
        dim_pred_state,
        frac_actor_critic_head_gradient = 0.5,
        entropy_weight = 0.02,
        eps_clip = 0.2,
        value_clip = 0.4
    ):
        super().__init__()
        self.backbone_transformer = backbone_transformer
        self.actor_transformer = actor_transformer
        self.critic_transformer = critic_transformer

        dim = backbone_transformer.attn_layers.dim

        self.action_embeds = nn.Embedding(num_actions, dim)

        self.to_dones = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Sigmoid()            
        )

        self.to_pred = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim_pred_state * 2),
            Rearrange('... (mean_var d) -> mean_var ... d', mean_var = 2)
        )

        self.critic_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, critic_dim_pred)
        )

         # https://arxiv.org/abs/2403.03950

        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = critic_min_max_value[0],
            max_value = critic_min_max_value[1],
            num_bins = critic_dim_pred,
            clamp_to_range = True
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, num_actions),
            nn.Softmax(dim = -1)
        )

        self.frac_actor_critic_head_gradient = frac_actor_critic_head_gradient

        # ppo loss related

        self.eps_clip = eps_clip
        self.entropy_weight = entropy_weight

        # clipped value loss related

        self.value_clip = value_clip

    def compute_autoregressive_loss(
        self,
        pred,
        real
    ):
        pred_mean, pred_var = pred[..., :-1, :] # todo: fix truncation scenario
        return F.gaussian_nll_loss(pred_mean, real[:, 1:], pred_var, reduction = 'none')

    def compute_done_loss(
        self,
        done_pred,
        dones
    ):
        return F.binary_cross_entropy(done_pred, dones.float(), reduction = 'none')

    def compute_actor_loss(
        self,
        action_probs,
        actions,
        old_log_probs,
        returns,
        old_values
    ):
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        scalar_old_values = self.critic_hl_gauss_loss(old_values)

        # calculate clipped surrogate objective, classic PPO loss

        ratios = (action_log_probs - old_log_probs).exp()

        advantages = normalize(returns - scalar_old_values.detach())

        surr1 = ratios * advantages
        surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = - torch.min(surr1, surr2) - self.entropy_weight * entropy
        return actor_loss, entropy

    def compute_critic_loss(
        self,
        values,
        returns,
        old_values
    ):
        clip, hl_gauss = self.value_clip, self.critic_hl_gauss_loss

        scalar_old_values = hl_gauss(old_values)
        scalar_values = hl_gauss(values)

        # using the proposal from https://www.authorea.com/users/855021/articles/1240083-on-analysis-of-clipped-critic-loss-in-proximal-policy-gradient

        clipped_returns = returns.clamp(scalar_old_values - clip, scalar_old_values + clip)

        clipped_loss = hl_gauss(values, clipped_returns, reduction = 'none')
        loss = hl_gauss(values, returns, reduction = 'none')

        old_values_lo = scalar_old_values - clip
        old_values_hi = scalar_old_values + clip

        def is_between(mid, lo, hi):
            return (lo < mid) & (mid < hi)

        critic_loss = torch.where(
            is_between(scalar_values, returns, old_values_lo) |
            is_between(scalar_values, old_values_hi, returns),
            0.,
            torch.min(loss, clipped_loss)
        )

        return critic_loss

    def forward(
        self,
        *args,
        actions = None,
        next_actions = None,
        return_pred_dones = False,
        cache = None,
        mask = None,
        **kwargs
    ):
        sum_embeds = 0.

        if exists(actions):
            has_actions = actions >= 0.
            actions = torch.where(has_actions, actions, 0)
            action_embeds = self.action_embeds(actions)
            action_embeds = einx.where('b n, b n d, ', has_actions, action_embeds, 0.)
            sum_embeds = sum_embeds + action_embeds

        # handle multi-cache

        backbone_cache = actor_cache = critic_cache = None
        if exists(cache):
            backbone_cache, actor_cache, critic_cache = cache

        embed, backbone_cache = self.backbone_transformer(*args, **kwargs, sum_embeds = sum_embeds, return_embeddings = True, return_intermediates = True, cache = backbone_cache)

        # if `next_actions` from agent passed in, use it to predict the next state + truncated / terminated signal

        embed_with_actions = None
        if exists(next_actions):
            next_action_embeds = self.action_embeds(next_actions)
            embed_with_actions = cat((embed, next_action_embeds), dim = -1)

        # predicting state and dones, based on agent's action

        state_pred = None
        dones = None

        if exists(embed_with_actions):
            state_mean, state_log_var = self.to_pred(embed_with_actions)

            state_pred = stack((state_mean, state_log_var.exp()))
            dones = self.to_dones(embed_with_actions)

        # branches

        embed = frac_gradient(embed, self.frac_actor_critic_head_gradient) # what fraction of the gradient to pass back to the world model from the actor / critic head

        actor_embed, actor_cache = self.actor_transformer(embed, mask = mask, cache = actor_cache, return_hiddens = True)
        critic_embed, critic_cache = self.critic_transformer(embed, mask = mask, cache = critic_cache, return_hiddens = True)

        # actions

        action_probs = self.action_head(actor_embed)

        # values

        values = self.critic_head(critic_embed)

        new_cache = (backbone_cache, actor_cache, critic_cache)

        return action_probs, values, state_pred, dones, new_cache

# RSM Norm

class RSMNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.dim = dim
        self.eps = 1e-5

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

        with torch.no_grad():
            new_obs_mean = reduce(x, '... d -> d', 'mean')
            delta = new_obs_mean - mean

            new_mean = mean + delta / time
            new_variance = (time - 1) / time * (variance + (delta ** 2) / time)

            self.step.add_(1)
            self.running_mean.copy_(new_mean)
            self.running_variance.copy_(new_variance)

        return normed

# GAE

@torch.no_grad()
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
        critic_pred_num_bins,
        reward_range: tuple[float, float],
        epochs,
        max_timesteps,
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
        hidden_dim = 32,
        backbone_depth = 1,
        actor_depth = 1,
        critic_depth = 1,
        world_model: dict = dict(
            attn_dim_head = 16,
            heads = 4,
            attn_gate_values = True,
            add_value_residual = True,
            learned_value_residual_mix = True
        ),
        dropout = 0.25,
        max_grad_norm = 0.5,
        frac_actor_critic_head_gradient = 0.5,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1250
        ),
        save_path = './ppo.pt',
        evo_layer_index = None
    ):
        super().__init__()

        self.model_dim = hidden_dim

        state_and_reward_dim = state_dim + 1

        branch_kwargs = {k: v for k, v in world_model.items() if k != 'depth'}

        self.model = WorldModelActorCritic(
            backbone_transformer = ContinuousTransformerWrapper(
                dim_in = state_and_reward_dim,
                dim_out = None,
                max_seq_len = max_timesteps,
                probabilistic = True,
                attn_layers = Decoder(
                    dim = hidden_dim,
                    depth = backbone_depth,
                    polar_pos_emb = True,
                    attn_dropout = dropout,
                    ff_dropout = dropout,
                    **branch_kwargs
                )
            ),
            actor_transformer = Decoder(
                dim = hidden_dim,
                depth = actor_depth,
                polar_pos_emb = True,
                attn_dropout = dropout,
                ff_dropout = dropout,
                **branch_kwargs
            ),
            critic_transformer = Decoder(
                dim = hidden_dim,
                depth = critic_depth,
                polar_pos_emb = True,
                attn_dropout = dropout,
                ff_dropout = dropout,
                **branch_kwargs
            ),
            num_actions = num_actions,
            critic_dim_pred = critic_pred_num_bins,
            critic_min_max_value = reward_range,
            dim_pred_state = state_and_reward_dim,
            entropy_weight = beta_s,
            eps_clip = eps_clip,
            value_clip = value_clip
        )

        self.frac_actor_critic_head_gradient = frac_actor_critic_head_gradient

        # state + reward normalization

        self.rsmnorm = RSMNorm(state_dim + 1)

        self.ema_model = EMA(self.model, beta = ema_decay, include_online_model = False, **ema_kwargs)

        # evolution optimization

        self.evo_layer_index = default(evo_layer_index, 0)
        
        if exists(self.evo_layer_index):
            # evo layer now acts on the actor transformer layers
            
            num_actor_layers = len(self.model.actor_transformer.layers)
            self.evo_layer_index = min(self.evo_layer_index, num_actor_layers - 1)

            evo_layer = self.model.actor_transformer.layers[self.evo_layer_index]
            evo_layer_params = set(evo_layer.parameters())
            ppo_params = [p for p in self.model.parameters() if p not in evo_layer_params]
        else:
            ppo_params = self.model.parameters()

        self.optimizer = AdoptAtan2(ppo_params, lr = lr, betas = betas, regen_reg_rate = regen_reg_rate, cautious_factor = cautious_factor)

        self.max_grad_norm = max_grad_norm

        self.ema_model.add_to_optimizer_post_step_hook(self.optimizer)

        # learning hparams

        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.save_path = Path(save_path)

    def save(self):
        torch.save({
            'model': self.model.state_dict(),
            'rsmnorm': self.rsmnorm.state_dict(),
        }, str(self.save_path))

    def load(self, device = None):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path), weights_only = True, map_location = device)

        self.model.load_state_dict(data['model'])
        if 'rsmnorm' in data:
            self.rsmnorm.load_state_dict(data['rsmnorm'])

    def learn(self, replay_buffer, device):

        model = self.model
        hl_gauss = self.model.critic_hl_gauss_loss

        # retrieve and prepare data from buffer for training
        
        data = replay_buffer.get_all_data()
        num_episodes = replay_buffer.num_episodes

        def to_device(t):
            return t.to(device)

        states = to_device(data['state'][:num_episodes])
        actions = to_device(data['action'][:num_episodes])
        old_log_probs = to_device(data['action_log_prob'][:num_episodes])
        rewards = to_device(data['reward'][:num_episodes])
        is_boundaries = to_device(data['is_boundary'][:num_episodes])
        values = to_device(data['value'][:num_episodes])
        dones = to_device(data['dones'][:num_episodes])
        
        episode_lens = torch.from_numpy(replay_buffer.meta_data['episode_lens'][:num_episodes]).to(device)

        masks = ~is_boundaries

        # calculate generalized advantage estimate

        scalar_values = hl_gauss(values)

        returns = calc_gae(
            rewards = rewards,            
            masks = masks,
            lam = self.lam,
            gamma = self.gamma,
            values = scalar_values,
            use_accelerated = False
        )

        dataset = TensorDataset(
            states,
            actions,
            rewards,
            old_log_probs,
            returns,
            values,
            dones,
            episode_lens
        )

        dl = DataLoader(dataset, batch_size = self.minibatch_size, shuffle = True)

        model.train()

        rsmnorm_copy = deepcopy(self.rsmnorm)
        rsmnorm_copy.train()

        all_metrics = []

        for _ in range(self.epochs):
            for (
                states,
                actions,
                rewards,
                old_log_probs,
                returns,
                old_values,
                dones,
                episode_lens
             ) in dl:

                seq = torch.arange(states.shape[1], device = device)
                mask = einx.less('n, b -> b n', seq, episode_lens)

                prev_actions = F.pad(actions, (1, -1), value = -1)
                rewards_pad = F.pad(rewards, (1, -1), value = 0.)

                raw_states_with_rewards, _ = pack((states, rewards_pad), 'b n *')

                with torch.no_grad():
                    self.rsmnorm.eval()
                    states_with_rewards = self.rsmnorm(raw_states_with_rewards)

                action_probs, values, states_with_rewards_pred, done_pred, _ = model(
                    states_with_rewards,
                    actions = prev_actions,
                    next_actions = actions,
                    mask = mask,
                    return_pred_dones = True
                )

                # world model loss

                world_model_loss = model.compute_autoregressive_loss(
                    states_with_rewards_pred,
                    states_with_rewards
                )
                world_model_loss = world_model_loss[mask[:, :-1]].mean()

                # done loss

                pred_done_loss = model.compute_done_loss(done_pred, dones)
                pred_done_loss = pred_done_loss[mask].mean()

                # actor critic loss

                actor_loss, entropy = model.compute_actor_loss(
                    action_probs,
                    actions,
                    old_log_probs,
                    returns,
                    old_values
                )
                actor_loss = actor_loss[mask].mean()
                entropy = entropy[mask].mean()

                critic_loss = model.compute_critic_loss(
                    values,
                    returns,
                    old_values,
                )
                critic_loss = critic_loss[mask].mean()

                loss = world_model_loss + actor_loss + critic_loss + pred_done_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

                rsmnorm_copy.train()
                rsmnorm_copy(raw_states_with_rewards[mask])

                all_metrics.append(dict(
                    world_model_loss = world_model_loss.item(),
                    actor_loss = actor_loss.item(),
                    critic_loss = critic_loss.item(),
                    pred_done_loss = pred_done_loss.item(),
                    entropy = entropy.item(),
                    loss = loss.item()
                ))

        self.rsmnorm.load_state_dict(rsmnorm_copy.state_dict())

        # return averaged metrics

        return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 400,
    critic_pred_num_bins = 600,
    reward_range = (-300, 300),
    minibatch_size = 8,
    update_episodes = 64,
    lr = 0.0008,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    value_clip = 0.4,
    beta_s = .01,
    regen_reg_rate = 1e-4,
    cautious_factor = 0.1,
    render = True,
    clear_videos = True,
    epochs = 4,
    ema_decay = 0.9,
    seed = None,
    render_every_eps = 100,
    save_every = 1000,
    video_folder = './lunar-recording',
    load = False,
    use_wandb = False,
    wandb_project = 'ppo-wm-evo',
    wandb_run_name = None,
    cpu = False,
    hidden_dim = 32,
    backbone_depth = 1,
    actor_depth = 1,
    critic_depth = 1,
    dropout = 0.1,
    evo_every = 0,
    evo_generations = 2,
    evo_pop_size = 32,
    evo_noise_scale = 1e-2,
    evo_layer_index = None
):
    assert divisible_by(update_episodes, minibatch_size)

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

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    if not exists(evo_layer_index):
        evo_layer_index = 2 # middle of 4 layers

    replay_buffer = ReplayBuffer(
        './replay-buffer',
        max_episodes = update_episodes * 2,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', (state_dim,)),
            action = 'int',
            action_log_prob = 'float',
            reward = 'float',
            is_boundary = 'bool',
            value = ('float', (critic_pred_num_bins,)),
            dones = ('float', (2,))
        ),
        circular = True,
        overwrite = True
    )

    agent = PPO(
        state_dim,
        num_actions,
        critic_pred_num_bins,
        reward_range,
        epochs,
        max_timesteps,
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
        hidden_dim = hidden_dim,
        backbone_depth = backbone_depth,
        actor_depth = actor_depth,
        critic_depth = critic_depth,
        dropout = dropout,
        evo_layer_index = evo_layer_index
    ).to(device)

    if load:
        agent.load(device = device)

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0

    # evolution strategy

    evo_strategy = None

    if evo_every > 0:
        def evo_environment(model):
            state, _ = env.reset()
            state = torch.from_numpy(state).to(device).float()
            cumulative_reward = 0
            prev_action = torch.tensor(-1, device = device)
            prev_reward = torch.tensor(0., dtype = torch.float32, device = device)
            world_model_cache = None

            model.eval()

            for _ in range(max_timesteps):
                state_with_reward = cat((state, rearrange(prev_reward, '-> 1')), dim = -1)
                agent.rsmnorm.eval()
                normed_state = agent.rsmnorm(state_with_reward)

                normed_state = rearrange(normed_state, 'd -> 1 1 d')
                input_prev_action = rearrange(prev_action, ' -> 1 1')

                action_probs, values, _, _, world_model_cache = model(
                    normed_state,
                    cache = world_model_cache,
                    input_not_include_cache = True,
                    actions = input_prev_action
                )

                action_probs = rearrange(action_probs, '1 1 d -> d')
                dist = Categorical(action_probs)
                action = dist.sample()
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                cumulative_reward += reward
                if terminated or truncated:
                    break
                state = torch.from_numpy(next_state).to(device).float()
                prev_action = action
                prev_reward = torch.tensor(reward, dtype = torch.float32, device = device)

            return float(cumulative_reward)

        evo_strategy = EvoStrategy(
            agent.model,
            environment = evo_environment,
            params_to_optimize = agent.model.actor_transformer.layers[agent.evo_layer_index],
            num_generations = evo_generations,
            noise_population_size = evo_pop_size,
            noise_scale = evo_noise_scale,
            accelerator = accelerator
        )

    memories = deque([])

    pbar = tqdm(range(num_episodes), desc = 'episodes')

    for eps in pbar:

        one_episode_memories = deque([])

        state, info = env.reset(seed = seed)
        state = torch.from_numpy(state).to(device).float()

        prev_action = torch.tensor(-1, device = device)
        prev_reward = torch.tensor(0., dtype = torch.float32, device = device)

        cumulative_reward = 0
        world_model_cache = None

        @torch.no_grad()
        def state_to_pred_action_and_value(state, prev_action, prev_reward):
            nonlocal world_model_cache

            state_with_reward = cat((state, rearrange(prev_reward, '-> 1')), dim = -1)

            agent.rsmnorm.eval()
            normed_state = agent.rsmnorm(state_with_reward)

            agent.ema_model.eval()

            normed_state = rearrange(normed_state, 'd -> 1 1 d')
            input_prev_action = rearrange(prev_action, ' -> 1 1')

            action_probs, values, _, _, world_model_cache = agent.ema_model.forward_eval(
                normed_state,
                cache = world_model_cache,
                input_not_include_cache = True,
                actions = input_prev_action
            )

            action_probs = rearrange(action_probs, '1 1 d -> d')
            values = rearrange(values, '1 1 d -> d')
            return action_probs, values

        for timestep in range(max_timesteps):
            time += 1
            
            action_probs, value = state_to_pred_action_and_value(state, prev_action, prev_reward)

            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            next_state = torch.from_numpy(next_state).to(device).float()

            reward = float(reward)
            cumulative_reward += reward

            prev_action = action
            prev_reward = torch.tensor(reward, dtype = torch.float32, device = device)

            dones_signal = torch.tensor([terminated, truncated], dtype = torch.float32)
            one_episode_memories.append(dict(
                state = state,
                action = action,
                action_log_prob = action_log_prob,
                reward = torch.tensor(reward, dtype = torch.float32),
                is_boundary = torch.tensor(terminated),
                value = value,
                dones = dones_signal
            ))

            state = next_state

            done = terminated or truncated

            if done and not terminated:
                _, next_value, *_ = state_to_pred_action_and_value(state, prev_action, prev_reward)

                one_episode_memories.append(dict(
                    state = state,
                    action = torch.tensor(-1),
                    action_log_prob = torch.tensor(0.),
                    reward = torch.tensor(0.),
                    is_boundary = torch.tensor(True),
                    value = next_value,
                    dones = torch.zeros(2)
                ))

            if done:
                break

        # store episode to replay buffer

        def list_dict_to_dict_list(ld):
            return {k: stack([d[k] for d in ld]) for k in ld[0]}

        replay_buffer.store_episode(**list_dict_to_dict_list(one_episode_memories))

        pbar.set_postfix(reward = cumulative_reward, steps = timestep + 1)

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

        if divisible_by(eps + 1, update_episodes):

            metrics = agent.learn(replay_buffer, device)
            num_policy_updates += 1

            if use_wandb:
                wandb.log(dict(
                    **metrics,
                    num_policy_updates = num_policy_updates
                ))

            if exists(evo_strategy) and divisible_by(num_policy_updates, evo_every):
                for _ in tqdm(range(evo_generations), desc = 'evolution generations'):
                    rewards = evo_strategy.forward(num_generations = 1)
                    agent.ema_model.update()

                    if use_wandb:
                        wandb.log(dict(
                            evo_reward_mean = rewards.mean().item(),
                            evo_reward_max = rewards.max().item()
                        ))

        if divisible_by(eps, save_every):
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
