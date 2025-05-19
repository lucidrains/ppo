from __future__ import annotations

import fire
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

from assoc_scan import AssocScan

import gymnasium as gym

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

def entropy(prob):
    return (-prob * log(prob)).sum()

def temp_batch_dim(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        args, kwargs = tree_map(lambda t: rearrange(t, '... -> 1 ...') if is_tensor(t) else t, (args, kwargs))

        out = fn(*args, **kwargs)

        out = tree_map(lambda t: rearrange(t, '1 ... -> ...') if is_tensor(t) else t, out)
        return out

    return inner

# a wrapper to slowly absorb actor / critic / world model into one, will just call it OneModelWrapper

class OneModelWrapper(Module):
    def __init__(
        self,
        transformer: Module,
        num_actions,
        critic_dim_pred,
        dim_pred_state,
        frac_actor_critic_head_gradient = 0.5,
    ):
        super().__init__()
        self.transformer = transformer
        dim = transformer.attn_layers.dim

        self.action_embeds = nn.Embedding(num_actions, dim)

        dim = transformer.attn_layers.dim

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
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, critic_dim_pred)
        )

        self.action_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, num_actions),
            nn.Softmax(dim = -1)
        )

        self.frac_actor_critic_head_gradient = frac_actor_critic_head_gradient

    def forward(
        self,
        *args,
        actions = None,
        next_actions = None,
        return_pred_dones = False,
        **kwargs
    ):
        sum_embeds = 0.

        if exists(actions):
            has_actions = actions >= 0.
            actions = torch.where(has_actions, actions, 0)
            action_embeds = self.action_embeds(actions)
            action_embeds = einx.where('b n, b n d, ', has_actions, action_embeds, 0.)
            sum_embeds = sum_embeds + action_embeds

        embed, cache = self.transformer(*args, **kwargs, sum_embeds = sum_embeds, return_embeddings = True, return_intermediates = True)

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

        # actor critic heads living on top of transformer - basically approaching online decision transformer except critic learn discounted returns

        embed = frac_gradient(embed, self.frac_actor_critic_head_gradient) # what fraction of the gradient to pass back to the world model from the actor / critic head

        # actions

        action_probs = self.action_head(embed)

        # values

        values = self.critic_head(embed)

        return action_probs, values, state_pred, dones, cache

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
        hidden_dim = 48,
        world_model: dict = dict(
            attn_dim_head = 16,
            heads = 4,
            depth = 4,
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
        save_path = './ppo.pt'
    ):
        super().__init__()

        self.model_dim = hidden_dim

        state_and_reward_dim = state_dim + 1

        self.model = OneModelWrapper(
            num_actions = num_actions,
            critic_dim_pred = critic_pred_num_bins,
            dim_pred_state = state_and_reward_dim,
            transformer = ContinuousTransformerWrapper(
                dim_in = state_and_reward_dim,
                dim_out = None,
                max_seq_len = max_timesteps,
                probabilistic = True,
                attn_layers = Decoder(
                    dim = hidden_dim,
                    rotary_pos_emb = True,
                    attn_dropout = dropout,
                    ff_dropout = dropout,
                    **world_model
                )
            )
        )

        self.frac_actor_critic_head_gradient = frac_actor_critic_head_gradient

        # state + reward normalization

        self.rsmnorm = RSMNorm(state_dim + 1)

        # https://arxiv.org/abs/2403.03950

        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = reward_range[0],
            max_value = reward_range[1],
            num_bins = critic_pred_num_bins,
            clamp_to_range = True
        )

        self.ema_model = EMA(self.model, beta = ema_decay, include_online_model = False, **ema_kwargs)

        self.optimizer = AdoptAtan2(self.model.parameters(), lr = lr, betas = betas, regen_reg_rate = regen_reg_rate, cautious_factor = cautious_factor)

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
        }, str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path), weights_only = True)

        self.model.load_state_dict(data['model'])

    def learn(self, memories, episode_lens):

        model = self.model
        hl_gauss = self.critic_hl_gauss_loss

        # retrieve and prepare data from memory for training - list[list[Memory]]

        def stack_and_to_device(t):
            return stack(t).to(device)

        def stack_memories(episode_memories):
            return tuple(map(stack_and_to_device, zip(*episode_memories)))

        memories = map(stack_memories, memories)

        episode_lens = episode_lens.to(device)

        (
            episodes,
            states,
            actions,
            old_log_probs,
            rewards,
            is_boundaries,
            values,
            dones
        ) = tuple(map(pad_sequence, zip(*memories)))

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

        # transformer world model is trained on all states per episode all at once
        # will slowly incorporate other ssl objectives + regularizations from the transformer field

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

        rsmnorm_copy = deepcopy(self.rsmnorm) # learn the state normalization alongside in a copy of the state norm module, copy back at the end
        rsmnorm_copy.train()

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

                rewards = F.pad(rewards, (1, -1), value = 0.)

                states_with_rewards, _ = pack((states, rewards), 'b n *')

                with torch.no_grad():
                    self.rsmnorm.eval()
                    states_with_rewards = self.rsmnorm(states_with_rewards)

                action_probs, values, state_and_reward_pred, done_pred, _ = model(
                    states_with_rewards,
                    actions = prev_actions,
                    next_actions = actions, # prediction of the next state needs to be conditioned on the agent's chosen action on that state, and will make the world model interactable
                    mask = mask,
                    return_pred_dones = True
                )

                pred_mean, pred_var = state_and_reward_pred[..., :-1, :] # todo: fix truncation scenario
                world_model_loss = F.gaussian_nll_loss(pred_mean, states_with_rewards[:, 1:], pred_var, reduction = 'none')
                world_model_loss = world_model_loss[mask[:, :-1]]

                # predicting termination head

                pred_done_loss = F.binary_cross_entropy(done_pred, dones.float(), reduction = 'none')
                pred_done_loss = pred_done_loss[mask]

                # update actor and critic

                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                scalar_old_values = hl_gauss(old_values)

                # calculate clipped surrogate objective, classic PPO loss

                ratios = (action_log_probs - old_log_probs).exp()

                advantages = normalize(returns - scalar_old_values.detach())

                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = - torch.min(surr1, surr2) - self.beta_s * entropy

                clip = self.value_clip

                scalar_values = hl_gauss(values)

                # using the proposal from https://www.authorea.com/users/855021/articles/1240083-on-analysis-of-clipped-critic-loss-in-proximal-policy-gradient

                clipped_returns = returns.clamp(-clip, clip)

                clipped_loss = hl_gauss(values, clipped_returns)
                loss = hl_gauss(values, returns)

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

                actor_critic_loss = (actor_loss + critic_loss)[mask]

                loss = world_model_loss.mean() + actor_critic_loss.mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

                rsmnorm_copy(states_with_rewards[mask])

        self.rsmnorm.load_state_dict(rsmnorm_copy.state_dict())

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 500,
    critic_pred_num_bins = 100,
    reward_range = (-100, 100),
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
    render_every_eps = 250,
    save_every = 1000,
    video_folder = './lunar-recording',
    load = False,
):
    assert divisible_by(update_episodes, minibatch_size)

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
    episode_lens = []

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
    ).to(device)

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0

    agent.eval()
    model = agent.ema_model

    for eps in tqdm(range(num_episodes), desc = 'episodes'):

        one_episode_memories = deque([])

        eps_tensor = tensor(eps)

        state, info = env.reset(seed = seed)
        state = torch.from_numpy(state).to(device)

        prev_action = tensor(-1).to(device)
        prev_reward = tensor(0.).to(device)

        world_model_cache = None

        @torch.no_grad()
        def state_to_pred_action_and_value(state, prev_action, prev_reward):
            nonlocal world_model_cache

            state_with_reward = cat((state, rearrange(prev_reward, '-> 1')), dim = -1)

            agent.rsmnorm.eval()
            normed_state = agent.rsmnorm(state_with_reward)

            model.eval()

            normed_state = rearrange(normed_state, 'd -> 1 1 d')
            prev_action = rearrange(prev_action, ' -> 1 1')
            prev_reward = rearrange(prev_reward, ' -> 1 1')

            action_probs, values, _, _, world_model_cache = model.forward_eval(
                normed_state,
                cache = world_model_cache,
                actions = prev_action
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

            next_state = torch.from_numpy(next_state).to(device)

            reward = float(reward)

            prev_action = action
            prev_reward = tensor(reward).to(device) # from the xval paper, we know pre-norm transformers can handle scaled tokens https://arxiv.org/abs/2310.02989

            dones_signal = tensor([terminated, truncated])
            memory = Memory(tensor(eps), state, action, action_log_prob, tensor(reward), tensor(terminated), value, dones_signal)

            one_episode_memories.append(memory)

            state = next_state

            # determine if truncating or terminated

            done = terminated or truncated

            # take care of truncated by adding a non-learnable memory storing the next value for GAE

            if done and not terminated:
                _, next_value, *_ = state_to_pred_action_and_value(state, prev_action, prev_reward)

                bootstrap_value_memory = memory._replace(
                    state = state,
                    eps = tensor(-1),
                    is_boundary = tensor(True),
                    value = next_value
                )

                memories.append(bootstrap_value_memory)

            # break if done

            if done:
                break

        episode_lens.append(timestep + 1)

        # add list[Memory] to all episode memories list[list[Memory]]

        memories.append(one_episode_memories)

        # updating of the agent

        if divisible_by(len(memories), update_episodes):

            agent.learn(memories, tensor(episode_lens))
            num_policy_updates += 1

            memories.clear()
            episode_lens.clear()

        if divisible_by(eps, save_every):
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
