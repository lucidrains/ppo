from __future__ import annotations

# DIAYN - Diversity is All You Need
# https://arxiv.org/abs/1802.06070

import math
import fire
from pathlib import Path
from collections import deque, namedtuple

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.distributions import Categorical

from einops import reduce, rearrange

from ema_pytorch import EMA

from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from hl_gauss_pytorch import HLGaussLoss

from hyper_connections import ManifoldConstrainedHyperConnections

from assoc_scan import AssocScan

import gymnasium as gym

from memmap_replay_buffer import ReplayBuffer

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Losses = namedtuple('Losses', ('policy', 'value', 'discriminator'))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def normalize(t, eps = 1e-5):
    if t.numel() <= 1:
        return torch.zeros_like(t)
    return (t - t.mean()) / (t.std(unbiased = False) + eps)

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

def skill_one_hot(skill_int, num_skills, device = None):
    return F.one_hot(tensor(skill_int, device = device), num_classes = num_skills).float()

# RSM Norm
# proposed by SimBa https://arxiv.org/abs/2410.09754

class RSMNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_variance', torch.ones(dim))

    def forward(self, x):
        assert x.shape[-1] == self.dim

        mean = self.running_mean
        variance = self.running_variance

        normed = (x - mean) / variance.sqrt().clamp(min = self.eps)

        if not self.training:
            return normed

        # update running mean and variance

        with torch.no_grad():
            time = self.step.item()

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
    """ following the design of SimBa https://arxiv.org/abs/2410.09754v1 """

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

        dim_hidden = default(dim_hidden, dim * expansion_factor)
        dim_inner = dim_hidden * expansion_factor

        self.proj_in = nn.Linear(dim, dim_hidden)

        # hyper connections

        init_hyper_conn, self.expand_stream, self.reduce_stream = ManifoldConstrainedHyperConnections.get_init_and_expand_reduce_stream_functions(1, num_fracs = num_residual_streams, sinkhorn_iters = 2)

        layers = []

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
        num_skills,
        mlp_depth = 2,
        dropout = 0.1,
        rsmnorm_input = True,
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        dim_hidden = hidden_dim * 2

        self.net = SimBa(
            state_dim + num_skills,
            dim_hidden = dim_hidden,
            depth = mlp_depth,
            dropout = dropout
        )

        self.action_head = nn.Sequential(
            nn.Linear(dim_hidden, hidden_dim),
            ReluSquared(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x, skill):
        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        x = torch.cat((x, skill), dim = -1)
        hidden = self.net(x)

        return self.action_head(hidden).softmax(dim = -1)

class Critic(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_skills,
        dim_pred = 1,
        mlp_depth = 6,
        dropout = 0.1,
        rsmnorm_input = True,
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim + num_skills,
            dim_hidden = hidden_dim,
            depth = mlp_depth,
            dropout = dropout
        )

        self.value_head = nn.Linear(hidden_dim, dim_pred)

    def forward(self, x, skill):
        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        x = torch.cat((x, skill), dim = -1)
        hidden = self.net(x)

        return self.value_head(hidden)

class Discriminator(Module):
    """ predicts skill from state alone - used for DIAYN diversity bonus """

    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_skills,
        mlp_depth = 2,
        dropout = 0.1,
        rsmnorm_input = True,
    ):
        super().__init__()
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim,
            dim_hidden = hidden_dim,
            depth = mlp_depth,
            dropout = dropout
        )

        self.logits_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ReluSquared(),
            nn.Linear(hidden_dim, num_skills)
        )

    def forward(self, x):
        with torch.no_grad():
            self.rsmnorm.eval()
            x = self.rsmnorm(x)

        hidden = self.net(x)
        return self.logits_head(hidden)

# GAE via associative scan

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

class PPO(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        num_skills,
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

        self.num_skills = num_skills

        # networks

        self.actor = Actor(state_dim, actor_hidden_dim, num_actions, num_skills)
        self.critic = Critic(state_dim, critic_hidden_dim, num_skills, dim_pred = critic_pred_num_bins)
        self.discriminator = Discriminator(state_dim, critic_hidden_dim, num_skills)

        # weight tie rsmnorm across all networks

        self.rsmnorm = self.actor.rsmnorm
        self.critic.rsmnorm = self.rsmnorm
        self.discriminator.rsmnorm = self.rsmnorm

        # hl gauss for distributional critic - https://arxiv.org/abs/2403.03950

        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = reward_range[0],
            max_value = reward_range[1],
            num_bins = critic_pred_num_bins,
            clamp_to_range = True
        )

        # ema

        self.ema_actor = EMA(self.actor, beta = ema_decay, include_online_model = False, **ema_kwargs)
        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False, **ema_kwargs)
        self.ema_discriminator = EMA(self.discriminator, beta = ema_decay, include_online_model = False, **ema_kwargs)

        # optimizers

        opt_kwargs = dict(lr = lr, betas = betas, regen_reg_rate = regen_reg_rate, cautious_factor = cautious_factor)

        self.opt_actor = AdoptAtan2(self.actor.parameters(), **opt_kwargs)
        self.opt_critic = AdoptAtan2(self.critic.parameters(), **opt_kwargs)
        self.opt_discriminator = AdoptAtan2(self.discriminator.parameters(), **opt_kwargs)

        self.ema_actor.add_to_optimizer_post_step_hook(self.opt_actor)
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)
        self.ema_discriminator.add_to_optimizer_post_step_hook(self.opt_discriminator)

        # learning hparams

        self.minibatch_size = minibatch_size
        self.epochs = epochs

        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.use_spo = use_spo
        self.asymmetric_spo = asymmetric_spo  # https://arxiv.org/abs/2510.06062v1

        self.save_path = Path(save_path)

    def save(self):
        torch.save(dict(
            actor = self.actor.state_dict(),
            critic = self.critic.state_dict(),
            discriminator = self.discriminator.state_dict()
        ), str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path), weights_only = True)

        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        self.discriminator.load_state_dict(data.get('discriminator', self.discriminator.state_dict()))

    @torch.no_grad()
    def calc_diayn_reward(self, state, skill_int):
        """ log q(z|s) - log p(z), the DIAYN diversity bonus """

        if state.ndim == 1:
            state = rearrange(state, '... -> 1 ...')
            skill_int = tensor([skill_int], device = state.device)

        logits = self.ema_discriminator.forward_eval(state)
        log_qz_s = logits.log_softmax(dim = -1)

        log_pz = math.log(1. / logits.shape[-1])
        reward = log_qz_s[torch.arange(state.shape[0]), skill_int] - log_pz

        return reward.squeeze()

    def learn(self, memories: ReplayBuffer, device = None) -> Losses:

        hl_gauss = self.critic_hl_gauss_loss

        # calculate generalized advantage estimate

        dl = memories.dataloader(
            batch_size = 4,
            return_indices = True,
            to_named_tuple = ('_index', 'is_boundary', 'value', 'reward'),
            device = device
        )

        for indices, is_boundaries, values, rewards in dl:
            with torch.no_grad():
                masks = 1. - is_boundaries.float()
                scalar_values = hl_gauss(values)

                returns = calc_gae(
                    rewards = rewards,
                    masks = masks,
                    lam = self.lam,
                    gamma = self.gamma,
                    values = scalar_values,
                    use_accelerated = False
                )

                memories.data['returns'][indices, :returns.shape[-1]] = returns.cpu().numpy()
                memories.flush()

        # get data

        dl = memories.dataloader(
            batch_size = self.minibatch_size,
            shuffle = True,
            filter_fields = dict(learnable = True),
            to_named_tuple = ('state', 'action', 'action_log_prob', 'returns', 'value', 'skill'),
            timestep_level = True,
            device = device
        )

        # policy phase training

        self.actor.train()
        self.critic.train()
        self.discriminator.train()

        total_policy_loss = 0.
        total_value_loss = 0.
        total_discrim_loss = 0.
        steps = 0

        for _ in range(self.epochs):
            for _, (states, actions, old_log_probs, returns, old_values, skill) in enumerate(dl):

                steps += 1
                skills = F.one_hot(skill.long(), num_classes = self.num_skills).float()

                # actor

                action_probs = self.actor(states, skills)
                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                scalar_old_values = hl_gauss(old_values)

                # clipped surrogate objective

                ratios = (action_log_probs - old_log_probs).exp()
                advantages = normalize(returns - scalar_old_values.detach())

                if self.use_spo or self.asymmetric_spo:
                    spo_policy_loss = -(
                        ratios * advantages -
                        (advantages.abs() * (ratios - 1.).square()) / (2 * self.eps_clip)
                    )

                if not self.use_spo or self.asymmetric_spo:
                    surr1 = ratios * advantages
                    surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    ppo_policy_loss = -torch.min(surr1, surr2)

                if self.asymmetric_spo:
                    policy_loss = torch.where(advantages > 0, ppo_policy_loss, spo_policy_loss)
                elif self.use_spo:
                    policy_loss = spo_policy_loss
                else:
                    policy_loss = ppo_policy_loss

                policy_loss = policy_loss - self.beta_s * entropy

                update_network_(policy_loss, self.opt_actor)
                total_policy_loss += policy_loss.mean().item()

                # critic - clipped value loss
                # https://www.authorea.com/users/855021/articles/1240083-on-analysis-of-clipped-critic-loss-in-proximal-policy-gradient

                clip = self.value_clip
                values = self.critic(states, skills)
                scalar_values = hl_gauss(values)

                clipped_returns = returns.clamp(scalar_old_values - clip, scalar_old_values + clip)

                clipped_loss = hl_gauss(values, clipped_returns, reduction = 'none')
                loss = hl_gauss(values, returns, reduction = 'none')

                old_values_lo = scalar_old_values - clip
                old_values_hi = scalar_old_values + clip

                is_between = lambda mid, lo, hi: (lo < mid) & (mid < hi)

                value_loss = torch.where(
                    is_between(scalar_values, returns, old_values_lo) |
                    is_between(scalar_values, old_values_hi, returns),
                    0.,
                    torch.min(loss, clipped_loss)
                ).mean()

                update_network_(value_loss, self.opt_critic)
                total_value_loss += value_loss.item()

                # discriminator

                discrim_logits = self.discriminator(states)
                discrim_loss = F.cross_entropy(discrim_logits, skill.long())

                update_network_(discrim_loss, self.opt_discriminator)
                total_discrim_loss += discrim_loss.item()

        # update rsmnorm running statistics

        self.rsmnorm.train()

        for states, *_ in dl:
            self.rsmnorm(states)

        return Losses(
            total_policy_loss / max(1, steps),
            total_value_loss / max(1, steps),
            total_discrim_loss / max(1, steps),
        )

# evaluation

@torch.no_grad()
def evaluate_skills(agent, env, num_skills, max_timesteps, device, seed = None):
    """ run one greedy episode per skill, return list of frame arrays (T, C, H, W) """

    videos = []

    for z in range(num_skills):
        state, _ = env.reset(seed = seed)
        frames = [env.render()]
        skill = skill_one_hot(z, num_skills, device = device)

        for _ in range(max_timesteps):
            state_t = torch.from_numpy(state).to(device)
            action_probs = agent.ema_actor.forward_eval(state_t, skill)
            action = action_probs.argmax(dim = -1).item()

            state, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())

            if terminated or truncated:
                break

        frames = np.array(frames)
        frames = rearrange(frames, 't h w c -> t c h w')
        videos.append(frames)

    return videos

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 10000,
    max_timesteps = 500,
    num_skills = 5,
    actor_hidden_dim = 64,
    critic_hidden_dim = 256,
    update_timesteps = 5000,
    buffer_episodes = 40,
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
    use_spo = False,
    asymmetric_spo = False,
    cautious_factor = 0.1,
    ema_decay = 0.9,
    epochs = 2,
    seed = None,
    render = True,
    render_every_eps = 100,
    save_every = 1000,
    load = False,
    diayn_weight = 0.1,
    wandb_project: str | None = 'lunar-diayn'
):
    # wandb

    if exists(wandb_project):
        import wandb
        from types import ModuleType
        config = {k: v for k, v in locals().items() if not callable(v) and not isinstance(v, (type, ModuleType))}
        wandb.init(project = wandb_project, name = f'diayn_skills_{num_skills}', config = config)
    else:
        wandb = None

    # environments

    env = gym.make(env_name)

    eval_env = None
    if render:
        eval_env = gym.make(env_name, render_mode = 'rgb_array')

    state_dim = int(env.observation_space.shape[0])
    num_actions = int(env.action_space.n)

    # replay buffer

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
            skill = 'int',
        ),
        circular = True,
        overwrite = True
    )

    # agent

    agent = PPO(
        state_dim,
        num_actions,
        num_skills,
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
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    # training loop

    time = 0
    num_policy_updates = 0
    recent_rewards = deque(maxlen = 100)
    last_losses = None

    pbar = tqdm(range(num_episodes), desc = 'episodes')

    for eps in pbar:

        state, _ = env.reset(seed = seed)
        state = torch.from_numpy(state).to(device)

        skill_z = eps % num_skills
        skill = skill_one_hot(skill_z, num_skills, device = device)

        eps_real_reward = 0.
        eps_pseudo_reward = 0.

        with memories.one_episode():
            for timestep in range(max_timesteps):
                time += 1

                action_probs = agent.ema_actor.forward_eval(state, skill)
                value = agent.ema_critic.forward_eval(state, skill)

                dist = Categorical(action_probs)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                next_state = torch.from_numpy(next_state).to(device)

                reward = float(reward)
                eps_real_reward += reward

                # diayn diversity bonus

                pseudo_reward = agent.calc_diayn_reward(next_state, skill_z).item()
                eps_pseudo_reward += pseudo_reward

                reward = reward + diayn_weight * pseudo_reward

                memory = memories.store(
                    learnable = True,
                    state = state,
                    action = action,
                    action_log_prob = action_log_prob,
                    reward = reward,
                    is_boundary = terminated,
                    value = value,
                    skill = skill_z
                )

                state = next_state

                # truncation handling

                updating_agent = divisible_by(time, update_timesteps)
                done = terminated or truncated or updating_agent

                if done and not terminated:
                    next_value = agent.ema_critic.forward_eval(state, skill)

                    bootstrap = memory._replace(
                        state = state,
                        learnable = False,
                        is_boundary = True,
                        value = next_value,
                        skill = skill_z
                    )

                    memories.store(**bootstrap._asdict())

                # learn

                if updating_agent:
                    last_losses = agent.learn(memories, device)
                    num_policy_updates += 1

                    if exists(wandb):
                        wandb.log(dict(
                            policy_loss = last_losses.policy,
                            value_loss = last_losses.value,
                            discriminator_loss = last_losses.discriminator,
                        ), step = time)

                if done:
                    break

        # logging

        recent_rewards.append(eps_real_reward)
        avg_reward = sum(recent_rewards) / len(recent_rewards)

        postfix = dict(avg_reward = f'{avg_reward:.1f}')

        if exists(last_losses):
            postfix.update(
                p_loss = f'{last_losses.policy:.3f}',
                v_loss = f'{last_losses.value:.3f}',
                d_loss = f'{last_losses.discriminator:.3f}'
            )

        pbar.set_postfix(**postfix)

        if exists(wandb):
            wandb.log({
                'reward/real': eps_real_reward,
                'reward/pseudo': eps_pseudo_reward,
                'reward/avg_real_100': avg_reward,
                'episode': eps,
            }, step = time)

        # eval videos

        if render and eps > 0 and divisible_by(eps, render_every_eps) and exists(eval_env):
            videos = evaluate_skills(agent, eval_env, num_skills, max_timesteps, device, seed)

            if exists(wandb):
                wandb.log({
                    f'eval_video/skill_{z}': wandb.Video(vid, fps = 30, format = 'mp4')
                    for z, vid in enumerate(videos)
                }, step = time)

        # save

        if divisible_by(eps, save_every):
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
