# /// script
# dependencies = [
#   "torch",
#   "einops",
#   "ema-pytorch",
#   "adam-atan2-pytorch",
#   "hl-gauss-pytorch",
#   "assoc-scan",
#   "gymnasium[box2d,other]",
#   "moviepy",
#   "memmap-replay-buffer",
#   "fire",
#   "tqdm",
#   "accelerate",
#   "wandb",
#   "evolutionary-policy-optimization>=0.2.16"
# ]
# ///

from __future__ import annotations

import fire
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
from shutil import rmtree
from collections import deque
from functools import partial

from tqdm import tqdm

from accelerate import Accelerator

from torch import nn, tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.distributions import Categorical

from einops import reduce, rearrange, pack, unpack
from ema_pytorch import EMA
from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2
from hl_gauss_pytorch import HLGaussLoss
from assoc_scan import AssocScan
from memmap_replay_buffer import ReplayBuffer
from evolutionary_policy_optimization import LatentGenePool

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def update_network_(loss, optimizer, network, max_grad_norm = None):
    optimizer.zero_grad()
    loss.mean().backward()

    if exists(max_grad_norm):
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)

    optimizer.step()

# SimBa - Kaist + SonyAI

class ReluSquared(Module):
    def forward(self, x):
        return F.relu(x) ** 2

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
        """
        following the design of SimBa https://arxiv.org/abs/2410.09754v1
        """

        dim_hidden = default(dim_hidden, dim * expansion_factor)

        self.proj_in = nn.Linear(dim, dim_hidden)

        dim_inner = dim_hidden * expansion_factor

        self.layers = ModuleList([])

        for _ in range(depth):

            layer = nn.Sequential(
                nn.RMSNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_inner),
                ReluSquared(),
                nn.Linear(dim_inner, dim_hidden),
                nn.Dropout(dropout),
            )

            self.layers.append(layer)

        # final layer out

        self.final_norm = nn.RMSNorm(dim_hidden)

    def forward(self, x):
        no_batch = x.ndim == 1

        if no_batch:
            x = rearrange(x, '... -> 1 ...')

        x = self.proj_in(x)

        for layer in self.layers:
            x = layer(x) + x

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
        dim_latent = 0,
        mlp_depth = 2,
        dropout = 0.1
    ):
        super().__init__()

        self.net = SimBa(
            state_dim + dim_latent,
            dim_hidden = hidden_dim * 2,
            depth = mlp_depth,
            dropout = dropout
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            ReluSquared(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x, latent = None):

        if exists(latent) and latent.shape[-1] > 0:
            x = torch.cat((x, latent), dim = -1)

        hidden = self.net(x)
        return self.action_head(hidden).softmax(dim = -1)

class Critic(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_actions,
        dim_latent = 0,
        dim_pred = 1,
        mlp_depth = 6,
        dropout = 0.1
    ):
        super().__init__()

        self.net = SimBa(
            state_dim + num_actions + dim_latent,
            dim_hidden = hidden_dim,
            depth = mlp_depth,
            dropout = dropout
        )

        self.value_head = nn.Linear(hidden_dim, dim_pred)

    def forward(self, x, past_action, latent = None):

        inputs = [x, past_action]

        if exists(latent) and latent.shape[-1] > 0:
            inputs.append(latent)

        x = torch.cat(inputs, dim = -1)
        hidden = self.net(x)

        return self.value_head(hidden)

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

    rewards, ps = pack([rewards], '* n')
    values, _ = pack([values], '* n')
    masks, _ = pack([masks], '* n')

    values = F.pad(values, (0, 1), value = 0.)

    values, values_next = values[..., :-1], values[..., 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    ret = gae + values
    return unpack(ret, ps, '* n')[0]

# agent

class PPO(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        dim_latent,
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
        max_grad_norm = 0.5,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1000
        ),
        save_path = './ppo.pt'
    ):
        super().__init__()

        self.actor = Actor(state_dim, actor_hidden_dim, num_actions, dim_latent = dim_latent)

        self.critic = Critic(state_dim, critic_hidden_dim, num_actions, dim_latent = dim_latent, dim_pred = critic_pred_num_bins)

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
        self.max_grad_norm = max_grad_norm

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

    def learn(self, memories: ReplayBuffer, device = None):

        hl_gauss = self.critic_hl_gauss_loss

        # get data

        dl = memories.dataloader(
            batch_size = self.minibatch_size,
            shuffle = True,
            filter_fields = dict(
                learnable = True
            ),
            to_named_tuple = ('state', 'action', 'action_log_prob', 'returns', 'value', 'past_action') + (('latent',) if 'latent' in memories.data else ()),
            timestep_level = True,
            device = device
        )

        # policy phase training, similar to original PPO

        self.actor.train()
        self.critic.train()

        total_policy_loss = 0.
        total_value_loss = 0.
        total_entropy = 0.
        num_batches = 0

        for _ in tqdm(range(self.epochs), desc = 'ppo epochs', leave = False):
            for _, batch in enumerate(tqdm(dl, desc = 'ppo batches', leave = False)):
                batch_dict = batch._asdict()

                states = batch_dict['state']
                actions = batch_dict['action']
                old_log_probs = batch_dict['action_log_prob']
                returns = batch_dict['returns']
                old_values = batch_dict['value']
                past_action = batch_dict['past_action']
                latents = batch_dict.get('latent')

                action_probs = self.actor(states, latent = latents)
                dist = Categorical(action_probs)

                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                scalar_old_values = hl_gauss(old_values)

                # calculate clipped surrogate objective, classic PPO loss

                ratios = (action_log_probs - old_log_probs).exp()
                advantages = normalize(returns - scalar_old_values.detach())

                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - torch.min(surr1, surr2)

                policy_loss = (policy_loss - self.beta_s * entropy).mean()

                update_network_(policy_loss, self.opt_actor, self.actor, max_grad_norm = self.max_grad_norm)

                # update critic

                def is_between(mid, lo, hi):
                    return (lo < mid) & (mid < hi)

                values = self.critic(states, past_action, latent = latents)
                scalar_values = hl_gauss(values)

                clipped_returns = returns.clamp(scalar_old_values - self.value_clip, scalar_old_values + self.value_clip)

                clipped_loss = hl_gauss(values, clipped_returns, reduction = 'none')
                loss = hl_gauss(values, returns, reduction = 'none')

                value_loss = torch.where(
                    is_between(scalar_values, returns, scalar_old_values - self.value_clip) |
                    is_between(scalar_values, scalar_old_values + self.value_clip, returns),
                    0.,
                    torch.min(loss, clipped_loss)
                )

                value_loss = value_loss.mean()

                update_network_(value_loss, self.opt_critic, self.critic, max_grad_norm = self.max_grad_norm)

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        return total_policy_loss / num_batches, total_value_loss / num_batches, total_entropy / num_batches

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 500,
    actor_hidden_dim = 64,
    critic_hidden_dim = 256,
    num_latents = 8,
    dim_latent = 32,
    at_least_timesteps_per_update = 8192,
    num_envs = 8,
    buffer_episodes = 40,
    critic_pred_num_bins = 250,
    reward_range = (-300., 300.),
    minibatch_size = 64,
    lr = 3e-4,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    value_clip = 0.4,
    beta_s = .01,
    regen_reg_rate = 1e-4,
    cautious_factor = 0.1,
    ema_decay = 0.9,
    epochs = 2,
    update_genetic_algorithm_every_n_policy_updates = 4,
    min_episodes_per_latent = 2,
    frac_tournaments = 0.5,
    frac_natural_selected = 0.5,
    num_islands = 1,
    migrate_every = 2,
    seed = None,
    render = True,
    save_every = 1000,
    clear_videos = True,
    video_folder = './lunar-recording',
    load = False,
    memory_path = None,
    rolling_window_size = 100,
    stop_at_reward = None,
    max_grad_norm = 0.5,
    ga_weight_latest_fitness_more = True,
    use_wandb = False,
    wandb_project = 'epo',
    cpu = False
):
    def make_env():
        def _make():
            env = gym.make(env_name)
            env = gym.wrappers.TimeLimit(env, max_episode_steps = max_timesteps)
            return env
        return _make

    accelerator = Accelerator(log_with = 'wandb' if use_wandb else None, cpu = cpu)
    accelerator.init_trackers(wandb_project)

    device = accelerator.device
    torch.autograd.set_detect_anomaly(True)

    env = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])

    if render:
        if clear_videos:
            rmtree(video_folder, ignore_errors = True)

    state_dim = int(env.single_observation_space.shape[0])
    num_actions = int(env.single_action_space.n)

    is_evo = num_latents > 1

    if not is_evo:
        dim_latent = 0

    latent_pool = None
    if is_evo:
        latent_pool = LatentGenePool(
            num_latents = num_latents,
            num_islands = num_islands,
            dim_latent = dim_latent,
            l2norm_latent = True,
            migrate_every = migrate_every,
            apply_genetic_algorithm_every = 1,
            frac_tournaments = frac_tournaments,
            frac_natural_selected = frac_natural_selected
        ).to(device)

    memories_fields = dict(
        learnable = 'bool',
        state = ('float', state_dim),
        action = 'int',
        action_log_prob = 'float',
        reward = 'float',
        is_boundary = 'bool',
        value = ('float', critic_pred_num_bins),
        returns = 'float',
        past_action = ('float', num_actions)
    )

    if dim_latent > 0:
        memories_fields['latent'] = ('float', dim_latent)

    if not exists(memory_path):
        memory_path = f"./{env_name.lower().replace('-v3', '')}-memories/evo-{num_latents}"

    memories = ReplayBuffer(
        memory_path,
        max_episodes = buffer_episodes,
        max_timesteps = 2048,
        fields = memories_fields,
        circular = True,
        overwrite = True
    )

    agent = PPO(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        dim_latent,
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
        max_grad_norm = max_grad_norm
    ).to(device)

    if load:
        agent.load()

    # seeds

    base_seed = default(seed, np.random.randint(0, 1000000))

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0
    generation = 0

    latent_fitnesses = [[] for _ in range(num_latents)] if is_evo else None

    # initialize environment

    states, _ = env.reset(seed = base_seed)
    states = torch.from_numpy(states).to(device)

    past_actions = torch.zeros(num_envs, num_actions).to(device)

    # assign latents to each env initially, cycling through them
    env_latent_ids = None
    latents = None
    latent_id_to_sample = 0

    if is_evo:
        env_latent_ids = torch.arange(num_envs, device = device) % num_latents
        latents = latent_pool(latent_id = env_latent_ids)
        latent_id_to_sample = num_envs % num_latents

    eps_rewards = torch.zeros(num_envs).to(device)
    eps_steps = torch.zeros(num_envs).to(device)

    rolling_reward = deque(maxlen = rolling_window_size)
    rolling_steps = deque(maxlen = rolling_window_size)

    # collector per env

    collectors = [[] for _ in range(num_envs)]

    pbar = tqdm(desc = 'timesteps')

    # main loop

    while time < num_episodes * max_timesteps:

        timesteps_since_update = 0
        envs_done_since_last_update = torch.zeros(num_envs, dtype = torch.bool, device = device)

        while timesteps_since_update < at_least_timesteps_per_update or not envs_done_since_last_update.all():
            pbar.update(num_envs)
            time += num_envs
            timesteps_since_update += num_envs

            with torch.no_grad():
                action_probs = agent.ema_actor.forward_eval(states, latent = latents)
                dist = Categorical(action_probs)

                actions = dist.sample()
                action_log_probs = dist.log_prob(actions)

                values = agent.ema_critic.forward_eval(states, past_actions, latent = latents)

            env_actions = actions.cpu().numpy()

            next_states, rewards, terminateds, truncateds, _ = env.step(env_actions)

            rewards = torch.from_numpy(rewards).float().to(device)
            next_states = torch.from_numpy(next_states).to(device)

            eps_rewards += rewards
            eps_steps += 1

            # store in local collectors

            for i in range(num_envs):
                collectors[i].append(dict(
                    learnable = True,
                    state = states[i],
                    action = actions[i],
                    action_log_prob = action_log_probs[i],
                    reward = rewards[i],
                    is_boundary = terminateds[i],
                    value = values[i],
                    past_action = past_actions[i],
                    latent = latents[i] if exists(latents) else None
                ))

            states = next_states
            past_actions = F.one_hot(actions, num_classes = num_actions).float()

            # check for completions

            for i in range(num_envs):
                done = terminateds[i] or truncateds[i]

                if not done:
                    continue

                envs_done_since_last_update[i] = True

                # calculate GAE and store

                collector = collectors[i]
                ep_len = len(collector)

                ep_rewards = torch.stack([c['reward'] for c in collector])
                ep_values = torch.stack([c['value'] for c in collector])
                ep_masks = torch.ones_like(ep_rewards)

                # hl_gauss converts the bin-based values to scalar values
                hl_gauss = agent.critic_hl_gauss_loss
                scalar_ep_values = hl_gauss(ep_values)

                next_value = 0.

                if truncateds[i]:
                    with torch.no_grad():
                        next_value = hl_gauss(agent.ema_critic.forward_eval(states[i], past_actions[i], latent = latents[i] if exists(latents) else None)).item()

                ep_returns = calc_gae(ep_rewards, scalar_ep_values, ep_masks, lam = lam, gamma = gamma, use_accelerated = False)

                if truncateds[i]:
                    # simple gae bootstrap adjustment for truncation
                    # return = reward + gamma * next_value
                    # add the discounted next value
                    ep_returns += (gamma ** torch.arange(ep_len, device = device).flip(0)) * next_value

                # store in memories

                ep_data = {}

                for key in collector[0].keys():
                    if key == 'latent' and not is_evo:
                        continue

                    entries = [c[key] for c in collector]

                    if not torch.is_tensor(entries[0]):
                        entries = [tensor(e, device = device) for e in entries]

                    ep_data[key] = torch.stack(entries)

                ep_data['returns'] = ep_returns

                memories.store_episode(**ep_data)

                # reset collector and env state

                collectors[i] = []
                past_actions[i] = 0.

                # update fitness

                if is_evo:
                    latent_id = env_latent_ids[i].item()
                    latent_fitnesses[latent_id].append(eps_rewards[i].item())

                # record for rolling statistics

                rolling_reward.append(eps_rewards[i].item())
                rolling_steps.append(eps_steps[i].item())

                eps_rewards[i] = 0.
                eps_steps[i] = 0.

                # systematic new latent

                if is_evo:
                    new_latent_id = latent_id_to_sample
                    env_latent_ids[i] = new_latent_id
                    latents[i] = latent_pool(new_latent_id)

                    latent_id_to_sample = (latent_id_to_sample + 1) % num_latents

            if len(rolling_reward) > 0:
                pbar.set_postfix(
                    reward = f'{np.mean(rolling_reward):.2f}',
                    steps = f'{np.mean(rolling_steps):.1f}'
                )

        # updating of the agent

        policy_loss, value_loss, entropy = agent.learn(memories, device)
        num_policy_updates += 1

        # logging

        log_data = dict(
            policy_loss = policy_loss,
            value_loss = value_loss,
            entropy = entropy,
            reward = np.mean(rolling_reward) if len(rolling_reward) > 0 else 0.,
            steps = np.mean(rolling_steps) if len(rolling_steps) > 0 else 0.
        )

        accelerator.log(log_data)

        # apply genetic algorithm updates

        do_ga_update = is_evo and divisible_by(num_policy_updates, update_genetic_algorithm_every_n_policy_updates)
        do_record_video = render and divisible_by(num_policy_updates, update_genetic_algorithm_every_n_policy_updates)

        if do_ga_update or do_record_video:

            latent_counts = tensor([len(f) for f in latent_fitnesses], device = device) if is_evo else None
            can_update_ga = not is_evo or latent_counts.min() >= min_episodes_per_latent

            if can_update_ga:
                if ga_weight_latest_fitness_more:
                    fitnesses = []
                    for f in latent_fitnesses:
                        if len(f) <= 1:
                            fitnesses.append(np.mean(f))
                            continue
                        w = np.arange(1, len(f) + 1)
                        weighted_mean = np.sum(np.array(f) * w) / np.sum(w)
                        fitnesses.append(weighted_mean)
                    fitness = tensor(fitnesses, dtype = torch.float32, device = device)
                else:
                    fitness = tensor([np.mean(f).astype(np.float32) for f in latent_fitnesses], device = device)

                # logging fitness

                if is_evo:
                    for i, f in enumerate(fitness):
                        accelerator.log(dict([(f'fitness/latent_{i}', f.item())]))

                    accelerator.log(dict(
                        mean = fitness.mean().item(),
                        max = fitness.max().item(),
                        min = fitness.min().item()
                    ))

                # record best individual video

                if do_record_video:
                    best_latent_id = fitness.argmax().item() if is_evo else 0

                    eval_env = gym.make(env_name, render_mode = 'rgb_array')
                    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps = max_timesteps)
                    eval_env = gym.wrappers.RecordVideo(
                        env = eval_env,
                        video_folder = video_folder,
                        name_prefix = f"best-updates-{num_policy_updates}",
                        disable_logger = True
                    )

                    eval_state, _ = eval_env.reset()
                    eval_state = torch.from_numpy(eval_state).to(device)
                    eval_past_action = torch.zeros(num_actions).to(device)
                    eval_latent = latent_pool(latent_id = best_latent_id) if is_evo else None

                    eval_done = False
                    while not eval_done:
                        action_probs = agent.ema_actor.forward_eval(eval_state, latent = eval_latent)
                        dist = Categorical(action_probs)
                        action = dist.sample()

                        next_state, _, terminated, truncated, _ = eval_env.step(action.item())
                        eval_state = torch.from_numpy(next_state).to(device)
                        eval_past_action = F.one_hot(action, num_classes = num_actions).float()
                        eval_done = terminated or truncated

                    eval_env.close()

                    # log video to wandb

                    if use_wandb:
                        import wandb
                        video_paths = list(Path(video_folder).glob(f"best-updates-{num_policy_updates}*.mp4"))
                        if len(video_paths) > 0:
                            accelerator.get_tracker('wandb').log(dict(
                                video = wandb.Video(str(video_paths[0]), fps = 4, format = "mp4")
                            ))

                if do_ga_update:
                    latent_pool.genetic_algorithm_step(fitness)

                    generation += 1
                    accelerator.print(f"\n[Generation {generation}] Fitness Stats: Mean {fitness.mean():.2f} | Max {fitness.max():.2f} | Min {fitness.min():.2f}")
                    accelerator.print(f"[Generation {generation}] Episode counts per latent: {latent_counts.tolist()}")

                    latent_fitnesses = [[] for _ in range(num_latents)]

        if divisible_by(num_policy_updates, save_every // at_least_timesteps_per_update + 1):
            agent.save()

        if exists(stop_at_reward) and len(rolling_reward) >= rolling_window_size and np.mean(rolling_reward) >= stop_at_reward:
            accelerator.print(f"Rolling reward reached {stop_at_reward} at policy update {num_policy_updates}, stopping training.")
            break

    env.close()

if __name__ == '__main__':
    fire.Fire(main)
