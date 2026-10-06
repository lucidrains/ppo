# /// script
# dependencies = [
#   "env-ssl-wrapper",
#   "fire",
#   "gymnasium[box2d,other]",
#   "imageio",
#   "imageio-ffmpeg",
#   "memmap-replay-buffer",
#   "numpy",
#   "torch",
# ]
# ///

# latent-conditioned actor with a DIAYN-style transition discriminator, after DADS - Sharma et al. https://arxiv.org/abs/1907.01657
# q(z | s, a, s') predicts the episode latent with mse, intrinsic reward = -mse
# benefit is measured against the deterministic actor

from __future__ import annotations

from collections import deque, namedtuple
from pathlib import Path

import fire
import numpy as np
import gymnasium as gym

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from einops import rearrange

from env_ssl_wrapper import compose_env, evaluate_actor

from memmap_replay_buffer import ReplayBuffer

from x_ppo import ppo_actor_loss, calc_gae

# helpers

def exists(v):
    return v is not None

def mlp(n_in, n_out, hidden = 64):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, n_out),
    )

# networks

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim, hidden = 64):
        super().__init__()
        self.net = mlp(obs_dim + latent_dim, act_dim, hidden)

    def forward(self, obs, z):
        return self.net(torch.cat((obs, z), dim = -1))

class Critic(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden = 64):
        super().__init__()
        self.net = mlp(obs_dim + latent_dim, 1, hidden)

    def forward(self, obs, z):
        return self.net(torch.cat((obs, z), dim = -1)).squeeze(-1)

class Discriminator(nn.Module):
    # predicts the episode latent from the state transition

    def __init__(self, obs_dim, act_dim, latent_dim, hidden = 64):
        super().__init__()
        self.net = mlp(obs_dim * 2 + act_dim, latent_dim, hidden)

    def forward(self, obs, action_onehot, next_obs):
        return self.net(torch.cat((obs, action_onehot, next_obs), dim = -1))

# envs

def make_env(seed, device, num_envs = 1, render = False, max_episode_steps = 1000):
    env = gym.make_vec(
        'LunarLander-v3',
        num_envs = num_envs,
        max_episode_steps = max_episode_steps,
        render_mode = 'rgb_array' if render else None
    )

    env = compose_env(env, ('tensor', dict(device = device)))
    env.seed(seed)
    return env

# evaluation

Transition = namedtuple('Transition', ('obs', 'action', 'next_obs', 'skill', 'seed'))

@torch.no_grad()
def eval_rollout(actor, num_skills, latent_dim, num_seeds, device, max_episode_steps = 1000):
    # one episode per (skill, seed), recording transitions for the metrics

    env = make_env(10_000, device, render = True, max_episode_steps = max_episode_steps)

    skills = torch.randn(num_skills, latent_dim, device = device)
    returns = np.zeros((num_skills, num_seeds))
    transitions = []

    for skill_id in range(num_skills):
        skill = skills[skill_id].expand(1, -1)

        for seed in range(num_seeds):
            obs, _ = env.reset(seed = 10_000 + seed)
            ep_ret = 0.

            while True:
                action = actor(obs, skill).argmax(dim = -1)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                transitions.append(Transition(
                    obs = obs[0].cpu().numpy(),
                    action = int(action[0]),
                    next_obs = next_obs[0].cpu().numpy(),
                    skill = skill_id,
                    seed = seed
                ))

                obs = next_obs
                ep_ret += reward[0].item()

                if bool((terminated | truncated)[0].item()):
                    break

            returns[skill_id, seed] = ep_ret

    env.close()
    return skills, returns, transitions

@torch.no_grad()
def skill_metrics(transitions, disc, skills, act_dim, num_skills, device):
    # identifiability of the skill from state alone (centroid probe) and from the transition (discriminator mse)

    states = np.stack([t.obs for t in transitions])
    seeds = np.array([t.seed for t in transitions])
    labels = np.array([t.skill for t in transitions])

    states = (states - states.mean(0)) / (states.std(0) + 1e-6)

    train, test = seeds % 2 == 0, seeds % 2 == 1
    centroids = np.stack([states[train & (labels == i)].mean(0) for i in range(num_skills)])
    preds = ((states[test, None] - centroids[None]) ** 2).sum(-1).argmin(-1)
    acc = float((preds == labels[test]).mean())

    all_centroids = np.stack([states[labels == i].mean(0) for i in range(num_skills)])
    state_dist = float(np.linalg.norm(all_centroids[:, None] - all_centroids[None], axis = -1).mean())

    obs = torch.tensor(np.stack([t.obs for t in transitions]), dtype = torch.float32, device = device)
    actions = F.one_hot(torch.tensor([t.action for t in transitions], device = device), act_dim).float()
    next_obs = torch.tensor(np.stack([t.next_obs for t in transitions]), dtype = torch.float32, device = device)
    z = skills[labels]

    disc_mse = float(F.mse_loss(disc(obs, actions, next_obs), z))

    perm = torch.randperm(len(transitions), device = device)
    disc_mse_shuffled = float(F.mse_loss(disc(obs, actions[perm], next_obs), z))

    print(f'skill centroid acc     {acc:6.3f}  (chance {1. / num_skills:.3f})')
    print(f'mean state dist        {state_dist:6.3f}')
    print(f'disc mse real {disc_mse:7.4f}  action-shuffled {disc_mse_shuffled:7.4f}')

    return dict(skill_acc = acc, state_dist = state_dist, disc_mse = disc_mse, disc_mse_shuffled = disc_mse_shuffled)

@torch.no_grad()
def save_skill_videos(actor, skills, returns, latent_dim, device, video, num_videos, max_episode_steps = 1000):
    video = Path(video)
    order = returns.mean(axis = 1).argsort()[::-1][:num_videos] if latent_dim > 0 else [0]

    env = make_env(10_000, device, render = True, max_episode_steps = max_episode_steps)

    for skill_id in order:
        skill = skills[skill_id].expand(1, -1)

        def policy(obs, skill = skill):
            return actor(obs, skill.expand(obs.shape[0], -1))

        path = str(video.with_name(f'{video.stem}_z{skill_id}{video.suffix}'))
        evaluate_actor(policy, env, episodes = 1, seed = 10_000, video_path = path, fps = 30)
        print(f'saved {path}')

    env.close()

def evaluate(
    actor,
    disc,
    act_dim,
    latent_dim,
    device,
    num_skills,
    num_seeds,
    video = None,
    video_skills = 4,
    max_episode_steps = 1000
):
    skills, returns, transitions = eval_rollout(actor, num_skills, latent_dim, num_seeds, device, max_episode_steps)

    eval_return = float(returns.mean())
    print(f'eval extrinsic return  {eval_return:+8.1f}')

    summary = dict(eval_return = eval_return)

    if latent_dim > 0:
        summary |= skill_metrics(transitions, disc, skills, act_dim, num_skills, device)

    if exists(video):
        save_skill_videos(actor, skills, returns, latent_dim, device, video, video_skills, max_episode_steps)

    return summary

# updates

def update_actor_critic(actor, critic, opt_actor, opt_critic, batch, z, clip, vf_coef, ent_coef):
    dist = Categorical(logits = actor(batch.state, z))

    policy_loss = ppo_actor_loss(
        dist.log_prob(batch.action.long()),
        batch.action_log_prob,
        batch.advantage,
        clip,
        normalize_advantages = True
    ).mean()

    value_loss = F.mse_loss(critic(batch.state, z), batch.returns)
    loss = policy_loss + vf_coef * value_loss - ent_coef * dist.entropy().mean()

    opt_actor.zero_grad()
    opt_critic.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
    opt_actor.step()
    opt_critic.step()

def update_discriminator(disc, opt_disc, disc_init, batch, z, act_dim, regen_reg_rate):
    action_onehot = F.one_hot(batch.action.long(), act_dim).float()
    loss = F.mse_loss(disc(batch.state, action_onehot, batch.next_state), z)

    opt_disc.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(disc.parameters(), 0.5)
    opt_disc.step()

    # pull the discriminator back toward init - regenerative regularization, Kumar et al.

    if exists(disc_init):
        with torch.no_grad():
            for name, param in disc.named_parameters():
                param.lerp_(disc_init[name], regen_reg_rate)

    return loss

# main

def main(
    mode = 'latent',            # det (no latent, extrinsic only) | latent (discriminator intrinsic reward)
    latent_dim = 2,
    beta = 0.1,
    extrinsic = 1.0,
    disc_hidden = 64,
    regen_reg_rate = 1e-4,      # regenerative regularization, discriminator only - Kumar et al. https://arxiv.org/abs/2308.11958
    total_steps = 400_000,
    max_episode_steps = 1000,
    num_envs = 8,
    num_steps = 256,
    epochs = 10,
    batch_size = 64,
    clip = 0.2,
    lr = 3e-4,
    gamma = 0.99,
    lam = 0.95,
    vf_coef = 0.5,
    ent_coef = 0.01,
    seed = 0,
    out = None,
    device = 'auto',            # auto | cpu | mps
    memmap_folder = './la-memories',
    eval_skills = 8,
    eval_seeds = 4,
    video = None
):
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    latent_dim = latent_dim if mode == 'latent' else 0

    env = make_env(seed, device, num_envs = num_envs, max_episode_steps = max_episode_steps)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.n

    actor = Actor(obs_dim, act_dim, latent_dim).to(device)
    critic = Critic(obs_dim, latent_dim).to(device)
    disc = Discriminator(obs_dim, act_dim, latent_dim, hidden = disc_hidden).to(device) if latent_dim > 0 else None

    opt_actor = torch.optim.Adam(actor.parameters(), lr = lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr = lr)
    opt_disc = torch.optim.Adam(disc.parameters(), lr = lr) if exists(disc) else None

    # snapshot the discriminator at init for the regenerative regularization

    disc_init = None
    if exists(disc) and regen_reg_rate > 0.:
        disc_init = {name: param.detach().clone() for name, param in disc.named_parameters()}

    fields = dict(
        learnable = 'bool',
        state = ('float', obs_dim),
        action = 'int',
        action_log_prob = 'float',
        reward = 'float',
        is_boundary = 'bool',
        terminated = 'bool',
        value = 'float',
        returns = 'float',
        advantage = 'float',
        next_state = ('float', obs_dim),
    )

    if latent_dim > 0:
        fields['latent'] = ('float', latent_dim)

    memories = ReplayBuffer(
        memmap_folder,
        max_episodes = num_envs,
        max_timesteps = num_steps,
        fields = fields,
        circular = True,
        overwrite = True
    )

    print(f'mode {mode}  latent_dim {latent_dim}  beta {beta}  regen_reg_rate {regen_reg_rate}  seed {seed}')

    obs, _ = env.reset()
    z = torch.randn(num_envs, latent_dim, device = device)
    ep_ext = torch.zeros(num_envs, device = device)
    running = deque(maxlen = 100)
    gstep, n_eps, best = 0, 0, -1e9
    curve = []

    while gstep < total_steps:
        collector = memories.create_rollout_collector(num_groups = num_envs)

        for _ in range(num_steps):
            with torch.no_grad():
                dist = Categorical(logits = actor(obs, z))
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = critic(obs, z)

            next_obs, r_env, terminated, truncated, _ = env.step(action)
            finished = terminated | truncated

            # intrinsic reward - negative prediction error of the discriminator on the transition

            r_int = torch.zeros(num_envs, device = device)
            if exists(disc):
                with torch.no_grad():
                    z_hat = disc(obs, F.one_hot(action, act_dim).float(), next_obs)
                    r_int = -F.mse_loss(z_hat, z, reduction = 'none').mean(dim = -1)

            collector.append(
                learnable = ~finished,
                state = obs,
                action = action,
                action_log_prob = log_prob,
                reward = extrinsic * r_env + beta * r_int,
                is_boundary = finished,
                terminated = terminated,
                value = value,
                next_state = next_obs,
                **({'latent': z} if exists(disc) else {})
            )

            ep_ext += r_env

            if finished.any():
                z = z.clone()  # entries already collected must keep their z
                for i in finished.nonzero().flatten():
                    n_eps += 1
                    running.append(float(ep_ext[i]))
                    ep_ext[i] = 0.
                    z[i] = torch.randn(latent_dim, device = device)

            obs = next_obs
            gstep += num_envs

        # generalized advantage estimation via x_ppo

        with torch.no_grad():
            next_value = critic(obs, z)

            rewards = collector.env_major('reward')
            values = collector.env_major('value')
            masks = (~collector.env_major('terminated')).float()

            gae = calc_gae(
                rewards,
                values,
                masks = masks,
                gamma = gamma,
                lam = lam,
                next_value = next_value,
                use_accelerated = False,
                return_advantages = True
            )

            collector.extend(
                returns = rearrange(gae.returns, 'n t -> t n'),
                advantage = rearrange(gae.advantages, 'n t -> t n')
            )

        collector.store()

        # actor, critic and discriminator updates

        batch_fields = ('state', 'action', 'action_log_prob', 'advantage', 'returns', 'value', 'next_state')
        batch_fields += ('latent',) if latent_dim > 0 else ()

        dl = memories.dataloader(
            batch_size = batch_size,
            shuffle = True,
            timestep_level = True,
            filter_fields = dict(learnable = True),
            to_named_tuple = batch_fields,
            device = device
        )

        total_disc, disc_steps = 0., 0

        for _ in range(epochs):
            for batch in dl:
                z_b = batch.latent if latent_dim > 0 else batch.state.new_zeros((batch.state.shape[0], 0))

                update_actor_critic(actor, critic, opt_actor, opt_critic, batch, z_b, clip, vf_coef, ent_coef)

                if exists(disc):
                    disc_loss = update_discriminator(disc, opt_disc, disc_init, batch, z_b, act_dim, regen_reg_rate)
                    total_disc += disc_loss.item()
                    disc_steps += 1

        disc_mse = total_disc / disc_steps if disc_steps > 0 else float('nan')
        best = max(best, max(running) if running else -1e9)
        mean100 = float(np.mean(running)) if running else float('nan')
        curve.append((gstep, mean100, disc_mse))
        print(f'step {gstep:7d}  eps {n_eps:5d}  ext_avg100 {mean100:+8.1f}  best {best:+8.1f}  disc_mse {disc_mse:7.4f}')

    env.close()

    if exists(out):
        np.savetxt(out, np.array(curve), delimiter = ',', header = 'step,ext_avg100,disc_mse', comments = '')

    summary = evaluate(
        actor,
        disc,
        act_dim,
        latent_dim,
        device,
        eval_skills,
        eval_seeds,
        video = video,
        max_episode_steps = max_episode_steps
    )
    summary.update(mode = mode, latent_dim = latent_dim, beta = beta, extrinsic = extrinsic, seed = seed, train_avg100 = mean100)

    return summary

if __name__ == '__main__':
    fire.Fire(main)
