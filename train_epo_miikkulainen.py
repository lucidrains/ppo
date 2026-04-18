from __future__ import annotations

# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "tqdm",
#   "wandb",
#   "gymnasium[box2d]",
#   "moviepy",
#   "fire",
#   "x-mlps-pytorch",
#   "einops"
# ]
# ///

# Evolutionary Policy Optimization
# Mustafaoglu et al. https://arxiv.org/abs/2504.12568

import os
import shutil
import random
import glob
import math
from copy import deepcopy

from tqdm import tqdm
import wandb
import fire

import numpy as np

import torch
import torch.nn.functional as F
from torch import tensor, stack
from torch.nn import Module, Sequential, Softmax
from torch.optim import Adam
from torch.distributions import Categorical

import gymnasium as gym

from einops.layers.torch import Rearrange
from x_mlps_pytorch import MLP
from moviepy import VideoFileClip, clips_array, ColorClip

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def maybe(fn):
    return lambda v: fn(v) if exists(v) else v

def module_device(module):
    if isinstance(module, torch.Tensor):
        return module.device
    return next(module.parameters()).device

def divisible_by(num, den):
    return (num % den) == 0

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# globals

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# networks

class ActorCritic(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim = 64
    ):
        super().__init__()

        self.actor = Sequential(
            MLP(state_dim, hidden_dim, hidden_dim, action_dim),
            Softmax(dim = -1)
        )

        self.critic = Sequential(
            MLP(state_dim, hidden_dim, hidden_dim, 1),
            Rearrange('... 1 -> ...')
        )

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.item(), action_logprob, state_val

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

# agent

class PPOAgent(Module):
    def __init__(
        self,
        config: dict
    ):
        super().__init__()

        self.config = config
        self.entropy_coef = config.get('ppo_entropy_coef', 0.01)

        self.policy = ActorCritic(config['state_dim'], config['action_dim'])
        self.optimizer = Adam(self.policy.parameters(), lr = config['ppo_lr'])

        self.clear_buffer()

    def select_action(self, state):
        device = module_device(self)
        state = tensor(state, dtype = torch.float32, device = device)

        with torch.no_grad():
            action, action_logprob, state_val = self.policy.act(state)

        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_logprobs.append(action_logprob)
        self.buffer_values.append(state_val)

        return action

    def update(self):
        device = module_device(self)

        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(self.buffer_rewards), reversed(self.buffer_is_terminals)):
            if is_terminal:
                discounted_reward = 0

            discounted_reward = reward + (self.config['ppo_gamma'] * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = tensor(rewards, dtype = torch.float32, device = device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = stack(self.buffer_states).to(device).detach()
        old_actions = torch.squeeze(tensor(self.buffer_actions, device = device)).detach()
        old_logprobs = torch.squeeze(stack(self.buffer_logprobs)).to(device).detach()
        old_values = torch.squeeze(stack(self.buffer_values)).to(device).detach()

        advantages = rewards.detach() - old_values.detach()

        for _ in range(self.config['ppo_k_epochs']):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config['ppo_eps_clip'], 1 + self.config['ppo_eps_clip']) * advantages

            policy_loss = -torch.min(surr1, surr2)

            if self.config['use_delightful_gating']:
                surprisal = -logprobs.detach()
                delight = advantages * surprisal
                gate = torch.sigmoid(delight / self.config['delight_temp'])
                policy_loss = policy_loss * gate

            policy_loss = policy_loss.mean()

            value_loss = 0.5 * F.mse_loss(state_values, rewards)
            entropy_bonus = -self.entropy_coef * dist_entropy.mean()

            loss = policy_loss + value_loss + entropy_bonus

            update_network_(loss, self.optimizer)

        self.clear_buffer()

    def clear_buffer(self):
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_is_terminals = []
        self.buffer_values = []

    def get_weights(self):
        return deepcopy(self.policy.state_dict())

    def set_weights(self, weights):
        self.policy.load_state_dict(weights)

# loops

def train_agent_for_steps(agent, config, total_timesteps, show_pbar = False):
    env = gym.make(config['env_id'])
    time = 0
    state, _ = env.reset()

    pbar = tqdm(total = total_timesteps, desc = "pre-training base model") if show_pbar else None

    while time < total_timesteps:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        agent.buffer_rewards.append(reward)
        agent.buffer_is_terminals.append(terminated)

        time += 1

        if exists(pbar):
            pbar.update(1)

        state = next_state
        done = terminated or truncated

        if divisible_by(time, config['ppo_update_steps']):
            agent.update()

        if done:
            state, _ = env.reset()

    if exists(pbar):
        pbar.close()

    if len(agent.buffer_states) > 0:
        agent.update()

    env.close()

def evaluate_agent(agent, config):
    env = gym.make(config['env_id'])
    rewards = []

    for _ in range(config['eval_episodes']):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = tensor(state, dtype = torch.float32, device = device)

            with torch.no_grad():
                action_probs = agent.policy.actor(state_tensor)
                action = torch.argmax(action_probs).item()

            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

    env.close()

    return np.mean(rewards)

# evolution

def crossover(weights1, weights2, fitness1, fitness2, min_fitness, eps = 1e-5):

    scaled_fitness1 = fitness1 - min_fitness + eps
    scaled_fitness2 = fitness2 - min_fitness + eps

    alpha = scaled_fitness1 / (scaled_fitness1 + scaled_fitness2 + eps)

    child_weights = {}

    for key, weight1 in weights1.items():
        if weight1.is_floating_point():
            if weight1.ndim >= 2:
                dim = random.choice([0, 1])
                shape = [1] * weight1.ndim
                shape[dim] = weight1.shape[dim]
                mask = torch.rand(shape, device = weight1.device) < alpha
            else:
                mask = torch.rand_like(weight1) < alpha

            child_weights[key] = torch.where(mask, weight1, weights2[key])
        else:
            child_weights[key] = weight1

    return child_weights, scaled_fitness1, scaled_fitness2

def mutate(weights, fitness1, fitness2, scaled_fitness1, scaled_fitness2, eps = 1e-5):
    scaling_factor = max(0.01, min(0.1, abs(fitness1 - fitness2) / max(scaled_fitness1 + scaled_fitness2, eps)))

    mutated_weights = {}

    for key, weight in weights.items():
        is_learnable = "weight" in key or "bias" in key

        if is_learnable and weight.is_floating_point():
            noise = torch.randn_like(weight) * scaling_factor
            mutated_weights[key] = weight + noise
        else:
            mutated_weights[key] = weight

    return mutated_weights

# migrate - ring topology, shift non-elite individuals to adjacent island

def migrate_islands_(islands, elite_count, num_migrants):
    num_islands = len(islands)
    migrants_per_island = []

    for island in islands:
        non_elites = island[elite_count:]
        random.shuffle(non_elites)

        migrants = non_elites[:num_migrants]
        island[elite_count:] = non_elites[num_migrants:]

        migrants_per_island.append(migrants)

    # ring shift - island i receives migrants from island i-1

    for i in range(num_islands):
        source_idx = (i - 1) % num_islands
        islands[i].extend(migrants_per_island[source_idx])

# record and stitch videos

def record_island_videos(island_fitnesses, config, generation):
    num_islands = config['num_islands']
    video_folder = config['video_folder']
    video_files = []

    for island_idx in range(num_islands):
        best_agent = island_fitnesses[island_idx][0][1]

        eval_env = gym.make(config['env_id'], render_mode = "rgb_array")
        eval_env = gym.wrappers.RecordVideo(
            env = eval_env,
            video_folder = video_folder,
            name_prefix = f"gen-{generation + 1}-island-{island_idx}",
            disable_logger = True
        )

        state, _ = eval_env.reset()
        done = False

        while not done:
            state_tensor = tensor(state, dtype = torch.float32, device = device)

            with torch.no_grad():
                action_probs = best_agent.policy.actor(state_tensor)
                action = torch.argmax(action_probs).item()

            state, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated

        eval_env.close()

        videos = glob.glob(os.path.join(video_folder, f"gen-{generation + 1}-island-{island_idx}*.mp4"))
        if videos:
            video_files.append(max(videos, key = os.path.getctime))

    if not video_files:
        return

    try:
        clips = [VideoFileClip(v).resize(0.5) for v in video_files]

        n = len(clips)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

        w, h = clips[0].size
        duration = max(c.duration for c in clips)

        if rows * cols > n:
            blank_clip = ColorClip(size = (w, h), color = (0, 0, 0), duration = duration)
            clips.extend([blank_clip] * (rows * cols - n))

        grid = [clips[r * cols : (r + 1) * cols] for r in range(rows)]

        final_clip = clips_array(grid)
        stitched_path = os.path.join(video_folder, f"gen-{generation + 1}-stitched.mp4")
        final_clip.write_videofile(stitched_path, logger = None)

        for clip in clips:
            clip.close()
        final_clip.close()

        wandb.log(dict(
            generation = generation,
            evaluation_video = wandb.Video(stitched_path, fps = 30, format = "mp4")
        ))

        for v in video_files:
            try:
                os.remove(v)
            except:
                pass

    except Exception as e:
        print(f"  failed to stitch videos: {e}")
        wandb.log(dict(
            generation = generation,
            evaluation_videos = [wandb.Video(v, fps = 30, format = "mp4") for v in video_files]
        ))

# main

def main(
    env_id = "LunarLander-v3",
    num_islands = 2,
    population_size = 16,
    elite_count = 3,
    mutation_prob = 0.3,
    frac_tournament = 0.25,
    migrate_every = 5,
    frac_migrate = 0.2,
    record_every = 2,
    total_generations = 50,
    pretrain_steps = 25000,
    finetune_steps = 500,
    eval_episodes = 5,
    ppo_lr = 3e-4,
    ppo_gamma = 0.99,
    ppo_eps_clip = 0.2,
    ppo_k_epochs = 4,
    ppo_update_steps = 2000,
    video_folder = "./lunar-recording",
    use_delightful_gating = False,
    delight_temp = 1.0,
    ppo_entropy_coef = 0.01
):
    try:
        gym.make(env_id)
    except gym.error.NameNotFound:
        print(f"environment {env_id} not found, falling back to LunarLander-v2")
        env_id = "LunarLander-v2"

    temp_env = gym.make(env_id)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    temp_env.close()

    num_migrants = int(population_size * frac_migrate)

    config = dict(
        env_id = env_id,
        num_islands = num_islands,
        population_size = population_size,
        elite_count = elite_count,
        mutation_prob = mutation_prob,
        frac_tournament = frac_tournament,
        migrate_every = migrate_every,
        frac_migrate = frac_migrate,
        num_migrants = num_migrants,
        record_every = record_every,
        total_generations = total_generations,
        pretrain_steps = pretrain_steps,
        finetune_steps = finetune_steps,
        eval_episodes = eval_episodes,
        ppo_lr = ppo_lr,
        ppo_gamma = ppo_gamma,
        ppo_eps_clip = ppo_eps_clip,
        ppo_k_epochs = ppo_k_epochs,
        ppo_update_steps = ppo_update_steps,
        use_delightful_gating = use_delightful_gating,
        delight_temp = delight_temp,
        ppo_entropy_coef = ppo_entropy_coef,
        video_folder = video_folder,
        state_dim = state_dim,
        action_dim = action_dim
    )

    wandb.init(
        project = "epo-lunarlander",
        config = config
    )

    print(f"using environment: {config['env_id']}")
    print(f"videos will be saved to: {os.path.abspath(config['video_folder'])}")

    if os.path.exists(config['video_folder']):
        shutil.rmtree(config['video_folder'])
    os.makedirs(config['video_folder'], exist_ok = True)

    # pre-train base model

    base_agent = PPOAgent(config).to(device)
    train_agent_for_steps(base_agent, config, config['pretrain_steps'], show_pbar = True)

    # initialize islands

    base_weights = base_agent.get_weights()

    islands = [
        [PPOAgent(config).to(device) for _ in range(population_size)]
        for _ in range(num_islands)
    ]

    for island in islands:
        for agent in island:
            agent.set_weights(base_weights)

    num_tournament_contenders = max(2, int(elite_count * frac_tournament))
    num_tournament_contenders = min(elite_count, num_tournament_contenders)

    # evolution loop

    for generation in tqdm(range(total_generations), desc = "evolution"):
        island_fitnesses = []
        next_islands = []

        all_best = []
        all_mean = []
        pbar_postfix = {}

        for island_idx, population in enumerate(islands):

            # evaluate fitness

            fitnesses = []

            for agent in population:
                fitness = evaluate_agent(agent, config)
                fitnesses.append((fitness, agent))

            fitnesses.sort(key = lambda x: x[0], reverse = True)
            island_fitnesses.append(fitnesses)

            raw_fitness_vals = [f for f, _ in fitnesses]
            min_fitness = min(raw_fitness_vals)
            best_fitness = raw_fitness_vals[0]
            mean_fitness = np.mean(raw_fitness_vals)

            all_best.append(best_fitness)
            all_mean.append(mean_fitness)
            pbar_postfix[f"i{island_idx + 1}"] = f"{best_fitness:.1f}"

            # elites survive directly

            elites = fitnesses[:elite_count]

            next_population = []

            for _, elite_agent in elites:
                clone = PPOAgent(config).to(device)
                clone.set_weights(elite_agent.get_weights())
                next_population.append(clone)

            # fill rest of population with tournament-selected crossover children

            while len(next_population) < population_size:
                if elite_count >= 2:
                    contenders1 = random.sample(elites, num_tournament_contenders)
                    parent1 = max(contenders1, key = lambda x: x[0])

                    contenders2 = random.sample(elites, num_tournament_contenders)
                    parent2 = max(contenders2, key = lambda x: x[0])
                else:
                    parent1 = parent2 = elites[0]

                fitness1, agent1 = parent1
                fitness2, agent2 = parent2

                child_weights, scaled_fitness1, scaled_fitness2 = crossover(
                    agent1.get_weights(), agent2.get_weights(),
                    fitness1, fitness2, min_fitness
                )

                child = PPOAgent(config).to(device)

                if random.random() < mutation_prob:
                    child.set_weights(mutate(
                        child_weights, fitness1, fitness2,
                        scaled_fitness1, scaled_fitness2
                    ))
                else:
                    child.set_weights(child_weights)
                    train_agent_for_steps(child, config, finetune_steps)

                next_population.append(child)

            next_islands.append(next_population)

        islands = next_islands

        # logging

        tqdm.write(f"gen {generation + 1} | best: {max(all_best):.1f} | mean: {np.mean(all_mean):.1f} | " + " | ".join(f"i{i+1}: {b:.1f}" for i, b in enumerate(all_best)))

        log_dict = dict(
            generation = generation,
            global_max_reward = max(all_best),
            global_mean_reward = np.mean(all_mean),
        )

        for i in range(num_islands):
            log_dict[f"island_{i}_max_reward"] = all_best[i]
            log_dict[f"island_{i}_mean_reward"] = all_mean[i]

        wandb.log(log_dict)

        # periodic migration

        if divisible_by(generation + 1, migrate_every) and num_migrants > 0:
            migrate_islands_(islands, elite_count, num_migrants)

        # periodic video recording

        if divisible_by(generation + 1, record_every):
            record_island_videos(island_fitnesses, config, generation)

    print("evolution complete.")
    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)
