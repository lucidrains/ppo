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
from copy import deepcopy

from tqdm import tqdm
import wandb
import fire

import numpy as np

import torch
import torch.nn.functional as F
from torch import tensor, stack, nn
from torch.nn import Module, Sequential, Softmax
from torch.optim import Adam
from torch.distributions import Categorical

import gymnasium as gym

from einops.layers.torch import Rearrange
from x_mlps_pytorch import MLP
from moviepy import VideoFileClip, clips_array, ColorClip
import math

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# globals

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# flat param helpers for CEM

def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(flat_params[offset:offset + numel].view(param.size()))
        offset += numel

# cross-entropy method for meta-evolution

class CEM:
    def __init__(
        self,
        init_params,
        pop_size,
        num_elite,
        tau = 0.5,
        sigma_init = 0.05,
        sigma_end = 0.001
    ):
        self.mu = init_params.clone()
        self.pop_size = pop_size
        self.num_elite = num_elite
        self.tau = tau
        self.sigma_init = sigma_init
        self.sigma_end = sigma_end
        self.epsilon = torch.ones_like(self.mu) * sigma_init
        self.sigma = self.epsilon.sqrt()

    def sample(self):
        noise = torch.randn(self.pop_size, *self.mu.shape, device = self.mu.device)
        return [self.mu + self.sigma * noise[i] for i in range(self.pop_size)]

    def update(self, samples, fitnesses):
        elite_indices = np.argsort(fitnesses)[::-1][:self.num_elite]
        elite_samples = torch.stack([samples[i] for i in elite_indices])

        self.mu = elite_samples.mean(dim = 0)
        self.epsilon = self.tau * self.epsilon + (1 - self.tau) * self.sigma_end
        self.sigma = (elite_samples.var(dim = 0, unbiased = False) + self.epsilon).sqrt()

# meta temperature - single parameter evolved by CEM to predict delightful gate temperature

class MetaTemperature(Module):
    def __init__(self, min_bound = -3., max_bound = 3.):
        super().__init__()
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self):
        return (self.param.sigmoid() * (self.max_bound - self.min_bound) + self.min_bound).exp()

# meta state tracker

class MetaState:
    def __init__(self, cem, pop_size):
        self.cem = cem
        self.pop_size = pop_size
        self.resample()

    def resample(self):
        self.sampled_params = self.cem.sample()
        self.meta_temps = []

        for params in self.sampled_params:
            temp_module = MetaTemperature().to(device)
            set_flat_params(temp_module, params)
            self.meta_temps.append(temp_module)

        self.fitnesses = [0.] * self.pop_size
        self.usage_counts = [0] * self.pop_size
        self.current_idx = 0

    @property
    def current_meta_temp(self):
        return self.meta_temps[self.current_idx]

    def record_fitness(self, fitness):
        self.fitnesses[self.current_idx] += fitness
        self.usage_counts[self.current_idx] += 1
        self.current_idx += 1

        if self.current_idx < self.pop_size:
            return

        # all meta-mlps evaluated, update CEM and resample

        avg_fitnesses = [
            f / max(1, c) for f, c in zip(self.fitnesses, self.usage_counts)
        ]

        self.cem.update(self.sampled_params, avg_fitnesses)
        self.resample()

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
        state = tensor(state, dtype = torch.float32, device = device)

        with torch.no_grad():
            action, action_logprob, state_val = self.policy.act(state)

        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_logprobs.append(action_logprob)
        self.buffer_values.append(state_val)

        return action

    def update(self, meta_temp = None):
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

                if exists(meta_temp):
                    with torch.no_grad():
                        temp = meta_temp()
                else:
                    temp = self.config['delight_temp']

                gate = torch.sigmoid(delight / temp)
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

def train_agent_for_steps(agent, config, total_timesteps, show_pbar = False, meta_temp = None):
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

        if (time % config['ppo_update_steps']) == 0:
            agent.update(meta_temp = meta_temp)

        if done:
            state, _ = env.reset()

    if exists(pbar):
        pbar.close()

    if len(agent.buffer_states) > 0:
        agent.update(meta_temp = meta_temp)

    env.close()

def evaluate_agent(agent, config):
    env = gym.make(config['env_id'])
    rewards = []
    entropies = []

    for _ in range(config['eval_episodes']):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = tensor(state, dtype = torch.float32, device = device)

            with torch.no_grad():
                action_probs = agent.policy.actor(state_tensor)
                dist = Categorical(action_probs)
                action = torch.argmax(action_probs).item()
                entropies.append(dist.entropy().item())

            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

    env.close()

    return np.mean(rewards), np.mean(entropies)

# evolution

def crossover(weights1, weights2, fitness1, fitness2, min_fitness, eps = 1e-5):
    scaled_fitness1 = fitness1 - min_fitness + eps
    scaled_fitness2 = fitness2 - min_fitness + eps

    alpha = scaled_fitness1 / (scaled_fitness1 + scaled_fitness2 + eps)

    child_weights = {}

    for key, weight1 in weights1.items():
        if weight1.is_floating_point():
            child_weights[key] = alpha * weight1 + (1 - alpha) * weights2[key]
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

# main

def main(
    env_id = "LunarLander-v3",
    population_size = 8,
    elite_count = 3,
    mutation_prob = 0.3,
    frac_tournament = 0.25,
    record_every = 2,
    total_generations = 60,
    pretrain_steps = 25000,
    finetune_steps = 750,
    eval_episodes = 5,
    ppo_lr = 3e-4,
    ppo_gamma = 0.99,
    ppo_eps_clip = 0.2,
    ppo_k_epochs = 4,
    ppo_update_steps = 2000,
    video_folder = "./lunar-recording",
    use_delightful_gating = True,
    delight_temp = 1.0,
    ppo_entropy_coef = 0.01,
    use_meta_evolution = True,
    meta_pop_size = 8,
    meta_num_elite = 3,
    meta_kappa = 1.0
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

    if use_meta_evolution:
        use_delightful_gating = True

    config = dict(
        env_id = env_id,
        population_size = population_size,
        elite_count = elite_count,
        mutation_prob = mutation_prob,
        frac_tournament = frac_tournament,
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
        action_dim = action_dim,
        use_meta_evolution = use_meta_evolution,
        meta_pop_size = meta_pop_size,
        meta_num_elite = meta_num_elite,
        meta_kappa = meta_kappa
    )

    wandb.init(
        project = "epo-lunarlander",
        config = config
    )

    print(f"using environment: {config['env_id']}")
    print(f"videos will be saved to: {os.path.abspath(config['video_folder'])}")

    if os.path.exists(config['video_folder']):
        shutil.rmtree(config['video_folder'])
    os.makedirs(config['video_folder'], exist_ok=True)

    print("pre-training base model...")
    base_agent = PPOAgent(config).to(device)
    train_agent_for_steps(base_agent, config, config['pretrain_steps'], show_pbar = True)

    # initialize meta-evolution

    meta_state = None

    if use_meta_evolution:
        dummy_meta = MetaTemperature().to(device)
        init_meta_params = get_flat_params(dummy_meta)

        meta_state = MetaState(
            cem = CEM(init_meta_params, meta_pop_size, meta_num_elite),
            pop_size = meta_pop_size
        )

    print("initializing population...")
    population = []
    base_weights = base_agent.get_weights()

    for _ in range(config['population_size']):
        clone = PPOAgent(config).to(device)
        clone.set_weights(base_weights)
        population.append(clone)

    for generation in range(config['total_generations']):
        print(f"generation {generation + 1}/{config['total_generations']}")

        fitnesses = []

        for agent in population:
            fitness, entropy = evaluate_agent(agent, config)
            fitnesses.append((fitness, entropy, agent))

        fitnesses.sort(key = lambda x: x[0], reverse = True)

        raw_fitness_vals = [fitness for fitness, _, _ in fitnesses]
        min_fitness = min(raw_fitness_vals)

        elites = fitnesses[:config['elite_count']]
        elite_agents = [agent for _, _, agent in elites]
        best_fitness = elites[0][0]
        mean_fitness = np.mean(raw_fitness_vals)

        print(f"  best reward: {best_fitness:.2f}")
        print(f"  mean reward: {mean_fitness:.2f}")

        wandb.log({
            "generation": generation,
            "max_reward": best_fitness,
            "mean_reward": mean_fitness
        })

        next_population = []

        for elite_agent in elite_agents:
            clone = PPOAgent(config).to(device)
            clone.set_weights(elite_agent.get_weights())
            next_population.append(clone)

        num_tournament_contenders = max(2, int(config['elite_count'] * config['frac_tournament']))
        num_tournament_contenders = min(config['elite_count'], num_tournament_contenders)

        while len(next_population) < config['population_size']:
            # deterministic tournament selection - let top 2 winners become parents
            contenders = random.sample(elites, num_tournament_contenders)
            contenders.sort(key = lambda x: x[0], reverse = True)
            parent1, parent2 = contenders[:2]

            fitness1, _, agent1 = parent1
            fitness2, _, agent2 = parent2

            weights1 = agent1.get_weights()
            weights2 = agent2.get_weights()

            child_weights, scaled_fitness1, scaled_fitness2 = crossover(
                weights1, weights2, fitness1, fitness2, min_fitness
            )

            child = PPOAgent(config).to(device)

            if random.random() < config['mutation_prob']:
                mutated_weights = mutate(
                    child_weights, fitness1, fitness2, scaled_fitness1, scaled_fitness2
                )
                child.set_weights(mutated_weights)
            else:
                child.set_weights(child_weights)
                # lamarckian fine-tuning, optionally guided by meta-evolved temperature

                meta_temp = None

                if use_meta_evolution:
                    meta_temp = meta_state.current_meta_temp

                train_agent_for_steps(child, config, config['finetune_steps'], meta_temp = meta_temp)

                if use_meta_evolution:
                    child_fitness, child_entropy = evaluate_agent(child, config)

                    alpha = scaled_fitness1 / (scaled_fitness1 + scaled_fitness2 + 1e-5)
                    expected_start_fitness = alpha * fitness1 + (1 - alpha) * fitness2
                    delta_return = child_fitness - expected_start_fitness

                    meta_fitness = delta_return + meta_kappa * child_entropy
                    meta_state.record_fitness(meta_fitness)

            next_population.append(child)

        population = next_population

        if (generation + 1) % config['record_every'] == 0:
            print(f"  recording videos for generation {generation + 1}...")
            video_files_logged = []

            sorted_agents = [agent for _, _, agent in fitnesses]

            for agent_idx, agent in enumerate(sorted_agents):
                eval_env = gym.make(config['env_id'], render_mode = "rgb_array")
                eval_env = gym.wrappers.RecordVideo(
                    env = eval_env,
                    video_folder = config['video_folder'],
                    name_prefix = f"gen-{generation + 1}-agent-{agent_idx}",
                    disable_logger = True
                )

                state, _ = eval_env.reset()
                done = False

                while not done:
                    state_tensor = tensor(state, dtype = torch.float32, device = device)

                    with torch.no_grad():
                        action_probs = agent.policy.actor(state_tensor)
                        action = torch.argmax(action_probs).item()

                    state, _, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated

                eval_env.close()

                videos = glob.glob(os.path.join(config['video_folder'], f"gen-{generation + 1}-agent-{agent_idx}*.mp4"))
                if videos:
                    video_files_logged.append(max(videos, key=os.path.getctime))

            # Stitch and log videos to wandb
            if video_files_logged:
                try:
                    clips = [VideoFileClip(v).resized(0.5) for v in video_files_logged]

                    n = len(clips)
                    cols = int(math.ceil(math.sqrt(n)))
                    rows = int(math.ceil(n / cols))

                    w, h = clips[0].size
                    duration = max([c.duration for c in clips])

                    if rows * cols > n:
                        blank_clip = ColorClip(size=(w, h), color=(0,0,0), duration=duration)
                        clips.extend([blank_clip] * (rows * cols - n))

                    grid = []
                    for r in range(rows):
                        grid.append(clips[r * cols : (r + 1) * cols])

                    final_clip = clips_array(grid)
                    stitched_path = os.path.join(config['video_folder'], f"gen-{generation + 1}-stitched.mp4")
                    final_clip.write_videofile(stitched_path, logger=None)

                    for clip in clips:
                        clip.close()
                    final_clip.close()

                    wandb.log(dict(
                        generation = generation,
                        evaluation_video = wandb.Video(stitched_path, fps=30, format="mp4")
                    ))

                    for v in video_files_logged:
                        try:
                            os.remove(v)
                        except:
                            pass
                except Exception as e:
                    print(f"  failed to stitch videos: {e}")
                    wandb.log(dict(
                        generation = generation,
                        evaluation_videos = [wandb.Video(v, fps=30, format="mp4") for v in video_files_logged]
                    ))

    print("evolution complete.")
    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)
