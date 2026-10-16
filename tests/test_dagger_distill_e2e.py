import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal, OneHotCategorical

from x_ppo.dagger_distill import DaggerDistiller, best_action

param = pytest.mark.parametrize

# end to end - a PD teacher drives a (pos, vel) integrator to the origin, and
# the DAgger loop distills its behaviour into the student, once per action-space
# family: plain / chunked normal, categorical, one-hot, bernoulli, mean-conc
# beta, and a composite (bernoulli, beta) tuple

class ToyGoalEnv:
    # vel' = 0.8 vel + control, pos' = pos + 0.1 vel'

    def __init__(self, num_envs, control_dim, seed = 0):
        self.num_envs, self.control_dim = num_envs, control_dim
        self.generator = torch.Generator().manual_seed(seed)
        self.state = self._sample()

    def _sample(self):
        return torch.randn(self.num_envs, 2 * self.control_dim, generator = self.generator)

    def reset(self):
        self.state = self._sample()
        return self.state.clone(), {}

    def step(self, control):
        control = torch.as_tensor(control, dtype = torch.float32)
        pos, vel = self.state[:, :self.control_dim], self.state[:, self.control_dim:]
        vel = 0.8 * vel + control
        pos = pos + 0.1 * vel
        self.state = torch.cat([pos, vel], -1)

        zeros = torch.zeros(self.num_envs)
        return self.state.clone(), zeros, zeros.bool(), zeros.bool(), {}

def make_net(state_dim, out_dim, dim = 32):
    return nn.Sequential(nn.Linear(state_dim, dim), nn.Tanh(), nn.Linear(dim, out_dim))

class NormalHead(nn.Module):
    def __init__(self, state_dim, action_dim, std = 0.2):
        super().__init__()
        self.net, self.std = make_net(state_dim, action_dim), std

    def forward(self, state):
        mean = self.net(state)
        return Normal(mean, torch.full_like(mean, self.std))

class ChunkNormalHead(nn.Module):
    def __init__(self, state_dim, chunk, action_dim, std = 0.2):
        super().__init__()
        self.chunk, self.action_dim, self.std = chunk, action_dim, std
        self.net = make_net(state_dim, chunk * action_dim)

    def forward(self, state):
        mean = self.net(state).reshape(*state.shape[:-1], self.chunk, self.action_dim)
        return Normal(mean, torch.full_like(mean, self.std))

class CategoricalHead(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.net = make_net(state_dim, num_actions)

    def forward(self, state):
        return Categorical(logits = self.net(state))

class OneHotHead(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.net = make_net(state_dim, num_actions)

    def forward(self, state):
        return OneHotCategorical(probs = torch.softmax(self.net(state), -1))

class BernoulliHead(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = make_net(state_dim, action_dim)

    def forward(self, state):
        return Bernoulli(logits = self.net(state))

def build_policy_family(family, state_dim, control_dim, num_actions = 5, chunk = 2, seed = 0):
    # one PD controller wearing each family's policy head -> (student, teacher, action_to_env_fn)

    torch.manual_seed(seed)

    scale = 2.0

    def desired(state):
        pos, vel = state[..., :control_dim], state[..., control_dim:]
        return -2.0 * pos - vel

    if family == 'normal':
        student = NormalHead(state_dim, control_dim)

        def teacher(state):
            mean = desired(state)
            return Normal(mean, torch.full_like(mean, 0.05))

        return student, teacher, None

    if family == 'normal_chunk':
        student = ChunkNormalHead(state_dim, chunk, control_dim)

        def teacher(state):
            mean = desired(state).unsqueeze(-2).expand(-1, chunk, -1)
            return Normal(mean, torch.full_like(mean, 0.05))

        return student, teacher, lambda action: action[..., 0, :]

    if family in ('categorical', 'onehot'):
        # action 0 coasts, the rest push along fixed directions
        directions = torch.zeros(num_actions, control_dim)
        directions[1:] = F.normalize(torch.randn(num_actions - 1, control_dim), dim = -1)

        def teacher_logits(state):
            goal = desired(state)
            fit = (goal.unsqueeze(-2) - scale * directions).pow(2).sum(-1)
            return -2.0 * fit

        if family == 'categorical':
            student = CategoricalHead(state_dim, num_actions)
            teacher = lambda state: Categorical(logits = teacher_logits(state))
        else:
            student = OneHotHead(state_dim, num_actions)
            teacher = lambda state: OneHotCategorical(probs = torch.softmax(teacher_logits(state), -1))

        return student, teacher, lambda action: scale * directions[action.long()]

    if family == 'bernoulli':
        student = BernoulliHead(state_dim, control_dim)

        def teacher(state):
            return Bernoulli(logits = 4.0 * desired(state))

        return student, teacher, lambda action: 2.0 * action.float() - 1.0

    if family in ('beta', 'hybrid'):
        from mean_conc_beta import Beta

        beta = Beta(init_conc = 20.)

    if family == 'beta':
        class BetaHead(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.net = make_net(state_dim, action_dim * 2)

            def forward(self, state):
                params = self.net(state).reshape(*state.shape[:-1], -1, 2)
                return beta(params, temperature = 1.)

        student = BetaHead(state_dim, control_dim)

        def teacher(state):
            goal = desired(state)
            return beta(torch.stack([goal, torch.zeros_like(goal)], -1), temperature = 1.)

        return student, teacher, lambda action: 3.0 * action

    if family == 'hybrid':
        num_binary, num_continuous = control_dim // 2, control_dim - control_dim // 2

        class HybridHead(nn.Module):
            def __init__(self, state_dim, num_binary, num_continuous):
                super().__init__()
                self.binary_net = make_net(state_dim, num_binary)
                self.arm_net = make_net(state_dim, num_continuous * 2)

            def forward(self, state):
                buttons = Bernoulli(logits = self.binary_net(state))
                params = self.arm_net(state).reshape(*state.shape[:-1], -1, 2)
                return buttons, beta(params, temperature = 1.)

        student = HybridHead(state_dim, num_binary, num_continuous)

        def teacher(state):
            goal = desired(state)
            binary_goal, continuous_goal = goal[..., :num_binary], goal[..., num_binary:]
            buttons = Bernoulli(logits = 4.0 * binary_goal)
            params = torch.stack([continuous_goal, torch.zeros_like(continuous_goal)], -1)
            return buttons, beta(params, temperature = 1.)

        def to_env(action):
            buttons, arm = action
            return torch.cat([2.0 * buttons.float() - 1.0, 3.0 * arm], -1)

        return student, teacher, to_env

    raise ValueError(f'{family=}')

@param('family', ('normal', 'normal_chunk', 'categorical', 'onehot', 'bernoulli', 'beta', 'hybrid'))
def test_e2e_distills_every_action_space(family):
    if family in ('beta', 'hybrid'):
        pytest.importorskip('mean_conc_beta')

    torch.manual_seed(0)
    num_envs, cycles, collect_steps = 16, 18, 4
    control_dim = 4 if family == 'hybrid' else 2
    state_dim = 2 * control_dim

    student, teacher, action_to_env_fn = build_policy_family(family, state_dim, control_dim)
    snapshot = copy.deepcopy(student)
    optimizer = torch.optim.Adam(student.parameters(), lr = 5e-3)

    distiller = DaggerDistiller(
        student, teacher,
        env = ToyGoalEnv(num_envs, control_dim, seed = 1),
        optimizer = optimizer,
        action_to_env_fn = action_to_env_fn,
        batch_size = 128,
        max_samples = 2048
    )

    for _ in range(cycles):
        stats = distiller.step(collect_steps = collect_steps)

    assert distiller.count == cycles * collect_steps * num_envs
    assert math.isfinite(stats['loss'])

    # imitation - mean error for continuous heads, per-dim agreement for discrete

    probe = torch.randn(512, state_dim)

    with torch.no_grad():
        teacher_out, student_out, snapshot_out = teacher(probe), student(probe), snapshot(probe)

    if family in ('normal', 'normal_chunk', 'beta'):
        error = lambda out: (out.mean - teacher_out.mean).pow(2).mean().item()
        assert error(student_out) < 0.1 and error(student_out) < error(snapshot_out) * 0.25, f'{family=} {error(snapshot_out)=:.4f} {error(student_out)=:.4f}'
    elif family == 'hybrid':
        agree = lambda out: ((out[0].probs > 0.5) == (teacher_out[0].probs > 0.5)).float().mean().item()
        error = lambda out: (out[1].mean - teacher_out[1].mean).pow(2).mean().item()
        assert agree(student_out) > 0.85 and agree(student_out) > agree(snapshot_out), f'{family=} {agree(snapshot_out)=:.2f} {agree(student_out)=:.2f}'
        assert error(student_out) < 0.05 and error(student_out) < error(snapshot_out) * 0.25, f'{family=} {error(snapshot_out)=:.4f} {error(student_out)=:.4f}'
    else:
        if family == 'bernoulli':
            agree = lambda out: ((out.probs > 0.5) == (teacher_out.probs > 0.5)).float().mean().item()
        else:
            agree = lambda out: (out.probs.argmax(-1) == teacher_out.probs.argmax(-1)).float().mean().item()

        assert agree(student_out) > 0.85 and agree(student_out) > agree(snapshot_out), f'{family=} {agree(snapshot_out):.2f} {agree(student_out):.2f}'

    # behaviour - the distilled student reaches the goal, the snapshot wanders

    def goal_distance(policy_fn, seed = 123, steps = 40):
        env = ToyGoalEnv(num_envs, control_dim, seed = seed)
        obs, _ = env.reset()
        total = 0.

        with torch.no_grad():
            for _ in range(steps):
                action = best_action(policy_fn(obs))

                if action_to_env_fn is not None:
                    action = action_to_env_fn(action)

                obs, *_ = env.step(action)
                total += obs[:, :control_dim].pow(2).sum(-1)

        return (total / steps).mean().item()

    student_goal = goal_distance(student)

    assert student_goal < goal_distance(snapshot) * 0.5, f'{family=} {student_goal=:.3f}'
    assert student_goal < goal_distance(teacher) * 2.0 + 0.05, f'{family=} {student_goal=:.3f}'
