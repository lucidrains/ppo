import copy
import math
import runpy

import pytest
import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.distributions import Bernoulli, Categorical, Distribution, Normal, OneHotCategorical

from x_ppo.dagger_distill import (
    DaggerDistiller,
    DISCRETE_DISTRIBUTIONS,
    best_action,
    is_discrete,
    kl_loss,
    log_prob_sum,
    mse_loss,
    nll_loss,
    policy_kind,
    to_log_prob_target
)

# helpers

class ToyVecEnv:
    # a minimal standardized vector env (autoreset semantics)

    def __init__(self, num_envs, state_dim, action_dim, step_fn, episode_len = None, seed = 0):
        self.num_envs, self.state_dim, self.action_dim = num_envs, state_dim, action_dim
        self.step_fn = step_fn
        self.episode_len = episode_len
        self.generator = torch.Generator().manual_seed(seed)
        self.state = torch.randn(num_envs, state_dim, generator = self.generator) * 0.5
        self.t = 0

    def reset(self):
        self.state = torch.randn(self.num_envs, self.state_dim, generator = self.generator) * 0.5
        self.t = 0
        return self.state.clone(), {}

    def step(self, actions):
        self.state = self.step_fn(self.state, actions)
        self.t += 1
        reward = torch.zeros(self.num_envs)
        term = torch.zeros(self.num_envs, dtype = torch.bool)
        trunc = torch.zeros(self.num_envs, dtype = torch.bool)

        if self.episode_len is not None:
            done = self.t >= self.episode_len
            trunc = torch.full((self.num_envs,), done, dtype = torch.bool)

            if done:
                self.state = torch.randn(self.num_envs, self.state_dim, generator = self.generator) * 0.5
                self.t = 0

        return self.state.clone(), reward, term, trunc, {}

class NormalPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, dim = 32, std = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, dim), nn.Tanh(), nn.Linear(dim, action_dim))
        self.std = std

    def forward(self, state):
        return Normal(self.net(state), torch.full_like(self.net(state), self.std))

class CategoricalPolicy(nn.Module):
    def __init__(self, state_dim, num_actions, dim = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, dim), nn.Tanh(), nn.Linear(dim, num_actions))

    def forward(self, state):
        return Categorical(logits = self.net(state))

def normal_teacher(action_dim, std = 1.):
    return lambda state: Normal(torch.zeros(*state.shape[:-1], action_dim), std)

# 1. action space helpers - labels, modes and sampling, per family

def test_action_spaces():
    # discrete families carry class labels, multi-dim bernoulli one 0/1 per dim

    onehot_dist = OneHotCategorical(probs = torch.softmax(torch.randn(16, 4), -1))
    onehot_target = to_log_prob_target(torch.randint(0, 4, (16,)), onehot_dist)

    assert onehot_target.shape == (16, 4)
    assert torch.isfinite(log_prob_sum(onehot_dist, onehot_target)).all()

    bernoulli_dist = Bernoulli(probs = torch.rand(16, 3))
    bern_target = to_log_prob_target(torch.randint(0, 2, (16, 3)).float(), bernoulli_dist)

    assert torch.isfinite(log_prob_sum(bernoulli_dist, bern_target)).all()

    assert is_discrete(onehot_dist) and is_discrete(bernoulli_dist) and is_discrete(Categorical(logits = torch.randn(4, 2)))
    assert not is_discrete(Normal(torch.zeros(4, 2), torch.ones(4, 2)))

    # bernoulli's mode is per-dimension, not an argmax over dims

    assert best_action(Bernoulli(probs = tensor([[0.9, 0.1, 0.7]]))).tolist() == [[1., 0., 1.]]

    # the mode is the class index, and a sampled one-hot draw collapses to it

    assert best_action(onehot_dist).shape == (16,)
    assert best_action(onehot_dist, sample = True).shape == (16,)
    assert best_action(onehot_dist, sample = True).dtype == torch.long

    # labels stored as (b, 1) land right - no silent (b, b) broadcast

    categorical = Categorical(logits = torch.randn(16, 4))
    labels = torch.randint(0, 4, (16,))

    assert torch.allclose(log_prob_sum(categorical, to_log_prob_target(labels[:, None], categorical)), categorical.log_prob(labels))

    # composite policies are pytrees of heads - discrete / continuous / mixed

    composite = (Bernoulli(probs = torch.rand(4, 2)), Normal(torch.zeros(4, 2), torch.ones(4, 2)))

    assert policy_kind(composite) == 'mixed' and policy_kind(composite[0]) == 'discrete' and policy_kind(composite[1]) == 'continuous'

    button_actions, arm_actions = best_action(composite)

    assert button_actions.shape == (4, 2) and arm_actions.shape == (4, 2)

    # student / teacher contradictions fail early

    mismatch = DaggerDistiller(CategoricalPolicy(2, 3), NormalPolicy(2, 1), action_kind = 'continuous')

    with pytest.raises(ValueError):
        mismatch(torch.randn(4, 2))

    # custom discrete families - append to DISCRETE_DISTRIBUTIONS

    class TinyCategorical(Distribution):
        arg_constraints = {}

        def __init__(self, logits):
            super().__init__(batch_shape = logits.shape[:-1])
            self.logits = logits

        def sample(self, sample_shape = torch.Size()):
            return Categorical(logits = self.logits).sample(sample_shape)

        def log_prob(self, value):
            return Categorical(logits = self.logits).log_prob(value)

    DISCRETE_DISTRIBUTIONS.append(TinyCategorical)
    tiny = TinyCategorical(torch.randn(4, 3))

    assert is_discrete(tiny) and best_action(tiny).shape == (4,)
    assert to_log_prob_target(torch.randint(0, 3, (4, 1)), tiny).shape == (4,)

    DISCRETE_DISTRIBUTIONS.remove(TinyCategorical)

# 2. env-free wrapper - student + teacher in, one distill loss out

def test_env_free():
    # the teacher never builds a graph

    torch.manual_seed(0)
    state_dim, action_dim = 4, 3

    teacher_w = torch.randn(state_dim, action_dim) * 0.5
    teacher_grad_modes = []

    def teacher(state):
        teacher_grad_modes.append(torch.is_grad_enabled())
        return Normal(state @ teacher_w, torch.full_like(state @ teacher_w, 0.1))

    student = NormalPolicy(state_dim, action_dim)
    snapshot = copy.deepcopy(student)
    optimizer = torch.optim.Adam(student.parameters(), lr = 5e-3)

    distiller = DaggerDistiller(student, teacher)

    for _ in range(40):
        states = torch.randn(64, state_dim)
        loss, out = distiller(states)

        assert loss.dim() == 0 and torch.isfinite(loss) and loss.requires_grad
        assert isinstance(out, Normal) and out.mean.shape == (64, action_dim)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probe = torch.randn(256, state_dim)
        start_err = (snapshot(probe).mean - teacher(probe).mean).pow(2).mean().item()
        end_err = (student(probe).mean - teacher(probe).mean).pow(2).mean().item()

    assert not any(teacher_grad_modes) and end_err < start_err * 0.2, f'{start_err=:.4f} {end_err=:.4f}'

    # hooks - the teacher's own state view, a label transform, info / step both

    wide_w = torch.randn(state_dim + 1, action_dim) * 0.5
    wide_teacher = lambda state: Normal(state @ wide_w, torch.full_like(state @ wide_w, 0.1))

    derive_steps, transform_steps = [], []

    def teacher_state_fn(state, info, step):
        derive_steps.append(step)
        return torch.cat([state, torch.full_like(state[:, :1], info['theta'])], -1)

    def label_transform(state, action, info, step):
        transform_steps.append((step, info.get('theta')))
        return action * 0.5

    hooked = DaggerDistiller(student, wide_teacher, teacher_state_fn = teacher_state_fn, label_transform = label_transform)

    states = torch.randn(8, state_dim)
    loss, out = hooked(states, info = dict(theta = 1.5), step = 7)
    loss.backward()

    assert derive_steps == [7] and transform_steps == [(7, 1.5)] and out.mean.shape == (8, action_dim)

    # explicit teacher data skips teacher_state_fn

    hooked(states, torch.randn(8, state_dim + 1), info = dict(theta = 1.5), step = 8)

    assert derive_steps == [7] and transform_steps[-1] == (8, 1.5)

    # research seam - a custom loss_fn sees both distributions, per-sample

    seen = []

    def mean_mse(student_distr, teacher_distr, label, info):
        seen.append((teacher_distr is not None, label.shape, info))
        return (student_distr.mean - teacher_distr.mean.detach()).pow(2).mean(dim = 1)

    custom = DaggerDistiller(student, teacher, loss_fn = mean_mse)
    loss, _ = custom(states, info = dict(tag = 'env-free'))

    assert seen == [(True, (8, action_dim), dict(tag = 'env-free'))]

    with torch.no_grad():
        manual = (student(states).mean - teacher(states).mean).pow(2).mean()

    assert torch.allclose(loss, manual, atol = 1e-6)

    # built-in objectives - nll is the default, kl matches dists, mse matches means

    states = torch.randn(16, state_dim)

    default_loss, _ = DaggerDistiller(student, teacher)(states)
    nll_value, _ = DaggerDistiller(student, teacher, loss_fn = nll_loss)(states)
    mse_value, _ = DaggerDistiller(student, teacher, loss_fn = mse_loss)(states)
    kl_value, _ = DaggerDistiller(student, teacher, loss_fn = kl_loss)(states)

    with torch.no_grad():
        teacher_out, student_out = teacher(states), student(states)
        manual_mse = (student_out.mean - teacher_out.mean).pow(2).mean()
        manual_kl = torch.distributions.kl_divergence(teacher_out, student_out).sum(-1).mean()

    assert torch.allclose(default_loss, nll_value, atol = 1e-6)
    assert torch.allclose(mse_value, manual_mse, atol = 1e-6)
    assert torch.allclose(kl_value, manual_kl, atol = 1e-6)

    # loss breakdown - the total by default, the components on demand

    breakdown, _ = DaggerDistiller(student, teacher, entropy_weight = 0.1)(states, return_loss_breakdown = True)

    assert torch.allclose(breakdown.loss, breakdown.policy - 0.1 * breakdown.entropy)
    assert breakdown.entropy.dim() == 0 and breakdown.alignment is None

    plain_breakdown, _ = DaggerDistiller(student, teacher)(states, return_loss_breakdown = True)

    assert plain_breakdown.entropy is None and torch.allclose(plain_breakdown.loss, plain_breakdown.policy)

    # representation alignment - hook a layer of each, 1 - cosine by default

    class TinyNet(nn.Module):
        def __init__(self, state_dim, hidden = 8):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh())
            self.head = nn.Linear(hidden, 3)

        def forward(self, state):
            return Normal(self.head(self.trunk(state)), 0.2)

    torch.manual_seed(0)
    student_net, teacher_net = TinyNet(state_dim), TinyNet(state_dim)

    aligned_breakdown, _ = DaggerDistiller(student_net, teacher_net, student_layer = 'trunk.1', teacher_layer = 'trunk.1', rep_align_weight = 0.25)(states, return_loss_breakdown = True)

    with torch.no_grad():
        manual_align = (1. - F.cosine_similarity(student_net.trunk(states), teacher_net.trunk(states), dim = -1)).mean()

    assert torch.allclose(aligned_breakdown.alignment, manual_align, atol = 1e-6)
    assert torch.allclose(aligned_breakdown.loss, aligned_breakdown.policy + 0.25 * manual_align, atol = 1e-6)

    # a projector bridges mismatched widths, and its grads flow

    wide_net = TinyNet(state_dim, hidden = 16)
    projector = nn.Linear(8, 16)

    projected_breakdown, _ = DaggerDistiller(student_net, wide_net, student_layer = 'trunk.1', teacher_layer = 'trunk.1', rep_align_projector = projector)(states, return_loss_breakdown = True)

    projected_breakdown.loss.backward()

    assert torch.isfinite(projected_breakdown.alignment) and projector.weight.grad is not None

    # layer names resolve at init, and both sides are required

    with pytest.raises(ValueError):
        DaggerDistiller(student_net, teacher_net, student_layer = 'nope', teacher_layer = 'trunk.1')

    with pytest.raises(AssertionError):
        DaggerDistiller(student_net, teacher_net, student_layer = 'trunk.1')

    # composite action spaces - a pytree of heads, the loss sums over them

    button_w, arm_w = torch.randn(state_dim, 2) * 0.5, torch.randn(state_dim, 2) * 0.5

    def composite_teacher(state):
        return Bernoulli(logits = state @ button_w), Normal(state @ arm_w, 0.2)

    class CompositeStudent(nn.Module):
        def __init__(self, state_dim, dim = 32):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(state_dim, dim), nn.Tanh(), nn.Linear(dim, 4))

        def forward(self, state):
            out = self.net(state)
            return Bernoulli(logits = out[..., :2]), Normal(out[..., 2:], 0.2)

    composite_student = CompositeStudent(state_dim)
    composite_optim = torch.optim.Adam(composite_student.parameters(), lr = 1e-2)
    composite = DaggerDistiller(composite_student, composite_teacher)

    actions = best_action(composite(states)[1])

    assert isinstance(actions, tuple) and actions[0].shape == (16, 2) and actions[1].shape == (16, 2)

    for _ in range(60):
        loss, _ = composite(torch.randn(32, state_dim))
        composite_optim.zero_grad()
        loss.backward()
        composite_optim.step()

    with torch.no_grad():
        probe = torch.randn(256, state_dim)
        t_buttons, t_arm = composite_teacher(probe)
        s_buttons, s_arm = composite_student(probe)

    assert ((s_buttons.probs > 0.5) == (t_buttons.probs > 0.5)).float().mean() > 0.85
    assert (s_arm.mean - t_arm.mean).pow(2).mean() < 0.05

    # soft kl sums over the heads too

    composite_kl, _ = DaggerDistiller(composite_student, composite_teacher, loss_fn = kl_loss)(states)

    with torch.no_grad():
        s_buttons, s_arm = composite_student(states)
        t_buttons, t_arm = composite_teacher(states)
        manual_kl = (torch.distributions.kl_divergence(t_buttons, s_buttons).sum(-1)
                     + torch.distributions.kl_divergence(t_arm, s_arm).sum(-1)).mean()

    assert torch.allclose(composite_kl, manual_kl, atol = 1e-6)

    # non-finite labels parked on the student's mean - a fully unusable batch is nan

    def bad_transform(state, action, info, step):
        action = action.clone()
        action[::2] = float('nan')
        return action

    loss, _ = DaggerDistiller(student, teacher, label_transform = bad_transform)(states)

    assert torch.isfinite(loss)

    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())

    all_nan = DaggerDistiller(student, teacher, label_transform = lambda state, action, info, step: action * float('nan'))

    assert torch.isnan(all_nan(states)[0])

    # discrete heads ride the same wrapper

    discrete_w = torch.randn(4, 3)
    discrete_teacher = lambda state: Categorical(logits = state @ discrete_w)
    discrete_student = CategoricalPolicy(4, 3)
    discrete_optim = torch.optim.Adam(discrete_student.parameters(), lr = 1e-2)
    discrete = DaggerDistiller(discrete_student, discrete_teacher)

    assert discrete.kind is None

    for _ in range(100):
        loss, out = discrete(torch.randn(32, 4))
        discrete_optim.zero_grad()
        loss.backward()
        discrete_optim.step()

    assert discrete.kind == 'discrete' and out.probs.shape == (32, 3)

    with torch.no_grad():
        probe = torch.randn(256, 4)
        agree = (discrete_student(probe).probs.argmax(-1) == discrete_teacher(probe).probs.argmax(-1)).float().mean().item()

    assert agree > 0.85, f'{agree=:.2f}'

# 3. the DAgger loop - roll the student out, label, fit

def test_env_loop():
    # episode bookkeeping - 3-step episodes cycle per-env counters 0, 1, 2, 0, ...

    seen = []

    def step_fn(state, actions):
        return 0.9 * state + 0.1 * actions

    env = ToyVecEnv(4, 2, 1, step_fn, episode_len = 3)
    student = NormalPolicy(2, 1)
    optimizer = torch.optim.Adam(student.parameters(), lr = 1e-3)

    def teacher_action_fn(teacher_state, info, step):
        seen.append(tuple(info['episode_steps'].tolist()))
        return torch.zeros(teacher_state.shape[0], 1)

    distiller = DaggerDistiller(student, teacher_action_fn = teacher_action_fn, env = env, optimizer = optimizer)
    distiller.collect(6)

    assert seen[0] == (0, 0, 0, 0) and seen[3] == (0, 0, 0, 0)
    assert distiller.count == 6 * 4

    # ring buffer wraps, subsampling is respected

    ring = DaggerDistiller(student, normal_teacher(1), env = ToyVecEnv(4, 2, 1, lambda s, a: s), optimizer = optimizer, buffer_size = 10, max_samples = 6)

    for _ in range(5):
        ring.collect(1)

    assert ring.count == 20 and ring.train(epochs = 1)['samples'] == 6

    # deterministic collect uses modes, custom env step + action conversion hooks

    calls = []
    hooked_env = ToyVecEnv(4, 2, 1, step_fn)

    def env_step_fn(actions):
        calls.append(torch.as_tensor(actions).dtype)
        return hooked_env.step(actions)

    hooked = DaggerDistiller(student, normal_teacher(1), env = hooked_env, optimizer = optimizer, env_step_fn = env_step_fn)

    assert hooked.collect(3, sample = False)['labeled'] == 12 and len(calls) == 3

    # representation alignment rides the loop - teacher reps buffer up at collect

    teacher_policy = NormalPolicy(2, 1)
    aligned = DaggerDistiller(student, teacher_policy, env = ToyVecEnv(4, 2, 1, step_fn), optimizer = optimizer, student_layer = 'net.2', teacher_layer = 'net.2')

    stats = aligned.step(collect_steps = 4, epochs = 1, return_loss_breakdown = True)

    assert aligned.rep_buffer.shape == (aligned.buffer_size, 1)
    assert distiller.rep_buffer is None
    assert math.isfinite(stats['loss']) and torch.isfinite(stats['loss_breakdown'].alignment)

    # the teacher's own state view inside the loop (a clock, here)

    phase_steps = []

    def teacher_state_fn(state, info, step):
        phase_steps.append(step)
        clock = torch.full_like(state[:, :1], 0.05 * info['episode_steps'][0])
        return torch.cat([state, clock], -1)

    clocked = DaggerDistiller(student, lambda teacher_state: Normal(0.1 * teacher_state[..., :1], 1.), env = ToyVecEnv(4, 2, 1, step_fn), optimizer = optimizer, teacher_state_fn = teacher_state_fn)
    clocked.collect(3)

    assert phase_steps == [0, 1, 2]

    # a dict observation (numpy, as a vec env hands back) keeps its fields -
    # the student reads its subset, the teacher its own

    class DictObsEnv(ToyVecEnv):
        def _wrap(self, obs):
            return dict(state = obs.numpy(), privileged = 2. * obs.numpy())

        def reset(self):
            obs, info = super().reset()
            return self._wrap(obs), info

        def step(self, actions):
            obs, reward, term, trunc, info = super().step(actions)
            return self._wrap(obs), reward, term, trunc, info

    views, teacher_inputs = [], []

    def dict_student_state_fn(obs, info, step):
        views.append(sorted(obs))
        return obs['state']

    def dict_teacher_obs_fn(obs, info, step):
        views.append(sorted(obs))
        return obs['privileged']

    def wide_teacher(teacher_state):
        teacher_inputs.append(teacher_state)
        return Normal(0.1 * teacher_state[..., :1], 1.)

    dict_obs = DaggerDistiller(student, wide_teacher, env = DictObsEnv(4, 2, 1, lambda s, a: s), optimizer = optimizer, student_state_fn = dict_student_state_fn, teacher_obs_fn = dict_teacher_obs_fn)
    dict_obs.collect(2)

    assert all(keys == ['privileged', 'state'] for keys in views)
    assert torch.allclose(torch.cat(teacher_inputs), 2. * dict_obs.state_buffers[0][:8])

    # teacher_state_fn still refines the student's state, as it did before

    raws, chained = [], []

    def scaling_student_state_fn(state, info, step):
        raws.append(state)
        return state * 3.

    def chaining_teacher_state_fn(state, info, step):
        chained.append(state)
        return state

    chain_distiller = DaggerDistiller(student, normal_teacher(1), env = ToyVecEnv(4, 2, 1, lambda s, a: s), optimizer = optimizer, student_state_fn = scaling_student_state_fn, teacher_state_fn = chaining_teacher_state_fn)
    chain_distiller.collect(1)

    assert torch.allclose(torch.cat(chained), 3. * torch.cat(raws))
    assert torch.allclose(chain_distiller.state_buffers[0][:4], 3. * torch.cat(raws))

    # a multi-key subset is a state too - the buffer carries the pytree

    class DictStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4, 1)

        def forward(self, state):
            return Normal(self.net(torch.cat([state['state'], state['privileged']], -1)), 1.)

    subset_student = DictStudent()
    subset = DaggerDistiller(subset_student, normal_teacher(1), env = DictObsEnv(4, 2, 1, lambda s, a: s), optimizer = torch.optim.Adam(subset_student.parameters()), student_state_fn = lambda obs, info, step: dict(state = obs['state'], privileged = obs['privileged']), teacher_obs_fn = dict_teacher_obs_fn)
    subset_stats = subset.step(collect_steps = 2, epochs = 1)

    assert math.isfinite(subset_stats['loss']) and len(subset.state_buffers) == 2

    # non-finite teacher labels never poison the student

    def bad_label_transform(state, action, info, step):
        action = action.clone()
        action[::2] = float('nan')
        return action

    nan_distiller = DaggerDistiller(student, normal_teacher(1), env = ToyVecEnv(8, 2, 1, lambda s, a: s), optimizer = optimizer, label_transform = bad_label_transform, batch_size = 64)

    for _ in range(3):
        nan_distiller.step(collect_steps = 4)

    assert all(torch.isfinite(p).all() for p in student.parameters())

    # custom objectives in the loop fit from the label-only buffer

    label_only = []

    def label_mse(student_distr, teacher_distr, label, info):
        label_only.append(teacher_distr is None and info is None)
        return (student_distr.mean - label).pow(2).mean(dim = 1)

    for loss_fn in (label_mse, kl_loss):
        loop = DaggerDistiller(student, normal_teacher(1), env = ToyVecEnv(4, 2, 1, step_fn), optimizer = optimizer, loss_fn = loss_fn)
        assert math.isfinite(loop.step(collect_steps = 4, epochs = 1)['loss'])

    assert all(label_only)

    # requirements fail early and loudly

    with pytest.raises(RuntimeError, match = 'no env'):
        DaggerDistiller(student, normal_teacher(1)).collect(1)

    with pytest.raises(AssertionError, match = 'optimizer'):
        DaggerDistiller(student, normal_teacher(1), env = env)

    # the __main__ demo runs the whole ladder

    runpy.run_module('x_ppo.dagger_distill', run_name = '__main__')
