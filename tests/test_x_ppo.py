import pytest
import torch
from torch_einops_utils import lens_to_mask

from x_ppo import (
    ppo_actor_loss,
    spo_actor_loss,
    calc_gae,
    calc_returns,
    combine_grouped_advantages,
    dones_to_masks,
    GAE,
    clipped_value_loss_fn,
    improved_value_clipping_fn
)

param = pytest.mark.parametrize

@param('normalize_advantages', (False, True))
@param('delightful', (False, True))
@param('use_mask', (False, True))
def test_actor_loss(
    normalize_advantages,
    delightful,
    use_mask
):
    batch, seq = 2, 4

    action_log_probs = torch.randn(batch, seq, requires_grad = True)
    old_action_log_probs = torch.randn(batch, seq)
    advantages = torch.randn(batch, seq)

    mask = lens_to_mask(torch.tensor([2, 4]), seq) if use_mask else None

    loss = ppo_actor_loss(
        action_log_probs,
        old_action_log_probs,
        advantages,
        mask = mask,
        normalize_advantages = normalize_advantages,
        delightful = delightful
    )

    loss.sum().backward()

    assert action_log_probs.grad is not None

@param('asymmetric', (False, True))
@param('normalize_advantages', (False, True))
@param('delightful', (False, True))
@param('use_mask', (False, True))
def test_spo_actor_loss(
    asymmetric,
    normalize_advantages,
    delightful,
    use_mask
):
    batch, seq = 2, 4

    action_log_probs = torch.randn(batch, seq, requires_grad = True)
    old_action_log_probs = torch.randn(batch, seq)
    advantages = torch.randn(batch, seq)

    mask = lens_to_mask(torch.tensor([2, 4]), seq) if use_mask else None

    loss = spo_actor_loss(
        action_log_probs,
        old_action_log_probs,
        advantages,
        mask = mask,
        normalize_advantages = normalize_advantages,
        delightful = delightful,
        asymmetric = asymmetric
    )

    loss.sum().backward()

    assert action_log_probs.grad is not None

# GAE vs Sequential Ground Truth

def sequential_gae(rewards, values, masks, gamma = 0.99, lam = 0.95, next_val = 0.):
    b, t = rewards.shape
    advantages = torch.zeros(b, t)
    last_gae = torch.zeros(b)

    for step in reversed(range(t)):
        v_next = next_val if step == t - 1 else values[:, step + 1]
        delta = rewards[:, step] + gamma * v_next * masks[:, step] - values[:, step]
        last_gae = delta + gamma * lam * masks[:, step] * last_gae
        advantages[:, step] = last_gae

    returns = advantages + values
    return returns, advantages

def test_gae_matches_sequential():
    b, t = 3, 10
    rewards = torch.randn(b, t)
    values = torch.randn(b, t)
    masks = torch.tensor([
        [1., 1., 1., 0., 1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 1., 0., 1., 1., 0.]
    ])
    gamma, lam = 0.99, 0.95

    expected_returns, expected_advantages = sequential_gae(rewards, values, masks, gamma, lam)

    out = calc_gae(rewards, values, masks, gamma = gamma, lam = lam, use_accelerated = False, return_advantages = True)

    assert torch.allclose(out.returns, expected_returns, atol = 1e-5)
    assert torch.allclose(out.advantages, expected_advantages, atol = 1e-5)

# Shapes: 1D, 2D, 3D

@param('shape', [(8,), (2, 8), (2, 3, 8)])
def test_gae_shapes(shape):
    rewards = torch.randn(*shape)
    values = torch.randn(*shape)
    masks = torch.ones(*shape)

    returns = calc_gae(rewards, values, masks, use_accelerated = False)
    assert returns.shape == rewards.shape

    out = calc_gae(rewards, values, masks, use_accelerated = False, return_advantages = True)
    assert out.returns.shape == rewards.shape
    assert out.advantages.shape == rewards.shape

# Bootstrapping modes

def test_gae_t_plus_one_values():
    b, t = 2, 6
    rewards = torch.randn(b, t)
    values = torch.randn(b, t + 1)
    masks = torch.ones(b, t)

    returns, advantages = calc_gae(rewards, values, masks, use_accelerated = False, return_advantages = True)

    expected_returns, expected_advantages = calc_gae(
        rewards, values[:, :-1], masks, next_value = values[:, -1], use_accelerated = False, return_advantages = True
    )

    assert torch.allclose(returns, expected_returns, atol = 1e-5)
    assert torch.allclose(advantages, expected_advantages, atol = 1e-5)

def test_gae_scalar_and_tensor_next_value():
    b, t = 2, 5
    rewards = torch.randn(b, t)
    values = torch.randn(b, t)
    masks = torch.ones(b, t)

    ret_scalar = calc_gae(rewards, values, masks, next_value = 2.5, use_accelerated = False)

    next_val_tensor = torch.full((b,), 2.5)
    ret_tensor = calc_gae(rewards, values, masks, next_value = next_val_tensor, use_accelerated = False)

    assert torch.allclose(ret_scalar, ret_tensor, atol = 1e-5)

def test_gae_explicit_next_values():
    b, t = 2, 5
    rewards = torch.randn(b, t)
    values = torch.randn(b, t)
    next_values = torch.randn(b, t)
    masks = torch.ones(b, t)

    returns, advantages = calc_gae(
        rewards, values, masks, next_values = next_values, use_accelerated = False, return_advantages = True
    )

    delta = rewards + 0.99 * next_values * masks - values
    assert torch.allclose(advantages[:, -1], delta[:, -1], atol = 1e-5)

# Dones vs Masks

def test_gae_dones():
    b, t = 2, 6
    rewards = torch.randn(b, t)
    values = torch.randn(b, t)
    dones = torch.tensor([[False, False, True, False, False, True], [False, True, False, False, False, False]])

    ret_dones = calc_gae(rewards, values, masks = dones_to_masks(dones), use_accelerated = False)
    ret_masks = calc_gae(rewards, values, masks = (~dones).float(), use_accelerated = False)

    assert torch.allclose(ret_dones, ret_masks, atol = 1e-5)

# Gymnasium Terminated vs Truncated Semantics

def test_gae_terminated_vs_truncated():
    b, t = 1, 4
    rewards = torch.zeros(b, t)
    values = torch.ones(b, t)
    next_values = torch.full((b, t), 10.0)

    # Step 1 is terminated
    terminated = torch.tensor([[False, True, False, False]])
    truncated = torch.tensor([[False, False, False, False]])

    term_mask, done_mask = dones_to_masks(terminated, truncated)
    out_term = calc_gae(
        rewards, values, masks = term_mask, done_masks = done_mask,
        next_values = next_values, gamma = 1.0, lam = 1.0, use_accelerated = False, return_advantages = True
    )
    assert torch.allclose(out_term.advantages[0, 1], torch.tensor(-1.0), atol = 1e-5)

    # Step 1 is truncated
    terminated_f = torch.tensor([[False, False, False, False]])
    truncated_t = torch.tensor([[False, True, False, False]])

    term_mask_t, done_mask_t = dones_to_masks(terminated_f, truncated_t)
    out_trunc = calc_gae(
        rewards, values, masks = term_mask_t, done_masks = done_mask_t,
        next_values = next_values, gamma = 1.0, lam = 1.0, use_accelerated = False, return_advantages = True
    )
    assert torch.allclose(out_trunc.advantages[0, 1], torch.tensor(9.0), atol = 1e-5)

# Grouped Advantages (Host-pytorch use case)

def test_grouped_advantages():
    # group_dim = 1
    b, g, n = 2, 3, 8
    rewards = torch.randn(b, g, n)
    values = torch.randn(b, g, n)
    masks = torch.ones(b, n)
    masks[:, -1] = 0.

    returns, advantages = calc_gae(
        rewards, values, masks = masks, use_accelerated = False, return_advantages = True
    )
    assert returns.shape == (b, g, n)
    assert advantages.shape == (b, g, n)

    weights = torch.tensor([0.5, 0.3, 0.2])
    combined = combine_grouped_advantages(
        advantages, weights = weights, normalize = True, group_dim = 1
    )
    assert combined.shape == (b, n)

    out_with_weights = calc_gae(
        rewards, values, masks = masks, weights = weights, group_dim = 1, use_accelerated = False, return_advantages = True
    )
    assert out_with_weights.advantages.shape == (b, n)

    # group_dim = 0
    rewards_g0 = torch.randn(g, b, n)
    values_g0 = torch.randn(g, b, n)
    masks_g0 = torch.ones(b, n)

    combined_g0 = combine_grouped_advantages(
        rewards_g0, weights = weights, normalize = True, group_dim = 0
    )
    assert combined_g0.shape == (b, n)

    out_g0 = calc_gae(
        rewards_g0, values_g0, masks = masks_g0, weights = weights, group_dim = 0, use_accelerated = False, return_advantages = True
    )
    assert out_g0.advantages.shape == (b, n)

# Variable lengths

def test_variable_lengths():
    lens = torch.tensor([3, 5])
    rewards = torch.randn(2, 5)
    values = torch.randn(2, 5)

    out = calc_gae(rewards, values, lens = lens, use_accelerated = False, return_advantages = True)

    assert torch.all(out.returns[0, 3:] == 0.)
    assert torch.all(out.advantages[0, 3:] == 0.)

    expected_out = calc_gae(rewards[:1, :3], values[:1, :3], use_accelerated = False, return_advantages = True)
    assert torch.allclose(out.returns[:1, :3], expected_out.returns, atol = 1e-5)
    assert torch.allclose(out.advantages[:1, :3], expected_out.advantages, atol = 1e-5)

# Bare returns

def test_calc_returns():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    gamma = 0.9

    expected = torch.tensor([[5.23, 4.7, 3.0]])

    returns = calc_returns(rewards, gamma = gamma, use_accelerated = False)
    assert torch.allclose(returns, expected, atol = 1e-5)

    returns_gae = calc_gae(rewards, values = torch.zeros_like(rewards), gamma = gamma, lam = 1.0, use_accelerated = False)
    assert torch.allclose(returns_gae, expected, atol = 1e-5)

def test_calc_returns_with_next_value():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    gamma = 0.9
    next_value = 10.0

    expected = torch.tensor([[12.52, 12.8, 12.0]])

    returns = calc_returns(rewards, gamma = gamma, next_value = next_value, use_accelerated = False)
    assert torch.allclose(returns, expected, atol = 1e-5)

def test_calc_returns_dones_and_lens():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    dones = torch.tensor([[False, True, False, False]])
    gamma = 0.9

    expected = torch.tensor([[2.8, 2.0, 6.6, 4.0]])

    returns = calc_returns(rewards, masks = dones_to_masks(dones), gamma = gamma, use_accelerated = False)
    assert torch.allclose(returns, expected, atol = 1e-5)

    lens = torch.tensor([2])
    returns_lens = calc_returns(rewards[:, :2], gamma = gamma, use_accelerated = False)
    returns_padded = calc_returns(rewards, lens = lens, gamma = gamma, use_accelerated = False)
    assert torch.allclose(returns_padded[:, :2], returns_lens, atol = 1e-5)
    assert torch.all(returns_padded[:, 2:] == 0.)

def test_grouped_advantages_b_n_g():
    # shape (b, n, g) with group_dim = -1
    b, n, g = 2, 8, 3
    rewards = torch.randn(b, n, g)
    values = torch.randn(b, n, g)
    masks = torch.ones(b, n)
    weights = torch.tensor([0.5, 0.3, 0.2])

    out = calc_gae(
        rewards, values, masks = masks, weights = weights, group_dim = -1, use_accelerated = False, return_advantages = True
    )
    assert out.advantages.shape == (b, n)
    assert out.returns.shape == (b, n, g)

    # without weights, both stay (b, n, g)
    out_no_weights = calc_gae(
        rewards, values, masks = masks, group_dim = -1, use_accelerated = False, return_advantages = True
    )
    assert out_no_weights.advantages.shape == (b, n, g)
    assert out_no_weights.returns.shape == (b, n, g)

# dones / truncated edge cases

def test_dones_to_masks_truncated_only():
    truncated = torch.tensor([[False, True, False]])

    term_mask, done_mask = dones_to_masks(truncated = truncated)

    assert torch.all(term_mask == 1.)
    assert torch.equal(done_mask.bool(), ~truncated)

def test_dones_to_masks_requires_an_input():
    with pytest.raises(ValueError):
        dones_to_masks()

# shared value head across groups

@param('group_dim', (-2, 0))
def test_gae_shared_values_across_groups(group_dim):
    b, g, n = 2, 3, 5
    rewards = torch.randn(b, g, n) if group_dim == -2 else torch.randn(g, b, n)
    values = torch.randn(b, n)
    masks = torch.ones(b, n)
    masks[:, -1] = 0.

    shared = calc_gae(rewards, values, masks = masks, group_dim = group_dim, use_accelerated = False, return_advantages = True)

    expand = (values.unsqueeze(1) if group_dim == -2 else values.unsqueeze(0)).expand_as(rewards)
    per_group = calc_gae(rewards, expand, masks = masks, group_dim = group_dim, use_accelerated = False, return_advantages = True)

    assert torch.allclose(shared.returns, per_group.returns, atol = 1e-5)
    assert torch.allclose(shared.advantages, per_group.advantages, atol = 1e-5)

@param('group_dim', (-2, 0))
def test_gae_next_value_grouped_shapes(group_dim):
    b, g, n = 4, 3, 5
    rewards = torch.randn(b, g, n) if group_dim == -2 else torch.randn(g, b, n)
    values = torch.randn(b, g, n) if group_dim == -2 else torch.randn(g, b, n)
    masks = torch.ones(b, n)
    next_value = torch.randn(b)

    out = calc_gae(rewards, values, masks = masks, next_value = next_value, group_dim = group_dim, use_accelerated = False)
    assert out.shape == rewards.shape

    # (b,) and (b, g) next values must agree

    next_values_flat = next_value[:, None].expand(b, g) if group_dim == -2 else next_value[None, :].expand(g, b)
    out_2d = calc_gae(rewards, values, masks = masks, next_value = next_values_flat, group_dim = group_dim, use_accelerated = False)

    assert torch.allclose(out, out_2d, atol = 1e-5)

# bootstrapping with t + 1 values

def test_gae_t_plus_one_values_with_next_values():
    b, t = 2, 6
    rewards = torch.randn(b, t)
    values = torch.randn(b, t + 1)
    next_values = torch.randn(b, t)
    masks = torch.ones(b, t)

    out = calc_gae(rewards, values, masks, next_values = next_values, use_accelerated = False, return_advantages = True)
    expected = calc_gae(rewards, values[:, :-1], masks, next_values = next_values, use_accelerated = False, return_advantages = True)

    assert torch.allclose(out.returns, expected.returns, atol = 1e-5)
    assert torch.allclose(out.advantages, expected.advantages, atol = 1e-5)

def test_gae_t_plus_one_values_with_next_value():
    b, t = 2, 6
    rewards = torch.randn(b, t)
    values = torch.randn(b, t + 1)
    next_value = torch.randn(b)
    masks = torch.ones(b, t)

    out = calc_gae(rewards, values, masks, next_value = next_value, use_accelerated = False, return_advantages = True)
    expected = calc_gae(rewards, values[:, :-1], masks, next_value = next_value, use_accelerated = False, return_advantages = True)

    assert torch.allclose(out.returns, expected.returns, atol = 1e-5)
    assert torch.allclose(out.advantages, expected.advantages, atol = 1e-5)

# variable lengths with explicit bootstrapping

def test_gae_variable_lengths_explicit_bootstrap():
    rewards = torch.tensor([[1., 2., 3., 4.]])
    values = torch.zeros_like(rewards)
    lens = torch.tensor([2])
    gamma = 0.9

    # next_value is the bootstrap at the last valid timestep

    out = calc_gae(rewards, values, lens = lens, gamma = gamma, lam = 1.0, next_value = 10., use_accelerated = False, return_advantages = True)
    assert torch.allclose(out.returns, torch.tensor([[10.9, 11.0, 0., 0.]]), atol = 1e-5)

    # explicit next values are respected at the last valid timestep

    next_values = torch.tensor([[0., 10., 0., 0.]])
    out = calc_gae(rewards, values, lens = lens, gamma = gamma, lam = 1.0, next_values = next_values, use_accelerated = False, return_advantages = True)
    assert torch.allclose(out.returns, torch.tensor([[10.9, 11.0, 0., 0.]]), atol = 1e-5)

    # but a terminated last timestep should still not bootstrap

    masks = torch.tensor([[1., 0., 1., 1.]])
    out = calc_gae(rewards, values, masks = masks, lens = lens, gamma = gamma, lam = 1.0, next_values = next_values, use_accelerated = False, return_advantages = True)
    assert torch.allclose(out.returns, torch.tensor([[2.8, 2.0, 0., 0.]]), atol = 1e-5)

def test_gae_variable_lengths_nan_padding_does_not_leak():
    rewards = torch.ones(2, 4)
    values = torch.tensor([
        [1., 1., float('nan'), float('nan')],
        [1., 1., 1., 1.]
    ])
    lens = torch.tensor([2, 4])

    out = calc_gae(rewards, values, lens = lens, use_accelerated = False, return_advantages = True)

    assert torch.isfinite(out.returns).all()
    assert torch.isfinite(out.advantages).all()
    assert torch.all(out.returns[0, 2:] == 0.)
    assert torch.allclose(out.returns[0, :2], torch.tensor([1.99, 1.]), atol = 1e-5)

def test_calc_returns_variable_lengths_next_value():
    b, n = 3, 6
    rewards = torch.randn(b, n)
    masks = torch.ones(b, n)
    masks[0, 3] = 0.
    lens = torch.tensor([4, 6, 2])
    next_value = torch.randn(b)

    returns = calc_returns(rewards, masks = masks, gamma = 0.99, next_value = next_value, lens = lens, use_accelerated = False)

    # equivalent to gae with zero values and lam = 1

    expected = calc_gae(
        rewards, torch.zeros_like(rewards), masks = masks, gamma = 0.99, lam = 1.0,
        next_value = next_value, lens = lens, use_accelerated = False
    )

    assert torch.allclose(returns, expected, atol = 1e-5)

def test_calc_returns_nan_padding_does_not_leak():
    rewards = torch.tensor([
        [1., 1., float('nan'), float('nan')],
        [1., 1., 1., 1.]
    ])
    lens = torch.tensor([2, 4])

    returns = calc_returns(rewards, lens = lens, use_accelerated = False)

    assert torch.isfinite(returns).all()
    assert torch.all(returns[0, 2:] == 0.)

def test_lens_zero_and_beyond_sequence_length():
    b, n = 3, 5
    rewards = torch.randn(b, n)
    values = torch.randn(b, n)
    masks = torch.ones(b, n)
    lens = torch.tensor([0, 3, n + 2])
    next_value = torch.randn(b)

    out = calc_gae(rewards, values, masks = masks, lens = lens, next_value = next_value, use_accelerated = False, return_advantages = True)

    assert torch.isfinite(out.returns).all()
    assert torch.isfinite(out.advantages).all()
    assert torch.all(out.returns[0] == 0.)
    assert torch.all(out.advantages[0] == 0.)

    returns = calc_returns(rewards, masks = masks, next_value = next_value, lens = lens, use_accelerated = False)

    assert torch.isfinite(returns).all()
    assert torch.all(returns[0] == 0.)

def test_gae_grouped_lens_next_value():
    b, g, n = 3, 2, 5
    rewards = torch.randn(g, b, n)
    values = torch.randn(g, b, n)
    masks = torch.ones(b, n)
    lens = torch.tensor([2, 5, 3])
    next_value = torch.randn(b)

    out = calc_gae(rewards, values, masks = masks, lens = lens, next_value = next_value, group_dim = 0, use_accelerated = False, return_advantages = True)

    # equivalent to broadcasting the next value across the groups

    out_broadcast = calc_gae(
        rewards, values, masks = masks, lens = lens, next_value = next_value[None].expand(g, b),
        group_dim = 0, use_accelerated = False, return_advantages = True
    )

    assert torch.allclose(out.returns, out_broadcast.returns, atol = 1e-5)
    assert torch.allclose(out.advantages, out_broadcast.advantages, atol = 1e-5)

# misc robustness

def test_combine_grouped_advantages_requires_group_dim():
    with pytest.raises(AssertionError):
        combine_grouped_advantages(torch.randn(8), weights = torch.tensor([0.5, 0.5]))

@pytest.mark.skipif(torch.cuda.is_available(), reason = 'accelerated scan is used when cuda is available')
def test_use_accelerated_falls_back_on_cpu():
    rewards = torch.randn(2, 5)
    values = torch.randn(2, 5)

    out = calc_gae(rewards, values, use_accelerated = True)

    assert out.shape == rewards.shape

# value clipping tests

def test_value_clipping_losses():
    values = torch.randn(8)
    old_values = torch.randn(8)
    returns = torch.randn(8)

    loss_std = clipped_value_loss_fn(values, old_values, returns, clip = 0.2)
    assert loss_std.ndim == 0 and loss_std >= 0.

    loss_adv = improved_value_clipping_fn(values, old_values, returns, clip = 0.2)
    assert loss_adv.ndim == 0 and loss_adv >= 0.

def test_value_clipping_masked_and_lens():
    values = torch.randn(2, 6, requires_grad = True)
    old_values = torch.randn(2, 6)
    returns = torch.randn(2, 6)
    lens = torch.tensor([3, 6])

    loss_std = clipped_value_loss_fn(values, old_values, returns, lens = lens)
    assert loss_std.ndim == 0 and loss_std >= 0.
    loss_std.backward()
    assert values.grad is not None

    values.grad = None
    loss_adv = improved_value_clipping_fn(values, old_values, returns, lens = lens)
    assert loss_adv.ndim == 0 and loss_adv >= 0.
    loss_adv.backward()
    assert values.grad is not None


# single sequence, no batch dimension

def test_gae_single_sequence():
    seq_len = 16
    rewards = torch.randn(seq_len)
    values = torch.randn(seq_len)
    masks = (torch.rand(seq_len) > 0.1).float()
    next_value = 2.5

    returns, advantages = calc_gae(rewards, values, masks = masks, next_value = next_value, use_accelerated = False, return_advantages = True)

    batched = calc_gae(rewards[None], values[None], masks = masks[None], next_value = next_value, use_accelerated = False, return_advantages = True)

    assert returns.shape == rewards.shape
    assert advantages.shape == rewards.shape

    assert torch.allclose(returns, batched.returns[0], atol = 1e-5)
    assert torch.allclose(advantages, batched.advantages[0], atol = 1e-5)

    # a tensor next value with batch size 1 is also valid

    tensor_next = calc_gae(rewards, values, masks = masks, next_value = torch.tensor([next_value]), use_accelerated = False)
    assert torch.allclose(tensor_next, batched.returns[0], atol = 1e-5)

def test_calc_returns_single_sequence():
    seq_len = 16
    rewards = torch.randn(seq_len)
    masks = (torch.rand(seq_len) > 0.1).float()
    next_value = 2.5

    returns = calc_returns(rewards, masks = masks, next_value = next_value, use_accelerated = False)
    batched_returns = calc_returns(rewards[None], masks = masks[None], next_value = next_value, use_accelerated = False)

    assert returns.shape == rewards.shape
    assert torch.allclose(returns, batched_returns[0], atol = 1e-5)

    tensor_next_returns = calc_returns(rewards, masks = masks, next_value = torch.tensor([next_value]), use_accelerated = False)
    assert torch.allclose(tensor_next_returns, batched_returns[0], atol = 1e-5)
