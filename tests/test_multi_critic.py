import pytest
import torch
from torch import einsum
from torch_einops_utils import z_score, masked_mean, lens_to_mask

from x_ppo import (
    ppo_actor_loss,
    spo_actor_loss,
    calc_gae,
    combine_grouped_advantages,
    clipped_value_loss_fn,
    improved_value_clipping_fn
)

param = pytest.mark.parametrize

def exists(val):
    return val is not None

# multi-critic - grouped advantages from separate value networks

@param('reduce, reference', (
    ('mean', lambda t: t.mean(dim = 1)),
    ('sum', lambda t: t.sum(dim = 1)),
    ('max', lambda t: t.amax(dim = 1)),
    ('min', lambda t: t.amin(dim = 1))
))
def test_combine_grouped_advantages_reduce_modes(reduce, reference):
    b, g, n = 2, 4, 6
    advantages = torch.randn(b, g, n)

    combined = combine_grouped_advantages(advantages, normalize = False, group_dim = 1, reduce = reduce)

    assert torch.allclose(combined, reference(advantages))

def test_combine_grouped_advantages_weights_and_callable():
    advantages = torch.randn(2, 4, 6)
    weights = torch.tensor([0.5, 0.3, 0.2, 1.0])
    expected = einsum('b g n, g -> b n', advantages, weights)

    weighted_sum = combine_grouped_advantages(advantages, weights = weights, normalize = False, group_dim = 1, reduce = 'sum')
    assert torch.allclose(weighted_sum, expected)

    # for 'mean' the weights act as importance and are renormalized

    weighted_mean = combine_grouped_advantages(advantages, weights = weights, normalize = False, group_dim = 1)
    assert torch.allclose(weighted_mean, expected / weights.sum())

    # custom combination rules receive the group first layout

    reduce_fn = lambda grouped: grouped[0] - grouped[1:].mean(dim = 0)
    combined = combine_grouped_advantages(advantages, normalize = False, group_dim = 1, reduce = reduce_fn)
    assert torch.allclose(combined, reduce_fn(advantages.movedim(1, 0)))

    with pytest.raises(ValueError):
        combine_grouped_advantages(advantages, group_dim = 1, reduce = 'median')

def test_combine_grouped_advantages_normalization():
    b, g, n = 2, 3, 6

    # each group on a wildly different reward scale

    advantages = torch.randn(b, g, n) * torch.tensor([1., 5., 25.])[None, :, None]
    mask = lens_to_mask(torch.tensor([4, 6]), n)

    combined = combine_grouped_advantages(advantages, group_dim = 1, mask = mask)

    normed = z_score(advantages, dim = (0, 2), mask = mask[:, None])
    assert torch.allclose(combined, masked_mean(normed, mask = mask[:, None], dim = 1), atol = 1e-5)

    # weighted mean is invariant to the scale of the weights

    weights = torch.tensor([1., 2., 4.])
    assert torch.allclose(
        combine_grouped_advantages(advantages, weights = weights, normalize = False, group_dim = 1),
        combine_grouped_advantages(advantages, weights = weights * 10, normalize = False, group_dim = 1)
    )

def test_calc_gae_grouped_weights_match_manual_combine():
    b, g, n = 2, 3, 6
    rewards = torch.randn(b, g, n)
    values = torch.randn(b, g, n)
    masks = torch.ones(b, n)
    masks[:, -1] = 0.
    weights = torch.tensor([0.5, 0.3, 0.2])

    out = calc_gae(rewards, values, masks = masks, weights = weights, group_dim = 1, use_accelerated = False, return_advantages = True)
    raw = calc_gae(rewards, values, masks = masks, group_dim = 1, use_accelerated = False, return_advantages = True)

    # returns stay per group for the separate value networks

    assert out.returns.shape == raw.returns.shape == (b, g, n)

    expected = combine_grouped_advantages(raw.advantages, weights = weights, group_dim = 1, mask = masks)
    assert torch.allclose(out.advantages, expected, atol = 1e-5)

    # alternative reduction rules can be threaded through

    out_sum = calc_gae(rewards, values, masks = masks, weights = weights, group_reduce = 'sum', group_dim = 1, use_accelerated = False, return_advantages = True)
    expected_sum = combine_grouped_advantages(raw.advantages, weights = weights, group_dim = 1, mask = masks, reduce = 'sum')
    assert torch.allclose(out_sum.advantages, expected_sum, atol = 1e-5)

def test_value_clipping_losses_grouped():
    b, g, n = 2, 3, 6
    values = torch.randn(b, g, n, requires_grad = True)
    old_values = torch.randn(b, g, n)
    returns = torch.randn(b, g, n)
    lens = torch.tensor([3, 6])

    loss = clipped_value_loss_fn(values, old_values, returns, lens = lens)
    loss.backward()
    assert exists(values.grad)

    values.grad = None
    loss_adv = improved_value_clipping_fn(values, old_values, returns, lens = lens)
    loss_adv.backward()
    assert exists(values.grad)

    # a singleton group is equivalent to an ungrouped value head

    single = clipped_value_loss_fn(values[:, :1], old_values[:, :1], returns[:, :1], lens = lens)
    ungrouped = clipped_value_loss_fn(values[:, 0], old_values[:, 0], returns[:, 0], lens = lens)
    assert torch.allclose(single, ungrouped, atol = 1e-5)

    # a mask is equivalent to lens, in either layout

    mask = lens_to_mask(lens, n)

    loss_mask = clipped_value_loss_fn(values, old_values, returns, mask = mask)
    assert torch.allclose(loss_mask, loss, atol = 1e-5)

    # group trailing layout

    loss_last = clipped_value_loss_fn(values.movedim(1, -1), old_values.movedim(1, -1), returns.movedim(1, -1), lens = lens, group_dim = -1)
    assert torch.allclose(loss_last, loss, atol = 1e-5)

    loss_last_mask = clipped_value_loss_fn(values.movedim(1, -1), old_values.movedim(1, -1), returns.movedim(1, -1), mask = mask, group_dim = -1)
    assert torch.allclose(loss_last_mask, loss, atol = 1e-5)

def test_multi_critic_flow():
    b, g, n = 2, 4, 6

    # grouped rewards - e.g. task / style / regularization / post-task

    rewards = torch.randn(b, g, n)
    values = torch.randn(b, g, n)
    masks = torch.ones(b, n)
    masks[:, -1] = 0.

    gae = calc_gae(rewards, values, masks = masks, group_dim = 1, use_accelerated = False, return_advantages = True)
    assert gae.returns.shape == gae.advantages.shape == (b, g, n)

    weights = torch.tensor([1., 0.5, 0.25, 0.1])
    combined = combine_grouped_advantages(gae.advantages, weights = weights, group_dim = 1, mask = masks)
    assert combined.shape == (b, n)

    action_log_probs = torch.randn(b, n, requires_grad = True)
    old_action_log_probs = torch.randn(b, n)

    loss = ppo_actor_loss(action_log_probs, old_action_log_probs, combined, mask = masks)
    loss.backward()
    assert exists(action_log_probs.grad)

    # uncombined grouped advantages are rejected instead of silently broadcast

    with pytest.raises(ValueError):
        ppo_actor_loss(action_log_probs, old_action_log_probs, gae.advantages)

    with pytest.raises(ValueError):
        spo_actor_loss(action_log_probs, old_action_log_probs, gae.advantages)

    # per group value loss across the separate heads

    values_new = torch.randn(b, g, n, requires_grad = True)
    critic_loss = clipped_value_loss_fn(values_new, values, gae.returns, lens = torch.tensor([4, 6]))
    critic_loss.backward()
    assert exists(values_new.grad)

@param('actor_loss_fn', (ppo_actor_loss, spo_actor_loss))
@param('group_first', (False, True))
def test_actor_loss_per_group(actor_loss_fn, group_first):
    b, n, g = 2, 5, 3
    shape = (b, g, n) if group_first else (b, n, g)
    group_dim = -2 if group_first else -1

    action_log_probs = torch.randn(*shape, requires_grad = True)
    old_action_log_probs = torch.randn(*shape)
    group_advantages = torch.randn(*shape)
    mask = torch.ones(b, n)

    loss = actor_loss_fn(action_log_probs, old_action_log_probs, group_advantages, mask = mask, group_dim = group_dim)
    loss.backward()
    assert exists(action_log_probs.grad)

    # unreduced per group losses - allowing combination rules that act on losses instead of advantages

    per_group_loss = actor_loss_fn(action_log_probs, old_action_log_probs, group_advantages, group_dim = group_dim)
    assert per_group_loss.shape == shape

    weights = torch.tensor([1., 2., 3.])
    combined = combine_grouped_advantages(per_group_loss, weights = weights, normalize = False, group_dim = group_dim, reduce = 'sum')
    reduced = masked_mean(combined, mask = mask)
    assert reduced.ndim == 0
