import pytest
import torch
from torch_einops_utils import lens_to_mask, z_score

from x_ppo import ppo_actor_loss

param = pytest.mark.parametrize


def test_actor_loss_is_negative_advantages_at_ratio_one():
    action_log_probs = torch.randn(4)
    advantages = torch.randn(4)

    loss = ppo_actor_loss(action_log_probs, action_log_probs, advantages)

    assert torch.allclose(loss, -advantages)


def test_actor_loss_clips_ratios():
    old_action_log_probs = torch.zeros(2)
    action_log_probs = torch.tensor([10., -10.])
    advantages = torch.tensor([1., -1.])

    loss = ppo_actor_loss(action_log_probs, old_action_log_probs, advantages, eps_clip = 0.2)

    assert torch.allclose(loss, torch.tensor([-1.2, 0.8]))


def test_actor_loss_normalizes_advantages():
    action_log_probs = torch.randn(8)
    old_action_log_probs = torch.randn(8)
    advantages = torch.randn(8)

    loss = ppo_actor_loss(action_log_probs, old_action_log_probs, advantages, normalize_advantages = True)
    expected = ppo_actor_loss(action_log_probs, old_action_log_probs, z_score(advantages))

    assert torch.allclose(loss, expected)


@param("shape, use_mask", [
    ((4,), False),
    ((2, 4), False),
    ((3, 5), False),
    ((4,), True),
    ((2, 4), True),
    ((3, 5), True),
])
def test_actor_loss_shapes(shape, use_mask):
    log_probs = torch.randn(shape)

    mask = None
    if use_mask:
        mask = torch.ones(shape, dtype = torch.bool)
        mask[..., -1] = False

    loss = ppo_actor_loss(log_probs, log_probs, log_probs, mask = mask)

    # regular batch and batch seq without mask are elementwise

    if not use_mask:
        assert loss.shape == shape
        return

    # with mask is a scalar masked mean

    elementwise = ppo_actor_loss(log_probs, log_probs, log_probs)

    assert loss.ndim == 0
    assert torch.allclose(loss, (elementwise * mask).sum() / mask.sum())


def test_actor_loss_handles_variable_lengths():
    batch, seq = 2, 4

    action_log_probs = torch.randn(batch, seq)
    old_action_log_probs = torch.randn(batch, seq)
    advantages = torch.randn(batch, seq)

    mask = lens_to_mask(torch.tensor([2, 4]), seq)

    loss = ppo_actor_loss(action_log_probs, old_action_log_probs, advantages, mask = mask, normalize_advantages = True)

    assert loss.ndim == 0

    perturbed = advantages.clone()
    perturbed[~mask] = 1e6

    perturbed_loss = ppo_actor_loss(action_log_probs, old_action_log_probs, perturbed, mask = mask, normalize_advantages = True)

    assert torch.allclose(loss, perturbed_loss)


def test_actor_loss_backpropagates():
    action_log_probs = torch.randn(4, requires_grad = True)

    loss = ppo_actor_loss(action_log_probs, torch.randn(4), torch.randn(4)).mean()
    loss.backward()

    assert action_log_probs.grad is not None
