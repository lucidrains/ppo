import pytest
import torch
from torch_einops_utils import lens_to_mask

from x_ppo import ppo_actor_loss

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
