from functools import partial

import pytest
import torch

from x_ppo import hpo_actor_loss, ppo_actor_loss


def grad_magnitude(loss_fn, log_ratios, advantages):
    x = log_ratios.clone().requires_grad_()
    loss = loss_fn(x, torch.zeros_like(x), advantages).sum()
    return torch.autograd.grad(loss, x)[0].abs()


def test_hpo_asymmetric_gradient():
    advantages = torch.tensor([0.2, 1., -0.4, -2.])
    log_probs = torch.zeros_like(advantages, requires_grad = True)

    hpo_actor_loss(log_probs, torch.zeros_like(advantages), advantages, eps_clip = (0.2, 0.28)).sum().backward()

    assert torch.allclose(log_probs.grad, -advantages * advantages.abs().clamp(max = 1.))


def test_hpo_asymmetric_huber_transition():
    eps_low, eps_high = 0.2, 0.28
    log_ratios = torch.tensor([2 * eps_high, -2 * eps_low], requires_grad = True)

    hpo_actor_loss(
        log_ratios,
        torch.zeros(2),
        torch.tensor([1., -1.]),
        eps_clip = (eps_low, eps_high)
    ).sum().backward()

    assert torch.allclose(log_ratios.grad.abs(), torch.ones(2))


def test_hpo_rejects_tight_log_ratio_clamp():
    with pytest.raises(ValueError):
        hpo_actor_loss(torch.zeros(1), torch.zeros(1), torch.ones(1), log_ratio_clamp = (-0.2, 0.2))


def test_hpo_log_ratio_clamp_gradient():
    log_probs = torch.tensor([5., -5.], requires_grad = True)

    hpo_actor_loss(
        log_probs,
        torch.zeros(2),
        torch.tensor([1., -1.]),
        log_ratio_clamp = (-1., 1.)
    ).sum().backward()

    assert torch.allclose(log_probs.grad.abs(), torch.ones(2))


def test_hpo_gradient_diminishes_near_target():
    eps = 0.2
    distances = torch.tensor([0., 0.05, 0.1, 0.15, 0.2, 0.5, 1.])
    log_ratios = eps - distances
    advantages = torch.ones_like(log_ratios)

    hpo = grad_magnitude(
        partial(hpo_actor_loss, eps_clip = eps),
        log_ratios,
        advantages
    )

    ppo = grad_magnitude(
        partial(ppo_actor_loss, eps_clip = eps),
        log_ratios,
        advantages
    )

    # smooth decay to target

    assert torch.all(hpo[1:] >= hpo[:-1] - 1e-6)
    assert hpo[0] < 1e-6
    assert torch.allclose(hpo[4], torch.ones(()))

    # clip cliff

    assert ppo[0] < 1e-6
    assert ppo[1] > 1.
    assert ppo[1] > ppo[4]
