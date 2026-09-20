import torch
from torch_einops_utils import z_score, masked_mean

from x_ppo.ppo import ppo_actor_loss

# helpers

def exists(val):
    return val is not None

# simple policy optimization - Xie et al. https://arxiv.org/abs/2401.16025

def spo_actor_loss(
    action_log_probs,
    old_action_log_probs,
    advantages,
    eps_clip = 0.2,
    mask = None,
    normalize_advantages = False,
    delightful = False,
    delight_temp = 1.,
    asymmetric = False
):
    if normalize_advantages:
        advantages = z_score(advantages, mask = mask)

    # delightful policy gradient - Ian Osband https://arxiv.org/abs/2603.14608

    if delightful:
        gate = (-action_log_probs * advantages / delight_temp).sigmoid().detach()
        advantages = advantages * gate

    ratios = (action_log_probs - old_action_log_probs).exp()

    loss = -(
        ratios * advantages -
        (advantages.abs() * (ratios - 1.).square()) / (2 * eps_clip)
    )

    if asymmetric:
        # asymmetric spo - https://arxiv.org/abs/2510.06062v1
        ppo_loss = ppo_actor_loss(action_log_probs, old_action_log_probs, advantages, eps_clip = eps_clip)
        loss = torch.where(advantages > 0, ppo_loss, loss)

    if not exists(mask):
        return loss

    return masked_mean(loss, mask = mask)
