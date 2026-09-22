import torch
from torch_einops_utils import z_score, masked_mean

from x_ppo.ppo import ppo_actor_loss, cast_tuple

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
    asymmetric = False,
    dual_clip = False,
    dual_clip_threshold = 3.
):
    if normalize_advantages:
        advantages = z_score(advantages, mask = mask)

    # delightful policy gradient - Ian Osband https://arxiv.org/abs/2603.14608

    if delightful:
        gate = (-action_log_probs * advantages / delight_temp).sigmoid().detach()
        advantages = advantages * gate

    ratios = (action_log_probs - old_action_log_probs).exp()

    # decoupled clipping - Zhou et al. https://arxiv.org/abs/2503.14476

    eps_clip_low, eps_clip_high = cast_tuple(eps_clip, 2)

    eps = torch.where(ratios >= 1., eps_clip_high, eps_clip_low)

    surr = (
        ratios * advantages -
        (advantages.abs() * (ratios - 1.).square()) / (2 * eps)
    )

    # dual-clip ppo - Ye et al. https://arxiv.org/abs/1912.09729

    if dual_clip:
        surr = torch.where(advantages < 0, torch.max(surr, dual_clip_threshold * advantages), surr)

    loss = -surr

    if asymmetric:
        # asymmetric spo - https://arxiv.org/abs/2510.06062v1
        ppo_loss = ppo_actor_loss(
            action_log_probs,
            old_action_log_probs,
            advantages,
            eps_clip = eps_clip,
            dual_clip = dual_clip,
            dual_clip_threshold = dual_clip_threshold
        )
        loss = torch.where(advantages > 0, ppo_loss, loss)

    if not exists(mask):
        return loss

    return masked_mean(loss, mask = mask)
