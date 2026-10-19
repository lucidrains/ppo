import torch
from torch_einops_utils import z_score, masked_mean

from x_ppo.ppo import cast_tuple, force_fp32, exists
from x_ppo.values import resolve_grouped_mask

@force_fp32
def hpo_actor_loss(
    action_log_probs,
    old_action_log_probs,
    advantages,
    eps_clip = 0.2,
    mask = None,
    normalize_advantages = False,
    delightful = False,
    delight_temp = 1.,
    log_ratio_clamp = None,
    group_dim = -2
):
    mask = resolve_grouped_mask(advantages, action_log_probs, mask, group_dim)

    if normalize_advantages:
        advantages = z_score(advantages, mask = mask)

    if delightful:
        gate = (-action_log_probs * advantages / delight_temp).sigmoid().detach()
        advantages = advantages * gate

    log_ratios = action_log_probs - old_action_log_probs
    eps_clip_low, eps_clip_high = cast_tuple(eps_clip, 2)

    if exists(log_ratio_clamp):
        clamp_low, clamp_high = log_ratio_clamp
        bound = 2 * max(eps_clip_low, eps_clip_high)

        if clamp_low > -bound or clamp_high < bound:
            raise ValueError(f'log_ratio_clamp too tight for hpo - needs ({-bound}, {bound})')

        # straight-through, preserve restoring gradient
        clamped = log_ratios.clamp(clamp_low, clamp_high)
        log_ratios = log_ratios + (clamped - log_ratios).detach()

    eps = torch.where(advantages >= 0., eps_clip_high, eps_clip_low)

    # proportional target + |A| gate - quadratic below 1 sigma, linear above
    target = (eps * advantages).clamp(-eps_clip_low, eps_clip_high)
    weight = advantages.abs()

    # per-sample huber
    error = log_ratios - target
    diff = error.abs()
    loss = weight * torch.where(diff < eps, 0.5 * error.square() / eps, diff - 0.5 * eps)

    return masked_mean(loss, mask = mask) if exists(mask) else loss
