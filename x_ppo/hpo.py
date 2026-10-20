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
    min_width = 1e-8,
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

    # proportional stop target, saturating at |A| = 1
    prop_target = eps * advantages.clamp(-1., 1.)
    approaching = (log_ratios - prop_target) * advantages < 0.

    # stop at proportional target - track inside trust band, restore past bound
    target = torch.where(approaching, prop_target, log_ratios.detach().clamp(-eps_clip_low, eps_clip_high))
    weight = advantages.abs()

    # huber width - |prop target| on approach, eps on restore
    width = torch.where(approaching, prop_target.abs().clamp(min = min_width), eps)

    error = log_ratios - target
    abs_error = error.abs()
    loss = weight * torch.where(abs_error < width, 0.5 * error.square() / width, abs_error - 0.5 * width)

    return masked_mean(loss, mask = mask) if exists(mask) else loss
