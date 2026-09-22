import torch
from torch_einops_utils import z_score, masked_mean

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def ppo_actor_loss(
    action_log_probs,
    old_action_log_probs,
    advantages,
    eps_clip = 0.2,
    mask = None,
    normalize_advantages = False,
    delightful = False,
    delight_temp = 1.,
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

    surr1 = ratios * advantages
    surr2 = ratios.clamp(1. - eps_clip_low, 1. + eps_clip_high) * advantages

    surr = torch.min(surr1, surr2)

    # dual-clip ppo - Ye et al. https://arxiv.org/abs/1912.09729

    if dual_clip:
        surr = torch.where(advantages < 0, torch.max(surr, dual_clip_threshold * advantages), surr)

    loss = -surr

    if not exists(mask):
        return loss

    return masked_mean(loss, mask = mask)
