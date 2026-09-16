import torch
from torch_einops_utils import z_score, masked_mean

# helpers

def exists(val):
    return val is not None

def ppo_actor_loss(
    action_log_probs,
    old_action_log_probs,
    advantages,
    eps_clip = 0.2,
    mask = None,
    normalize_advantages = False
):
    if normalize_advantages:
        advantages = z_score(advantages, mask = mask)

    ratios = (action_log_probs - old_action_log_probs).exp()

    surr1 = ratios * advantages
    surr2 = ratios.clamp(1. - eps_clip, 1. + eps_clip) * advantages

    loss = - torch.min(surr1, surr2)

    if not exists(mask):
        return loss

    return masked_mean(loss, mask = mask)
