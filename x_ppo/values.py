from __future__ import annotations
from typing import NamedTuple, Callable

import torch
from torch import Tensor, as_tensor, cat, is_tensor
import torch.nn.functional as F

from einops import rearrange
import einx

from assoc_scan import AssocScan

from torch_einops_utils import (
    z_score,
    pack_with_inverse,
    lens_to_mask,
    pad_right_at_dim,
    pad_right_ndim_to,
    batched_index_select,
    tree_map_tensor,
    tree_map_tensor_to_device,
    masked_reduce
)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)

def align_group_dim(t, group_dim):
    if not is_tensor(t):
        return t

    if t.ndim == 2:
        return rearrange(t, 'b n -> b 1 n')

    return t.movedim(group_dim, -2)

def align_grouped_next_value(next_value, group_dim, ndim):
    if not is_tensor(next_value) or next_value.ndim <= 1:
        return next_value

    # group first layouts have a trailing batch dimension

    return next_value.movedim(0, -1) if group_dim in (0, -ndim) else next_value

def lens_to_last_mask(lens, max_len):
    # (b n) one hot mask at each sequence's last valid timestep

    last_idx = (lens - 1).clamp(min = 0, max = max_len - 1)
    return F.one_hot(last_idx.long(), max_len).bool()

class GAE(NamedTuple):
    returns: Tensor
    advantages: Tensor

# dones to masks helper (e.g. gymnasium)

def dones_to_masks(
    terminated: Tensor | None = None,         # (b n)
    truncated: Tensor | None = None,          # (b n)
    dones: Tensor | None = None               # (b n)
) -> tuple[Tensor, Tensor] | Tensor:
    terminated = default(terminated, dones)

    if not exists(terminated) and not exists(truncated):
        raise ValueError('at least one of `terminated`, `truncated` or `dones` must be passed in')

    if not exists(terminated):
        terminated = torch.zeros_like(truncated, dtype = torch.bool)

    term_mask = (~terminated.bool()).float()

    if not exists(truncated):
        return term_mask

    done_mask = (~(terminated.bool() | truncated.bool())).float()
    return term_mask, done_mask

# grouped advantages (multi-critic)

def combine_grouped_advantages(
    advantages: Tensor,                       # (b n g) or (b g n)
    weights: Tensor | None = None,            # (g)
    normalize: bool = True,
    group_dim: int = -2,
    mask: Tensor | None = None,               # (b n)
    eps: float = 1e-5
) -> Tensor:                                  # (b n)
    assert advantages.ndim >= 2, 'advantages must have at least 2 dimensions'

    if is_tensor(mask) and mask.ndim < advantages.ndim:
        mask = mask.unsqueeze(group_dim)

    if normalize:
        adv_groups = advantages.movedim(group_dim, 0)
        mask_groups = mask.expand_as(advantages).movedim(group_dim, 0).bool() if is_tensor(mask) else None
        normed = z_score(adv_groups, dim = tuple(range(1, adv_groups.ndim)), mask = mask_groups, eps = eps)
        advantages = normed.movedim(0, group_dim)

    if not exists(weights):
        return advantages.mean(dim = group_dim)

    weights = as_tensor(weights, device = advantages.device, dtype = advantages.dtype)
    return advantages.movedim(group_dim, -1) @ weights

# scans

accelerated_scan = AssocScan(reverse = True, use_accelerated = True, has_no_feature_dim = True)
cpu_scan = AssocScan(reverse = True, use_accelerated = False, has_no_feature_dim = True)

def reverse_assoc_scan(
    gates: Tensor,                            # (* n)
    inputs: Tensor,                           # (* n)
    use_accelerated: bool | None = None
) -> Tensor:
    # accelerated scan is only available on cuda

    use_accelerated = default(use_accelerated, inputs.is_cuda) and inputs.is_cuda
    scan = accelerated_scan if use_accelerated else cpu_scan

    inputs, inverse_pack = pack_with_inverse(inputs, '* n')
    gates, _ = pack_with_inverse(gates, '* n')

    return inverse_pack(scan(gates, inputs))

# generalized advantage estimation - Schulman et al.

def calc_gae(
    rewards: Tensor,                          # (b n) or (b n g)
    values: Tensor,                           # (b n) or (b n + 1) or (b n g)
    masks: Tensor | None = None,              # (b n)
    gamma: float | Tensor = 0.99,
    lam: float | Tensor = 0.95,
    done_masks: Tensor | None = None,         # (b n)
    next_value: float | Tensor | None = None, # () or (b) or (b g)
    next_values: Tensor | None = None,        # (b n) or (b n g)
    lens: Tensor | None = None,               # (b)
    weights: Tensor | None = None,            # (g)
    normalize_grouped: bool = True,
    group_dim: int = -2,
    use_accelerated: bool | None = None,
    return_advantages: bool = False
) -> Tensor | GAE:
    device, dtype = rewards.device, rewards.dtype

    # handle single sequence input, with no batch dimension

    is_single_sequence = rewards.ndim == 1

    if is_single_sequence:
        rewards, values, masks, done_masks, next_values, gamma, lam = tree_map_tensor(
            lambda t: t[None] if t.ndim == 1 else t,
            (rewards, values, masks, done_masks, next_values, gamma, lam)
        )

    grouped = rewards.ndim > 2

    # canonicalize grouped inputs to (b g n), with shared value heads broadcast across groups

    masks = default(masks, 1.)
    done_masks = default(done_masks, masks)

    if grouped:
        next_value = align_grouped_next_value(next_value, group_dim, rewards.ndim)

        rewards = rewards.movedim(group_dim, -2)
        values = align_group_dim(values, group_dim)
        next_values = align_group_dim(next_values, group_dim) if exists(next_values) else next_values

        masks = align_group_dim(masks, group_dim)
        done_masks = align_group_dim(done_masks, group_dim)

    seq_len = rewards.shape[-1]

    values, next_values, masks, done_masks, gamma, lam, lens = tree_map_tensor_to_device(
        (values, next_values, masks, done_masks, gamma, lam, lens),
        device
    )

    assert values.shape[-1] in (seq_len, seq_len + 1), f'values must have sequence length of {seq_len} or {seq_len + 1}, but received {values.shape[-1]}'

    has_explicit_bootstrap = exists(next_values) or exists(next_value)

    # derive next values from values if not explicitly provided

    values_next = None
    if values.shape[-1] == seq_len + 1:
        values, values_next = values[..., :-1], values[..., 1:]

    if exists(next_values):
        assert next_values.shape[-1] == seq_len, f'next_values must have sequence length of {seq_len}, but received {next_values.shape[-1]}'
    elif exists(values_next) and not exists(next_value):
        next_values = values_next
    else:
        next_val = as_tensor(default(next_value, 0.), device = device, dtype = dtype)
        next_val = pad_right_ndim_to(next_val, values.ndim - 1).expand_as(values[..., 0])

        next_values = cat((values[..., 1:], next_val[..., None]), dim = -1)

        # with variable lengths, the bootstrap value belongs at the last valid timestep, not the padded end

        if exists(next_value) and exists(lens):
            last_mask = lens_to_last_mask(lens, seq_len)
            pattern = 'b n, b g, b g n -> b g n' if grouped else 'b n, b, b n -> b n'
            next_values = einx.where(pattern, last_mask, next_val, next_values)

    # variable length masking
    # with an explicit bootstrap, the last valid timestep stays unmasked, but recurrence is always cut across padding

    len_mask = None
    if exists(lens):
        len_mask = lens_to_mask(lens, max_len = seq_len)
        is_next = pad_right_at_dim(len_mask[..., 1:], 1, value = False)

        if grouped:
            len_mask = rearrange(len_mask, 'b n -> b 1 n')
            is_next = rearrange(is_next, 'b n -> b 1 n')

        masks = masks * (len_mask if has_explicit_bootstrap else is_next)
        done_masks = done_masks * is_next

    # deltas and gates
    # mask next values first so NaNs at padded timesteps cannot leak in through `0. * nan`

    if is_tensor(masks):
        next_values = next_values.masked_fill(masks == 0., 0.)

    delta = rewards + gamma * next_values * masks - values

    if exists(len_mask):
        delta = delta.masked_fill(~len_mask, 0.)

    gates = as_tensor(gamma * lam * done_masks, device = device, dtype = dtype).expand_as(delta)

    # scan

    gae = reverse_assoc_scan(gates, delta, use_accelerated)
    returns = gae + values

    if exists(len_mask):
        gae = gae.masked_fill(~len_mask, 0.)
        returns = returns.masked_fill(~len_mask, 0.)

    if grouped:
        returns = returns.movedim(-2, group_dim)

    remove_batch_dim = lambda t: t[0] if is_single_sequence else t

    # early return if advantages are not needed

    if not return_advantages:
        return remove_batch_dim(returns)

    # combine grouped advantages, if weights are given

    if exists(weights):
        gae = combine_grouped_advantages(gae, weights = weights, normalize = normalize_grouped, group_dim = -2, mask = done_masks)
    elif grouped:
        gae = gae.movedim(-2, group_dim)

    returns, gae = tree_map_tensor(remove_batch_dim, (returns, gae))

    return GAE(returns, gae)

# returns

def calc_returns(
    rewards: Tensor,                          # (b n)
    masks: Tensor | None = None,              # (b n)
    gamma: float | Tensor = 0.99,
    next_value: float | Tensor | None = None, # () or (b)
    lens: Tensor | None = None,               # (b)
    use_accelerated: bool | None = None
) -> Tensor:                                  # (b n)
    # handle single sequence input, with no batch dimension

    is_single_sequence = rewards.ndim == 1

    if is_single_sequence:
        rewards, masks, gamma = tree_map_tensor(
            lambda t: t[None] if t.ndim == 1 else t,
            (rewards, masks, gamma)
        )

    device, dtype, seq_len = rewards.device, rewards.dtype, rewards.shape[-1]

    masks = default(masks, 1.)
    masks, gamma, lens = tree_map_tensor_to_device((masks, gamma, lens), device)
    raw_masks = masks

    # variable length masking

    len_mask = None
    if exists(lens):
        len_mask = lens_to_mask(lens, max_len = seq_len)
        masks = masks * pad_right_at_dim(len_mask[..., 1:], 1, value = False)

    # bootstrap next value at each sequence's last valid timestep

    if exists(next_value) and not exists(lens):
        next_val = as_tensor(next_value, device = device, dtype = dtype)
        next_mask = raw_masks[..., -1] if is_tensor(raw_masks) else raw_masks
        rewards = rewards.clone()
        rewards[..., -1] = rewards[..., -1] + gamma * next_val * next_mask

    elif exists(next_value) and exists(lens):
        next_val = as_tensor(next_value, device = device, dtype = dtype)
        last_idx = (lens - 1).clamp(min = 0, max = seq_len - 1)
        last_mask = batched_index_select(raw_masks, last_idx) if is_tensor(raw_masks) else raw_masks
        last_pos = lens_to_last_mask(lens, seq_len).type_as(rewards)

        rewards = rewards.clone() + pad_right_ndim_to(gamma * next_val * last_mask, 2) * last_pos

    if exists(len_mask):
        rewards = rewards.masked_fill(~len_mask, 0.)

    # scan

    gates = as_tensor(gamma * masks, device = device, dtype = dtype).expand_as(rewards)
    returns = reverse_assoc_scan(gates, rewards, use_accelerated)

    if exists(len_mask):
        returns = returns.masked_fill(~len_mask, 0.)

    if is_single_sequence:
        return returns[0]

    return returns

# value clipping functions

# classic clipped value loss - Schulman et al.

def clipped_value_loss_fn(
    values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip: float = 0.2,
    loss_fn: Callable = F.mse_loss,
    mask: Tensor | None = None,
    lens: Tensor | None = None,
    reduction: str = 'mean'
) -> Tensor:
    if exists(lens):
        mask = default(mask, lens_to_mask(lens, max_len = values.shape[-1]))

    clipped_values = old_values + (values - old_values).clamp(-clip, clip)

    loss = loss_fn(values, returns, reduction = 'none')
    clipped_loss = loss_fn(clipped_values, returns, reduction = 'none')

    value_loss = torch.max(loss, clipped_loss)

    return masked_reduce(value_loss, mode = reduction, mask = mask)

# improved value clipping - https://www.authorea.com/users/855021/articles/1240083-on-analysis-of-clipped-critic-loss-in-proximal-policy-gradient

def improved_value_clipping_fn(
    values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip: float = 0.2,
    loss_fn: Callable = F.mse_loss,
    scalar_values: Tensor | None = None,
    scalar_old_values: Tensor | None = None,
    mask: Tensor | None = None,
    lens: Tensor | None = None,
    reduction: str = 'mean'
) -> Tensor:
    if exists(lens):
        mask = default(mask, lens_to_mask(lens, max_len = values.shape[-1]))

    scalar_values = default(scalar_values, values)
    scalar_old_values = default(scalar_old_values, old_values)

    clipped_returns = returns.clamp(scalar_old_values - clip, scalar_old_values + clip)

    loss = loss_fn(values, returns, reduction = 'none')
    clipped_loss = loss_fn(values, clipped_returns, reduction = 'none')

    old_values_lo = scalar_old_values - clip
    old_values_hi = scalar_old_values + clip

    def is_between(mid, lo, hi):
        return (lo < mid) & (mid < hi)

    value_loss = torch.where(
        is_between(scalar_values, returns, old_values_lo) |
        is_between(scalar_values, old_values_hi, returns),
        0.,
        torch.min(loss, clipped_loss)
    )

    return masked_reduce(value_loss, mode = reduction, mask = mask)
