import pytest
import torch
from torch import nn, tensor
from torch.distributions import Categorical
from torch_einops_utils import lens_to_mask

from x_ppo import (
    ppo_actor_loss,
    spo_actor_loss,
    calc_gae,
    combine_grouped_advantages,
    clipped_value_loss_fn,
    improved_value_clipping_fn
)

param = pytest.mark.parametrize

# end to end - synthetic rollout through gae, actor and critic losses, with optimizer steps

@param('actor_loss_fn', (ppo_actor_loss, spo_actor_loss))
@param('critic_loss_fn', (clipped_value_loss_fn, improved_value_clipping_fn))
def test_e2e_single_critic_update(actor_loss_fn, critic_loss_fn):
    torch.manual_seed(0)
    b, n, state_dim, num_actions = 4, 6, 4, 3

    states = torch.randn(b, n, state_dim)
    actions = torch.randint(0, num_actions, (b, n))
    rewards = torch.randn(b, n)
    lens = tensor([4, 6, 5, 6])
    mask = lens_to_mask(lens, n)
    per_sample_gammas = tensor([0.99, 0.95, 0.9, 0.99])

    actor = nn.Sequential(nn.Linear(state_dim, 16), nn.Tanh(), nn.Linear(16, num_actions))
    critic = nn.Sequential(nn.Linear(state_dim, 16), nn.Tanh(), nn.Linear(16, 1))
    opt_actor = torch.optim.Adam(actor.parameters(), lr = 1e-2)
    opt_critic = torch.optim.Adam(critic.parameters(), lr = 1e-2)

    with torch.no_grad():
        old_action_log_probs = Categorical(logits = actor(states)).log_prob(actions)
        old_values = critic(states).squeeze(-1)

    returns, advantages = calc_gae(
        rewards, old_values,
        lens = lens,
        gamma = per_sample_gammas,
        use_accelerated = False,
        return_advantages = True
    )

    actor_losses, critic_losses = [], []

    for _ in range(30):
        action_log_probs = Categorical(logits = actor(states)).log_prob(actions)
        actor_loss = actor_loss_fn(action_log_probs, old_action_log_probs, advantages, mask = mask)

        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()

        values = critic(states).squeeze(-1)
        critic_loss = critic_loss_fn(values, old_values, returns, lens = lens)

        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

    # both losses improve on the fixed batch

    assert actor_losses[-1] < actor_losses[0]
    assert critic_losses[-1] < critic_losses[0]

def test_e2e_multi_critic_update():
    torch.manual_seed(0)
    b, g, n, state_dim, num_actions = 4, 3, 6, 4, 3

    # reward groups on wildly different scales - per group advantage normalization should absorb it

    states = torch.randn(b, n, state_dim)
    actions = torch.randint(0, num_actions, (b, n))
    rewards = torch.randn(b, g, n) * tensor([1., 10., 100.])[:, None]
    masks = torch.ones(b, n)
    masks[:, -1] = 0.
    weights = tensor([1., 0.5, 0.25])

    actor = nn.Sequential(nn.Linear(state_dim, 16), nn.Tanh(), nn.Linear(16, num_actions))
    critic = nn.Sequential(nn.Linear(state_dim, 16), nn.Tanh(), nn.Linear(16, g))
    opt_actor = torch.optim.Adam(actor.parameters(), lr = 1e-2)
    opt_critic = torch.optim.Adam(critic.parameters(), lr = 1e-2)

    with torch.no_grad():
        old_action_log_probs = Categorical(logits = actor(states)).log_prob(actions)
        old_values = critic(states)
        initial_value_head = critic[-1].weight.clone()

    # grouped gae - one value head per reward group - normalized groups combined for the actor

    gae = calc_gae(rewards, old_values.movedim(-1, 1), masks = masks, group_dim = 1, use_accelerated = False, return_advantages = True)

    assert gae.returns.shape == gae.advantages.shape == (b, g, n)

    advantages = combine_grouped_advantages(gae.advantages, weights = weights, group_dim = 1, mask = masks)

    assert advantages.shape == (b, n)
    assert torch.isfinite(advantages).all()

    returns = gae.returns.movedim(1, -1)

    actor_losses, critic_losses = [], []

    for _ in range(30):
        action_log_probs = Categorical(logits = actor(states)).log_prob(actions)
        actor_loss = ppo_actor_loss(action_log_probs, old_action_log_probs, advantages, mask = masks)

        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()

        values = critic(states)
        critic_loss = clipped_value_loss_fn(values, old_values, returns, mask = masks, group_dim = -1)

        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

    assert actor_losses[-1] < actor_losses[0]
    assert critic_losses[-1] < critic_losses[0]

    # every value head was updated, regardless of reward scale

    assert (critic[-1].weight != initial_value_head).any(dim = -1).all()
