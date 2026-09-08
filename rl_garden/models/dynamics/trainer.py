"""GPU-native training loop + OPE rollout scorer for `EnsembleDynamicsModel`.

The actual rewrite work vs. upstream: `3rd_party/Uni-O4/transition_model/
dynamics/ensemble_dynamics.py`'s `EnsembleDynamics.step()`/`train()`/
`validate()` round-trip every call through `.cpu().numpy()`; this module is
pure-tensor on whatever device the model/data already live on, per AGENTS.md's
"avoid NumPy handoffs in hot paths."

Two upstream mechanics preserved deliberately, not simplified:
- **Bootstrap-with-replacement per ensemble member** (`torch.randint`, not a
  permutation) -- upstream's `data_idxes = np.random.randint(train_size,
  size=[num_ensemble, train_size])`, reshuffled (column order only, not
  redrawn) each epoch via `shuffle_rows`. This is bagging-style diversity
  across members, not an implementation detail to drop.
- **Stochastic OPE rollout** -- `rollout_q_mean` samples `mean + randn()*std`
  then picks one random elite member per row, matching the ensemble's own
  aleatoric-uncertainty design (`3rd_party/Uni-O4/dynamics_eval.py::rollout`).
"""
from __future__ import annotations

from typing import Optional

import torch

from rl_garden.models.dynamics.network import EnsembleDynamicsModel
from rl_garden.models.dynamics.termination_fns import TerminationFn


def _shuffle_columns(
    idxes: torch.Tensor, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Permute each row's column order independently -- torch port of
    upstream's `shuffle_rows` (re-orders an existing bootstrap sample, does
    not redraw it)."""
    order = torch.argsort(
        torch.rand(idxes.shape, device=idxes.device, generator=generator), dim=-1
    )
    return torch.gather(idxes, 1, order)


def train_ensemble(
    model: EnsembleDynamicsModel,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
    *,
    max_epochs_since_update: int = 5,
    max_epochs: Optional[int] = None,
    batch_size: int = 256,
    holdout_ratio: float = 0.2,
    logvar_loss_coef: float = 0.01,
    generator: Optional[torch.Generator] = None,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Trains `model` in place (early-stopped, elite-selected, loaded back
    to each member's best-holdout snapshot). Returns `(metrics, input_mean,
    input_std)` -- the fitted input normalization, needed again at rollout
    time (`rollout_q_mean`)."""
    device = obs.device
    inputs = torch.cat([obs, actions], dim=-1)
    targets = torch.cat([next_obs - obs, rewards.unsqueeze(-1)], dim=-1)

    data_size = inputs.shape[0]
    holdout_size = min(int(data_size * holdout_ratio), 1000)
    train_size = data_size - holdout_size
    perm = torch.randperm(data_size, device=device, generator=generator)
    train_inputs, train_targets = inputs[perm[:train_size]], targets[perm[:train_size]]
    holdout_inputs = inputs[perm[train_size:]]
    holdout_targets = targets[perm[train_size:]]

    input_mean = train_inputs.mean(dim=0, keepdim=True)
    input_std = train_inputs.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_inputs = (train_inputs - input_mean) / input_std
    holdout_inputs = (holdout_inputs - input_mean) / input_std

    num_ensemble = model.num_ensemble
    data_idxes = torch.randint(
        0, train_size, (num_ensemble, train_size), device=device, generator=generator
    )

    holdout_losses = torch.full((num_ensemble,), 1e10, device=device)
    epoch = 0
    cnt = 0
    while True:
        epoch += 1
        model.train()
        num_batches = max(1, (train_size + batch_size - 1) // batch_size)
        for b in range(num_batches):
            batch_idx = data_idxes[:, b * batch_size : (b + 1) * batch_size]
            inputs_batch = train_inputs[batch_idx]
            targets_batch = train_targets[batch_idx]

            mean, logvar = model(inputs_batch)
            inv_var = torch.exp(-logvar)
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + model.get_decay_loss()
            loss = (
                loss
                + logvar_loss_coef * model.max_logvar.sum()
                - logvar_loss_coef * model.min_logvar.sum()
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        data_idxes = _shuffle_columns(data_idxes, generator=generator)

        model.eval()
        with torch.no_grad():
            mean, _ = model(holdout_inputs)
            new_holdout_losses = ((mean - holdout_targets) ** 2).mean(dim=(1, 2))

        indexes = []
        for m in range(num_ensemble):
            improvement = (holdout_losses[m] - new_holdout_losses[m]) / holdout_losses[m]
            if improvement > 0.01:
                indexes.append(m)
                holdout_losses[m] = new_holdout_losses[m]
        if indexes:
            model.update_save(indexes)
            cnt = 0
        else:
            cnt += 1
        if cnt >= max_epochs_since_update or (max_epochs is not None and epoch >= max_epochs):
            break

    elites = torch.argsort(holdout_losses)[: model.num_elites].tolist()
    model.set_elites(elites)
    model.load_save()
    model.eval()

    metrics = {
        "dynamics_epochs": float(epoch),
        "dynamics_holdout_loss": float(holdout_losses[elites].mean().item()),
    }
    return metrics, input_mean, input_std


def rollout_q_mean(
    policy,
    q_net,
    model: EnsembleDynamicsModel,
    termination_fn: TerminationFn,
    init_obs: torch.Tensor,
    rollout_length: int,
    *,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> float:
    """OPE score for `policy`: mean `Q(s, policy(s))` along a stochastic
    dynamics-model rollout from `init_obs`, early-terminated once every row
    has terminated. Port of `3rd_party/Uni-O4/dynamics_eval.py::rollout`
    (mean-Q scoring, not discounted-return estimation -- upstream's own
    metric).

    No `penalty_coef`/uncertainty-penalty parameter: upstream's own
    `rollout()` only applies it to the *reward* accumulator, which its own
    caller (`dynamics_eval()`) discards (`best_mean_q, _ = dynamics_eval(...)`)
    -- the gating score is `Q_mean` alone, never touched by the penalty.
    Confirmed by reading both call sites directly; there is nothing for a
    penalty coefficient to do here."""
    obs = init_obs
    q_values: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(rollout_length):
            action = policy.predict(obs, deterministic=True)
            q_values.append(q_net(obs, action).view(-1))

            obs_action = torch.cat([obs, action], dim=-1)
            obs_action_norm = (obs_action - input_mean) / input_std
            mean, logvar = model(obs_action_norm)
            std = torch.exp(0.5 * logvar)
            samples = mean + torch.randn(
                mean.shape, device=mean.device, generator=generator
            ) * std

            row_idx = torch.arange(obs.shape[0], device=obs.device)
            member_idx = model.random_elite_member(obs.shape[0])
            chosen = samples[member_idx, row_idx]
            delta_obs = chosen[:, :-1]
            next_obs = obs + delta_obs

            terminal = termination_fn(obs, action, next_obs).view(-1)
            nonterm_mask = ~terminal
            if nonterm_mask.sum() == 0:
                break
            obs = next_obs[nonterm_mask]

    return float(torch.cat(q_values).mean().item())
