"""DDPG / TD3 / TD3+BC family under ``encoder_sharing="shared"``: the actor loss
trains the shared extractor through ``actor_optimizer`` while the Q-loss keeps
training it through the critic optimizer.

The family has no DDP path, so there is no DDP variant of these checks.
"""

from __future__ import annotations

import pytest
import torch
from test_ddpg_optimizer_snapshot import (
    FAMILY_CASES,
    SNAPSHOT,
    family_digests,
    run_single_threaded,
)
from test_sac_optimizer_snapshot import requires_snapshot_build

#: The critic optimizer's attribute name: DDPG and TD3 call it ``q_optimizer``.
CRITIC_OPTIMIZER = {"ddpg": "q_optimizer", "td3": "q_optimizer"}


def critic_optimizer(case: str, agent) -> torch.optim.Optimizer:
    return getattr(agent, CRITIC_OPTIMIZER.get(case, "critic_optimizer"))


def _clone_parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _any_parameter_changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(old, new) for old, new in zip(before, module.parameters(), strict=True)
    )


@requires_snapshot_build
@pytest.mark.parametrize("case", sorted(FAMILY_CASES))
def test_shared_diverges_from_its_recorded_run_only_in_the_extractor_after_one_actor_update(case):
    # DDPG, TD3 and TD3BC reuse the detached pre-step critic features for the
    # actor under "shared_critic_grad" but re-extract them under "shared", so
    # their actors already differed between the two modes before the fix; the
    # reference is therefore the "shared" run recorded before the fix.
    recorded = SNAPSHOT[case]["shared"]
    shared = family_digests(case, "shared")

    changed_after_first = {
        name for name in recorded["first"] if shared["first"][name] != recorded["first"][name]
    }
    assert changed_after_first == {"actor_extractor"}
    assert shared["second"] != recorded["second"]


@pytest.mark.parametrize(
    ("encoder_sharing", "extractor_moves"), [("shared", True), ("shared_critic_grad", False)]
)
@pytest.mark.parametrize("case", sorted(FAMILY_CASES))
def test_actor_update_alone_moves_the_extractor_only_under_shared(
    case, encoder_sharing, extractor_moves
):
    builder, first_steps = FAMILY_CASES[case]
    agent = builder(encoder_sharing)
    # A zero learning rate leaves every parameter the critic optimizer holds
    # exactly where it is, so only the actor optimizer moves anything.
    for group in critic_optimizer(case, agent).param_groups:
        group["lr"] = 0.0
    policy = agent.policy
    extractor_before = _clone_parameters(policy.actor_extractor)
    critic_before = _clone_parameters(policy.critic)
    actor_before = _clone_parameters(policy.actor)

    agent.train(first_steps)

    assert _any_parameter_changed(actor_before, policy.actor)
    assert _any_parameter_changed(extractor_before, policy.actor_extractor) is extractor_moves
    assert not _any_parameter_changed(critic_before, policy.critic)


def _train_from(agent, rng_state: torch.Tensor, gradient_steps: int) -> dict[str, torch.Tensor]:
    torch.set_rng_state(rng_state)
    agent.train(gradient_steps)
    return {key: value.clone() for key, value in agent.policy.state_dict().items()}


@pytest.mark.parametrize("case", sorted(FAMILY_CASES))
def test_shared_checkpoint_resume_with_optimizers_matches_uninterrupted_training(case, tmp_path):
    builder, first_steps = FAMILY_CASES[case]

    def run():
        uninterrupted = builder("shared")
        uninterrupted.train(first_steps)
        path = uninterrupted.save(tmp_path / f"{case}.pt", include_replay_buffer=True)

        critic_state = critic_optimizer(case, uninterrupted).state
        actor_state = uninterrupted.actor_optimizer.state
        for param in uninterrupted.policy.actor_extractor.parameters():
            assert not torch.equal(critic_state[param]["exp_avg"], actor_state[param]["exp_avg"])

        rng_state = torch.get_rng_state()
        expected = _train_from(uninterrupted, rng_state, first_steps)

        resumed = builder("shared")
        resumed.load(path)
        resumed_state = _train_from(resumed, rng_state, first_steps)

        without_optimizers = builder("shared")
        without_optimizers.load(path, load_optimizers=False)
        fresh_optimizer_state = _train_from(without_optimizers, rng_state, first_steps)
        return expected, resumed_state, fresh_optimizer_state

    expected, resumed, fresh_optimizer_state = run_single_threaded(run)
    assert expected.keys() == resumed.keys()
    for key, value in expected.items():
        assert torch.equal(value, resumed[key]), key
    assert any(not torch.equal(value, fresh_optimizer_state[key]) for key, value in expected.items())
