"""SAC family under ``encoder_sharing="shared"``: the actor loss trains the
shared trunk (extractor, plus the sequence trunk of sequence SAC) through
``actor_optimizer`` while the Q-loss keeps training it through
``q_optimizer``."""

from __future__ import annotations

import pytest
import torch
from test_sac_optimizer_snapshot import (
    ENCODER_CONFIG,
    DictObsVecEnv,
    build_dict_sac,
    dict_sac_digests,
    run_single_threaded,
)

from rl_garden.algorithms import SAC, RecurrentSAC


def _clone_parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _any_parameter_changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(old, new) for old, new in zip(before, module.parameters(), strict=True)
    )


def test_shared_diverges_from_shared_critic_grad_only_in_the_extractor_after_one_update():
    shared = dict_sac_digests("shared")
    critic_only = dict_sac_digests("shared_critic_grad")

    changed_after_one = {name for name in critic_only[1] if shared[1][name] != critic_only[1][name]}
    assert changed_after_one == {"actor_extractor"}
    assert shared[3] != critic_only[3]


@pytest.mark.parametrize(
    ("encoder_sharing", "extractor_moves"), [("shared", True), ("shared_critic_grad", False)]
)
def test_actor_update_alone_moves_the_extractor_only_under_shared(encoder_sharing, extractor_moves):
    def run():
        agent = build_dict_sac(encoder_sharing)
        policy = agent.policy
        extractor_before = _clone_parameters(policy.actor_extractor)
        critic_before = _clone_parameters(policy.critic)

        actor_loss, _ = agent._actor_loss_from_batch(agent._sample_train_batch(agent.batch_size))
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        assert _any_parameter_changed(extractor_before, policy.actor_extractor) is extractor_moves
        assert not _any_parameter_changed(critic_before, policy.critic)

    run_single_threaded(run)


def _train_one_update_from(agent: SAC, rng_state: torch.Tensor) -> dict[str, torch.Tensor]:
    torch.set_rng_state(rng_state)
    agent.train(gradient_steps=1)
    return {key: value.clone() for key, value in agent.policy.state_dict().items()}


def test_shared_checkpoint_resume_with_optimizers_matches_uninterrupted_training(tmp_path):
    def run():
        uninterrupted = build_dict_sac("shared")
        uninterrupted.train(gradient_steps=1)
        path = uninterrupted.save(tmp_path / "shared.pt", include_replay_buffer=True)

        extractor_params = list(uninterrupted.policy.actor_extractor.parameters())
        for param in extractor_params:
            critic_moment = uninterrupted.q_optimizer.state[param]["exp_avg"]
            actor_moment = uninterrupted.actor_optimizer.state[param]["exp_avg"]
            assert not torch.equal(critic_moment, actor_moment)

        rng_state = torch.get_rng_state()
        expected = _train_one_update_from(uninterrupted, rng_state)

        resumed = build_dict_sac("shared")
        resumed.load(path)
        resumed_state = _train_one_update_from(resumed, rng_state)

        without_optimizers = build_dict_sac("shared")
        without_optimizers.load(path, load_optimizers=False)
        fresh_optimizer_state = _train_one_update_from(without_optimizers, rng_state)
        return expected, resumed_state, fresh_optimizer_state

    expected, resumed, fresh_optimizer_state = run_single_threaded(run)
    assert expected.keys() == resumed.keys()
    for key, value in expected.items():
        assert torch.equal(value, resumed[key]), key
    assert any(not torch.equal(value, fresh_optimizer_state[key]) for key, value in expected.items())


@pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
def test_recurrent_sac_actor_update_alone_moves_the_rnn_under_shared(rnn_type):
    agent = RecurrentSAC(
        env=DictObsVecEnv(),
        encoder_sharing="shared",
        rnn_type=rnn_type,
        rnn_hidden_size=8,
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=4,
        learning_starts=10,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        std_log=False,
        net_arch=[16],
        burn_in_len=2,
        learning_len=2,
        forward_len=1,
        encoder_config=ENCODER_CONFIG,
    )
    agent.learn(total_timesteps=12)
    policy = agent.policy
    trunk_before = {
        "extractor": _clone_parameters(policy.actor_extractor),
        "rnn": _clone_parameters(policy.recurrent_encoder),
    }
    critic_before = _clone_parameters(policy.critic)

    actor_loss, _ = agent._actor_loss_from_batch(agent._sample_train_batch(agent.batch_size))
    agent.actor_optimizer.zero_grad()
    actor_loss.backward()
    agent.actor_optimizer.step()

    assert _any_parameter_changed(trunk_before["extractor"], policy.actor_extractor)
    assert _any_parameter_changed(trunk_before["rnn"], policy.recurrent_encoder)
    assert not _any_parameter_changed(critic_before, policy.critic)
