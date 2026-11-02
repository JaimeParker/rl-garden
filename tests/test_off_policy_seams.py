"""Off-policy loop and replay seams of the extension contract
(``docs/guides/extending.md``, "Off-policy loop and replay seams"): the rollout
hooks of ``OffPolicyAlgorithm``, the evaluation hooks of ``BaseAlgorithm``, the
sampling and construction hooks of the SAC family and the prior-data buffer
hooks. Each test subclasses the algorithm, overrides the hook, drives the
smallest loop that reaches it and checks the arguments the hook receives."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from gymnasium import spaces
from test_optimizer_ownership import ACTION_DIM, COMMON_KWARGS, STATE_DIM, TinyVecEnv
from test_sac_optimizer_snapshot import fill_random_transitions

from rl_garden.algorithms import RLPD, SAC
from rl_garden.buffers.demo_intervention import DemoInterventionMixin
from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.common.types import ReplayBufferSample

NUM_ENVS = TinyVecEnv.num_envs
#: Two rollout steps of the two envs per ``learn()`` iteration.
ONE_ITERATION_KWARGS = {**COMMON_KWARGS, "training_freq": 4}
ONE_ITERATION_STEPS = 4
ROLLOUT_STEPS = ONE_ITERATION_STEPS // NUM_ENVS


class RecordingEnv(TinyVecEnv):
    """``TinyVecEnv`` that keeps the actions it was stepped with."""

    def __init__(self) -> None:
        super().__init__()
        self.received_actions: list[torch.Tensor] = []

    def step(self, actions):
        self.received_actions.append(actions.clone())
        return super().step(actions)


def assert_state_observation(obs: Any) -> None:
    assert set(obs) == {"state"}
    assert obs["state"].shape == (NUM_ENVS, STATE_DIM)


# --- rollout hooks ----------------------------------------------------------------------------


class RolloutRecordingSAC(SAC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.events: list[str] = []
        self.reset_calls: list[Any] = []
        self.rollout_calls: list[dict[str, Any]] = []
        self.add_kwargs_calls: list[dict[str, Any]] = []
        self.step_kwargs_calls: list[dict[str, Any]] = []
        self.post_step_calls: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _on_env_reset(self, obs) -> None:
        self.events.append("on_env_reset")
        self.reset_calls.append(obs)
        super()._on_env_reset(obs)

    def _rollout_action(self, obs, learning_has_started: bool):
        self.events.append("rollout_action")
        replay_action, _, _ = super()._rollout_action(obs, learning_has_started)
        env_action = 0.5 * replay_action
        context = {"step": len(self.rollout_calls)}
        self.rollout_calls.append(
            {
                "obs": obs,
                "learning_has_started": learning_has_started,
                "replay_action": replay_action,
                "env_action": env_action,
                "context": context,
            }
        )
        return replay_action, env_action, context

    def _replay_buffer_add_kwargs(
        self, action_context, obs, next_obs, real_next_obs, infos, need_final_obs
    ) -> dict[str, Any]:
        self.events.append("add_kwargs")
        self.add_kwargs_calls.append(
            {
                "action_context": action_context,
                "obs": obs,
                "next_obs": next_obs,
                "real_next_obs": real_next_obs,
                "infos": infos,
                "need_final_obs": need_final_obs,
            }
        )
        return {"from_add_kwargs": 1, "in_both": "add_kwargs"}

    def _replay_buffer_step_kwargs(self, terminations, truncations) -> dict[str, Any]:
        self.events.append("step_kwargs")
        self.step_kwargs_calls.append({"terminations": terminations, "truncations": truncations})
        return {"from_step_kwargs": 2, "in_both": "step_kwargs"}

    def _post_rollout_step(self, action_context, terminations, truncations, infos) -> None:
        self.events.append("post_rollout_step")
        self.post_step_calls.append(
            {
                "action_context": action_context,
                "terminations": terminations,
                "truncations": truncations,
                "infos": infos,
            }
        )
        super()._post_rollout_step(action_context, terminations, truncations, infos)


def test_rollout_hooks_run_in_loop_order_with_documented_arguments():
    env = RecordingEnv()
    agent = RolloutRecordingSAC(env=env, **ONE_ITERATION_KWARGS)
    written: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    buffer_add = agent.replay_buffer.add

    def recording_add(*args: Any, **kwargs: Any) -> None:
        agent.events.append("add")
        written.append((args, kwargs))
        # The plain ReplayBuffer takes no extra keywords.
        buffer_add(*args)

    agent.replay_buffer.add = recording_add

    agent.learn(total_timesteps=ONE_ITERATION_STEPS)

    step_events = ["rollout_action", "add_kwargs", "step_kwargs", "add", "post_rollout_step"]
    assert agent.events == ["on_env_reset"] + step_events * ROLLOUT_STEPS

    assert len(agent.reset_calls) == 1
    assert_state_observation(agent.reset_calls[0])

    for step in range(ROLLOUT_STEPS):
        rollout = agent.rollout_calls[step]
        add_kwargs = agent.add_kwargs_calls[step]
        step_kwargs = agent.step_kwargs_calls[step]
        post_step = agent.post_step_calls[step]
        args, kwargs = written[step]

        assert_state_observation(rollout["obs"])
        assert rollout["learning_has_started"] is True
        assert rollout["replay_action"].shape == (NUM_ENVS, ACTION_DIM)

        # replay_action is stored, env_action is executed.
        assert torch.equal(env.received_actions[step], rollout["env_action"])
        assert args[2] is rollout["replay_action"]

        # The action context travels to the replay-write and post-step hooks.
        assert add_kwargs["action_context"] is rollout["context"]
        assert post_step["action_context"] is rollout["context"]

        assert_state_observation(add_kwargs["obs"])
        assert_state_observation(add_kwargs["next_obs"])
        assert_state_observation(add_kwargs["real_next_obs"])
        assert add_kwargs["infos"] == {}
        assert add_kwargs["need_final_obs"].shape == (NUM_ENVS,)
        assert add_kwargs["need_final_obs"].dtype == torch.bool

        for flags in (
            step_kwargs["terminations"],
            step_kwargs["truncations"],
            post_step["terminations"],
            post_step["truncations"],
        ):
            assert flags.shape == (NUM_ENVS,)
            assert flags.dtype == torch.bool
        assert post_step["infos"] == {}

        # Both dicts reach the buffer's add; the step keywords win a clash.
        assert kwargs == {"from_add_kwargs": 1, "from_step_kwargs": 2, "in_both": "step_kwargs"}


class ExploreRecordingSAC(SAC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.explore_obs: list[Any] = []
        super().__init__(*args, **kwargs)

    def _explore_action(self, obs) -> torch.Tensor:
        self.explore_obs.append(obs)
        return torch.full((NUM_ENVS, ACTION_DIM), 0.25)


def test_explore_action_chooses_the_actions_before_learning_starts():
    env = RecordingEnv()
    agent = ExploreRecordingSAC(env=env, **{**ONE_ITERATION_KWARGS, "learning_starts": 100})

    agent.learn(total_timesteps=ONE_ITERATION_STEPS)

    assert len(agent.explore_obs) == ROLLOUT_STEPS
    for obs in agent.explore_obs:
        assert_state_observation(obs)
    assert len(env.received_actions) == ROLLOUT_STEPS
    for action in env.received_actions:
        assert torch.equal(action, torch.full((NUM_ENVS, ACTION_DIM), 0.25))


def test_explore_action_default_is_uniform_over_the_action_box():
    agent = SAC(env=TinyVecEnv(), **COMMON_KWARGS)

    action = agent._explore_action(agent.env.reset()[0])

    assert action.shape == agent.env.action_space.shape
    assert action.min() >= -1.0 and action.max() <= 1.0


# --- evaluation hooks -------------------------------------------------------------------------


class EvalRecordingSAC(SAC):
    """Records the evaluation hooks; ``_eval_action`` returns a constant."""

    eval_action_value = 0.5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.events: list[str] = []
        self.action_obs: list[Any] = []
        self.step_calls: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _eval_start_hook(self) -> None:
        self.events.append("start")
        super()._eval_start_hook()

    def _eval_action(self, obs) -> torch.Tensor:
        self.events.append("action")
        self.action_obs.append(obs)
        return torch.full((NUM_ENVS, ACTION_DIM), self.eval_action_value)

    def _eval_step_hook(
        self, obs_before, critic_action, rewards, terminations, truncations, infos
    ) -> None:
        self.events.append("step")
        self.step_calls.append(
            {
                "obs_before": obs_before,
                "critic_action": critic_action,
                "rewards": rewards,
                "terminations": terminations,
                "truncations": truncations,
                "infos": infos,
            }
        )
        super()._eval_step_hook(obs_before, critic_action, rewards, terminations, truncations, infos)


class SplitEvalActionSAC(EvalRecordingSAC):
    """Executes one action and has the critic score another."""

    def _eval_action_and_critic_action(self, obs):
        self.events.append("action")
        self.action_obs.append(obs)
        return (
            torch.full((NUM_ENVS, ACTION_DIM), 0.5),
            torch.full((NUM_ENVS, ACTION_DIM), 0.25),
        )


NUM_EVAL_STEPS = 3


def assert_eval_loop(agent: EvalRecordingSAC, eval_env: RecordingEnv) -> None:
    assert agent.events == ["start"] + ["action", "step"] * NUM_EVAL_STEPS
    assert len(agent.action_obs) == NUM_EVAL_STEPS
    assert len(eval_env.received_actions) == NUM_EVAL_STEPS
    for obs in agent.action_obs:
        assert_state_observation(obs)
    for call in agent.step_calls:
        assert_state_observation(call["obs_before"])
        assert call["rewards"].shape == (NUM_ENVS,)
        assert call["terminations"].shape == (NUM_ENVS,)
        assert call["truncations"].shape == (NUM_ENVS,)
        assert call["infos"] == {}


def test_eval_action_feeds_the_environment_and_the_critic_action():
    eval_env = RecordingEnv()
    agent = EvalRecordingSAC(
        env=TinyVecEnv(), eval_env=eval_env, num_eval_steps=NUM_EVAL_STEPS, **COMMON_KWARGS
    )

    agent._evaluate()

    assert_eval_loop(agent, eval_env)
    expected = torch.full((NUM_ENVS, ACTION_DIM), 0.5)
    for executed in eval_env.received_actions:
        assert torch.equal(executed, expected)
    for call in agent.step_calls:
        assert torch.equal(call["critic_action"], expected)


def test_eval_action_and_critic_action_split_the_executed_and_the_scored_action():
    eval_env = RecordingEnv()
    agent = SplitEvalActionSAC(
        env=TinyVecEnv(), eval_env=eval_env, num_eval_steps=NUM_EVAL_STEPS, **COMMON_KWARGS
    )

    agent._evaluate()

    assert_eval_loop(agent, eval_env)
    for executed in eval_env.received_actions:
        assert torch.equal(executed, torch.full((NUM_ENVS, ACTION_DIM), 0.5))
    for call in agent.step_calls:
        assert torch.equal(call["critic_action"], torch.full((NUM_ENVS, ACTION_DIM), 0.25))


# --- construction hooks -----------------------------------------------------------------------


class MarkedReplayBuffer(ReplayBuffer):
    """Distinguishable ``ReplayBuffer``."""


class ConstructionRecordingSAC(SAC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.construction_calls: list[str] = []
        super().__init__(*args, **kwargs)

    def _policy_action_space(self) -> spaces.Box:
        self.construction_calls.append("policy_action_space")
        return super()._policy_action_space()

    def _build_replay_buffer(self) -> MarkedReplayBuffer:
        self.construction_calls.append("replay_buffer")
        return MarkedReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
            mmap_dir=self.mmap_dir,
            mmap_mode=self.mmap_mode,
        )


def test_construction_hooks_build_the_policy_space_and_the_replay_buffer_once():
    agent = ConstructionRecordingSAC(env=TinyVecEnv(), **ONE_ITERATION_KWARGS)

    assert agent.construction_calls.count("replay_buffer") == 1
    assert "policy_action_space" in agent.construction_calls
    assert type(agent.replay_buffer) is MarkedReplayBuffer

    agent.learn(total_timesteps=ONE_ITERATION_STEPS)

    # The rollout wrote into, and training sampled from, the subclass's buffer.
    assert agent.replay_buffer.pos == ROLLOUT_STEPS


# --- sampling and update hooks ----------------------------------------------------------------


class TrainRecordingSAC(SAC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sample_calls: list[tuple[int, Any]] = []
        self.target_calls: list[tuple[Any, Any]] = []
        self.actor_calls: list[tuple[Any, Any]] = []
        super().__init__(*args, **kwargs)

    def _sample_train_batch(self, batch_size: int):
        data = super()._sample_train_batch(batch_size)
        self.sample_calls.append((batch_size, data))
        return data

    def _target_action_log_prob(self, data):
        result = super()._target_action_log_prob(data)
        self.target_calls.append((data, result))
        return result

    def _actor_loss_from_batch(self, data):
        result = super()._actor_loss_from_batch(data)
        self.actor_calls.append((data, result))
        return result


def test_train_samples_a_batch_per_gradient_step_and_routes_it_through_the_loss_hooks():
    agent = TrainRecordingSAC(env=TinyVecEnv(), **COMMON_KWARGS)
    fill_random_transitions(agent)
    gradient_steps = 2

    agent.train(gradient_steps=gradient_steps)

    batch_size = COMMON_KWARGS["batch_size"]
    assert [size for size, _ in agent.sample_calls] == [batch_size] * gradient_steps
    sampled = [data for _, data in agent.sample_calls]

    assert len(agent.target_calls) == gradient_steps
    assert all(data is batch for (data, _), batch in zip(agent.target_calls, sampled))
    for _, result in agent.target_calls:
        next_action, next_log_prob, _ = result
        assert next_action.shape == (batch_size, ACTION_DIM)
        assert next_log_prob.shape[0] == batch_size

    assert len(agent.actor_calls) == gradient_steps
    assert all(data is batch for (data, _), batch in zip(agent.actor_calls, sampled))
    for _, (loss, log_prob) in agent.actor_calls:
        assert loss.ndim == 0
        assert log_prob.shape[0] == batch_size
        assert not log_prob.requires_grad


@dataclass
class TaggedSample(ReplayBufferSample):
    tag: torch.Tensor


class TaggedBatchSAC(SAC):
    _extra_batch_slice_keys = ("tag",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.critic_tags: list[list[int]] = []
        self.actor_tags: list[list[int]] = []
        super().__init__(*args, **kwargs)

    def _sample_train_batch(self, batch_size: int) -> TaggedSample:
        sample = super()._sample_train_batch(batch_size)
        fields = {field.name: getattr(sample, field.name) for field in dataclasses.fields(sample)}
        return TaggedSample(**fields, tag=torch.arange(batch_size))

    def _critic_step(self, data):
        self.critic_tags.append(data.tag.tolist())
        return super()._critic_step(data)

    def _actor_step(self, data):
        self.actor_tags.append(data.tag.tolist())
        return super()._actor_step(data)


class UndeclaredTagSAC(TaggedBatchSAC):
    _extra_batch_slice_keys = ()


def test_extra_batch_slice_keys_carry_a_sample_field_through_high_utd_slicing():
    agent = TaggedBatchSAC(env=TinyVecEnv(), utd=2, **COMMON_KWARGS)
    fill_random_transitions(agent)

    agent.train(gradient_steps=2)

    # One group: a critic step per minibatch slice, then one actor step on the full batch.
    assert agent.critic_tags == [[0, 1], [2, 3]]
    assert agent.actor_tags == [[0, 1, 2, 3]]


def test_a_sample_field_missing_from_extra_batch_slice_keys_fails_high_utd_slicing():
    agent = UndeclaredTagSAC(env=TinyVecEnv(), utd=2, **COMMON_KWARGS)
    fill_random_transitions(agent)

    with pytest.raises(TypeError, match="tag"):
        agent.train(gradient_steps=2)


# --- prior-data buffer hooks ------------------------------------------------------------------


class RecordingReplayBuffer(ReplayBuffer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def add(self, *args: Any, **kwargs: Any) -> None:
        self.add_calls.append((args, kwargs))
        super().add(*args, **kwargs)


class DemoRLPD(DemoInterventionMixin, RLPD):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.prior_buffer_sizes: list[int] = []
        super().__init__(*args, **kwargs)

    def _build_prior_data_buffer(self, buffer_size: int) -> RecordingReplayBuffer:
        self.prior_buffer_sizes.append(buffer_size)
        return RecordingReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=1,
            buffer_size=buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )


def test_init_demo_buffer_builds_through_the_prior_data_hook_and_demos_reach_its_add():
    agent = DemoRLPD(env=TinyVecEnv(), **COMMON_KWARGS)
    fill_random_transitions(agent)

    agent.init_demo_buffer(8, demo_data_ratio=0.5)

    assert agent.prior_buffer_sizes == [8]
    assert type(agent.offline_replay_buffer) is RecordingReplayBuffer
    assert agent.offline_data_ratio == 0.5

    obs = {"state": torch.zeros(STATE_DIM)}
    action = torch.zeros(ACTION_DIM)
    reward = torch.tensor(1.0)
    done = torch.tensor(False)
    demo_transitions = 4
    for _ in range(demo_transitions):
        agent.add_demo_transition(obs, obs, action, reward, done)

    assert len(agent.offline_replay_buffer.add_calls) == demo_transitions
    args, kwargs = agent.offline_replay_buffer.add_calls[0]
    assert args == (obs, obs, action, reward, done)
    assert kwargs == {}

    # The mixed batch draws from both buffers through the inherited sampler.
    batch = agent._sample_train_batch(COMMON_KWARGS["batch_size"])
    assert batch.actions.shape == (COMMON_KWARGS["batch_size"], ACTION_DIM)
