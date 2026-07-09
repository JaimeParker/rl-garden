"""ActorLoop: fixed-control-frequency rollout loop driving a real-robot env.

Deliberately independent of ``OffPolicyAlgorithm.learn()`` -- that loop
inlines rollout and gradient updates together for a single process; a real
robot's ``env.step()`` must run at a fixed wall-clock cadence and can never be
paused for a gradient update, so the actor only ever does inference (under
``torch.no_grad()``) against a policy module, never training.

HITL intervention / reward-classifier / safety-box logic all live in the env
wrapper stack (``rl_garden/envs/wrappers/``) and are exercised transparently
through ``env.step()`` -- ``ActorLoop`` itself doesn't need to know about any
of them, it only reads ``info["intervene_action"]`` when present so the
replay data records the action that was actually executed, matching SERL's
convention for off-policy consistency under human intervention.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import torch

from rl_garden.policies.base import BasePolicy
from rl_garden.real_world.sync import ActorSyncClient


class ActorLoop:
    def __init__(
        self,
        env: Any,
        policy: BasePolicy,
        sync_client: ActorSyncClient,
        control_hz: float = 10.0,
        device: str | torch.device = "cpu",
        deterministic: bool = False,
        seed: int = 1,
    ) -> None:
        if getattr(env, "num_envs", 1) != 1:
            raise ValueError(
                f"ActorLoop drives exactly one real robot; got env.num_envs="
                f"{getattr(env, 'num_envs', None)!r}."
            )
        self.env = env
        self.policy = policy
        self.sync_client = sync_client
        self.control_period = 1.0 / control_hz
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.seed = seed

    def _obs_to_policy_device(self, obs):
        if isinstance(obs, dict):
            return {k: v.to(self.device) for k, v in obs.items()}
        return obs.to(self.device)

    def _maybe_refresh_policy(self) -> None:
        params = self.sync_client.latest_policy_params()
        if params is not None:
            self.policy.load_state_dict(params)

    def _predict(self, obs) -> torch.Tensor:
        with torch.no_grad():
            return self.policy.predict(
                self._obs_to_policy_device(obs), deterministic=self.deterministic
            )

    def run(self, total_steps: Optional[int] = None) -> None:
        self.sync_client.start()
        self.policy.eval()
        try:
            obs, _ = self.env.reset(seed=self.seed)
            step = 0
            while total_steps is None or step < total_steps:
                loop_start = time.perf_counter()
                self._maybe_refresh_policy()

                policy_action = self._predict(obs)
                env_action = policy_action.to(self.env_device(obs))
                next_obs, reward, terminated, truncated, info = self.env.step(env_action)

                executed_action = info.get("intervene_action", policy_action)
                self.sync_client.push_transition(
                    {
                        "obs": obs,
                        "next_obs": next_obs,
                        "action": executed_action,
                        "reward": reward,
                        "done": terminated,
                    }
                )

                if bool(terminated) or bool(truncated):
                    obs, _ = self.env.reset(seed=self.seed)
                else:
                    obs = next_obs
                step += 1

                elapsed = time.perf_counter() - loop_start
                sleep_for = self.control_period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            self.sync_client.stop()

    @staticmethod
    def env_device(obs) -> torch.device:
        sample = next(iter(obs.values())) if isinstance(obs, dict) else obs
        return sample.device


class FWBWActorLoop:
    """Reset-free forward/backward variant of :class:`ActorLoop`.

    Drives one env wrapped with
    ``rl_garden.envs.wrappers.fwbw_reset_free.FWBWResetFreeWrapper`` (which
    must sit outermost, above ``RewardClassifierWrapper``), holding two full
    (policy, learner-sync-client) pairs and switching which pair does
    inference / receives each transition based on ``info["fwbw_direction"]``
    -- a single ``ActorLoop`` can't do this since it only ever knows about
    one policy and one learner.
    """

    def __init__(
        self,
        env: Any,
        policies: dict[str, BasePolicy],
        sync_clients: dict[str, ActorSyncClient],
        control_hz: float = 10.0,
        device: str | torch.device = "cpu",
        deterministic: bool = False,
        seed: int = 1,
    ) -> None:
        if getattr(env, "num_envs", 1) != 1:
            raise ValueError(
                f"FWBWActorLoop drives exactly one real robot; got env.num_envs="
                f"{getattr(env, 'num_envs', None)!r}."
            )
        if set(policies) != {"forward", "backward"} or set(sync_clients) != {"forward", "backward"}:
            raise ValueError(
                "policies and sync_clients must each have exactly 'forward' and "
                "'backward' keys."
            )
        self.env = env
        self.policies = policies
        self.sync_clients = sync_clients
        self.control_period = 1.0 / control_hz
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.seed = seed

    def _obs_to_policy_device(self, obs):
        if isinstance(obs, dict):
            return {k: v.to(self.device) for k, v in obs.items()}
        return obs.to(self.device)

    def run(self, total_steps: Optional[int] = None) -> None:
        for client in self.sync_clients.values():
            client.start()
        for policy in self.policies.values():
            policy.eval()
        try:
            obs, info = self.env.reset(seed=self.seed)
            direction = info.get("fwbw_direction", "forward")
            step = 0
            while total_steps is None or step < total_steps:
                loop_start = time.perf_counter()

                policy = self.policies[direction]
                sync_client = self.sync_clients[direction]
                params = sync_client.latest_policy_params()
                if params is not None:
                    policy.load_state_dict(params)

                with torch.no_grad():
                    policy_action = policy.predict(
                        self._obs_to_policy_device(obs), deterministic=self.deterministic
                    )
                env_action = policy_action.to(ActorLoop.env_device(obs))
                next_obs, reward, terminated, truncated, info = self.env.step(env_action)

                executed_action = info.get("intervene_action", policy_action)
                sync_client.push_transition(
                    {
                        "obs": obs,
                        "next_obs": next_obs,
                        "action": executed_action,
                        "reward": reward,
                        "done": terminated,
                    }
                )

                direction = info.get("fwbw_direction", direction)
                if bool(terminated) or bool(truncated):
                    obs, info = self.env.reset(seed=self.seed)
                    direction = info.get("fwbw_direction", direction)
                else:
                    obs = next_obs
                step += 1

                elapsed = time.perf_counter() - loop_start
                sleep_for = self.control_period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            for client in self.sync_clients.values():
                client.stop()
