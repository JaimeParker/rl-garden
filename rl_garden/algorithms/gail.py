"""GAIL (Ho & Ermon 2016, https://arxiv.org/abs/1606.03476): PPO generator +
adversarial discriminator, ported from ``3rd_party/imitation``'s
``algorithms/adversarial/{gail,common}.py`` (``GAIL``/``AdversarialTrainer``).
Upstream is deeply coupled to Stable-Baselines3 (``VecEnv``, numpy
``ReplayBuffer``); only the algorithm is ported, not the code.

Each round: PPO's own rollout collects a window of environment transitions
under the *current* discriminator's reward (substituted by
``rl_garden.envs.wrappers.gail_reward.GAILRewardWrapper``, wrapped around
``self.env`` in ``_setup_model()`` -- no changes to ``on_policy.py``/
``ppo.py``), PPO's own clipped-surrogate update trains the policy against
that reward, then the discriminator is updated via BCE classification of
"expert vs generator" ``(obs, action)`` samples (matching upstream's
``compute_train_stats``/``train_disc``, labels: expert=1, generator=0).

Upstream's own default discriminator config is
``BasicRewardNet(use_state=True, use_action=True, use_next_state=False,
use_done=False)`` -- the discriminator only needs ``(obs, action)``, both
already stored by ``RolloutBuffer`` every step, so no buffer schema change
is needed either. Box observations only (matches the D4RL MuJoCo locomotion
target; Dict/image support would need a Dict-aware discriminator).

Expert demonstrations are loaded once at construction time into a plain
``TensorReplayBuffer`` via the existing
``rl_garden.buffers.d4rl_legacy_dataset.load_d4rl_legacy_dataset_to_replay_buffer``
-- no new dataset-loading code.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from rl_garden.algorithms.ppo import PPO
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.optim import make_optimizer
from rl_garden.envs.wrappers.gail_reward import GAILRewardWrapper
from rl_garden.networks.discriminator import GAILDiscriminator


class GAIL(PPO):
    """GAIL: PPO generator with reward substituted by an adversarially
    trained discriminator."""

    _compatible_checkpoint_algorithms = ("GAIL",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        demo_env_id: str,
        demo_dataset_backend: str = "d4rl_legacy",
        demo_buffer_size: int = 1_000_000,
        demo_batch_size: int = 1024,
        n_disc_updates_per_round: int = 4,
        disc_net_arch: Sequence[int] = (32, 32),
        disc_lr: float = 3e-4,
        disc_use_adamw: bool = False,
        **ppo_kwargs: Any,
    ) -> None:
        # Assigned before super().__init__(): PPO.__init__ calls
        # self._setup_model() as its own last line, which dynamically
        # dispatches to GAIL._setup_model() below -- these attributes must
        # already exist by then.
        self.demo_env_id = demo_env_id
        self.demo_dataset_backend = demo_dataset_backend
        self.demo_buffer_size = demo_buffer_size
        self.demo_batch_size = demo_batch_size
        self.n_disc_updates_per_round = n_disc_updates_per_round
        self.disc_net_arch = tuple(disc_net_arch)
        self.disc_lr = disc_lr
        self.disc_use_adamw = disc_use_adamw
        super().__init__(env, eval_env, **ppo_kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self.discriminator = GAILDiscriminator(
            self.env.single_observation_space,
            self.env.single_action_space,
            net_arch=self.disc_net_arch,
        ).to(self.device)
        self.disc_optimizer = make_optimizer(
            self.discriminator.parameters(),
            lr=self.disc_lr,
            use_adamw=self.disc_use_adamw,
        )
        self._demo_buffer = self._build_demo_buffer()
        if self.demo_batch_size > self.rollout_buffer.buffer_size:
            raise ValueError(
                f"demo_batch_size ({self.demo_batch_size}) must be <= "
                f"num_steps * num_envs ({self.rollout_buffer.buffer_size})."
            )
        # Wrap only the training env -- eval_env stays unwrapped so
        # evaluation reports ground-truth env reward.
        self.env = GAILRewardWrapper(self.env, reward_fn=self._discriminator_reward)

    def _build_demo_buffer(self) -> TensorReplayBuffer:
        from rl_garden.buffers.d4rl_legacy_dataset import (
            load_d4rl_legacy_dataset_to_replay_buffer,
        )

        if not self.demo_env_id:
            raise ValueError("demo_env_id must be set to a D4RL dataset id.")
        if self.demo_dataset_backend != "d4rl_legacy":
            raise NotImplementedError(
                "GAIL currently only supports demo_dataset_backend="
                f"'d4rl_legacy', got {self.demo_dataset_backend!r}."
            )
        buffer = TensorReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=1,
            buffer_size=self.demo_buffer_size,
            storage_device=self.device,
            sample_device=self.device,
        )
        load_d4rl_legacy_dataset_to_replay_buffer(buffer, self.demo_env_id)
        return buffer

    def _discriminator_reward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # CPU-backed env backends (e.g. d4rl_legacy's mujoco_py) hand obs/action
        # to the wrapper on CPU while the discriminator lives on self.device --
        # same CPU-env/GPU-training gap _obs_to_policy_device() exists for.
        # rollout_buffer.add() itself moves rewards to self.device, so
        # returning the reward on self.device (not the original obs device)
        # is fine.
        obs = self._obs_to_policy_device(obs)
        action = action if action.device == self.device else action.to(self.device)
        with torch.no_grad():
            logits = self.discriminator(obs, action)
        # RewardNetFromDiscriminatorLogit (imitation gail.py): R = -log(sigmoid(-logit)).
        return -F.logsigmoid(-logits)

    def train(self) -> dict[str, float]:
        losses = super().train()
        disc_losses = []
        disc_accs = []
        for _ in range(self.n_disc_updates_per_round):
            gen_sample = next(self.rollout_buffer.get(self.demo_batch_size))
            expert_sample = self._demo_buffer.sample(self.demo_batch_size)
            stats = self._train_discriminator_step(
                gen_sample.obs,
                gen_sample.actions,
                expert_sample.obs,
                expert_sample.actions,
            )
            disc_losses.append(stats["disc_loss"])
            disc_accs.append(stats["disc_acc"])
        losses["disc_loss"] = sum(disc_losses) / len(disc_losses)
        losses["disc_acc"] = sum(disc_accs) / len(disc_accs)
        return losses

    def _train_discriminator_step(
        self,
        gen_obs: torch.Tensor,
        gen_actions: torch.Tensor,
        expert_obs: torch.Tensor,
        expert_actions: torch.Tensor,
    ) -> dict[str, float]:
        # Expert=1, generator=0 -- matches imitation's compute_train_stats/train_disc.
        obs = torch.cat([expert_obs, gen_obs], dim=0)
        actions = torch.cat([expert_actions, gen_actions], dim=0)
        labels = torch.cat(
            [
                torch.ones(expert_obs.shape[0], device=self.device),
                torch.zeros(gen_obs.shape[0], device=self.device),
            ]
        )
        self.disc_optimizer.zero_grad()
        logits = self.discriminator(obs, actions)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        self.disc_optimizer.step()
        with torch.no_grad():
            acc = ((logits > 0).float() == labels).float().mean()
        return {"disc_loss": loss.item(), "disc_acc": acc.item()}

    # --- Checkpoint extension points ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return super()._optimizer_names() + ("disc_optimizer",)

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            **super()._extra_checkpoint_state(),
            "discriminator": self.discriminator.state_dict(),
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        if "discriminator" in state:
            self.discriminator.load_state_dict(state["discriminator"])

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "demo_env_id": self.demo_env_id,
            "demo_dataset_backend": self.demo_dataset_backend,
            "demo_batch_size": self.demo_batch_size,
            "n_disc_updates_per_round": self.n_disc_updates_per_round,
            "disc_net_arch": list(self.disc_net_arch),
            "disc_lr": self.disc_lr,
        }
