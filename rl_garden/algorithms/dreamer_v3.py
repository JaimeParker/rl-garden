"""``DreamerV3``: RSSM world model + imagination-based actor-critic, ported
from ``3rd_party/r2dreamer/dreamer.py`` (``rep_loss="dreamer"`` branch);
official semantics win where r2dreamer deviates (see the plan's
"Decisions" list). Reference: scratchpad ``dreamer-code-survey.md``
sections 3-4, 9-10; plan ``~/.claude/plans/dreamer-v3.md`` sections B/E/F.

Model-based-base architecture: ``ModelBasedAlgorithm(OffPolicyAlgorithm)``
supplies the vectorized rollout loop, replay-buffer wiring, checkpointing,
and eval loop; this class overrides the three gradient-step hooks
(``_update_model``, ``_update_actor_critic``, ``_update_targets``) plus the
rollout hooks (``_on_env_reset``, ``_rollout_action``,
``_replay_buffer_add_kwargs``, ``_post_rollout_step``) and the eval hooks
(``_eval_start_hook``, ``_eval_action_and_critic_action``,
``_eval_step_hook`` -- mirroring ``rl_garden.algorithms.sequence_sac
.SequenceSAC``'s ``_rollout_hidden``/``_eval_hidden`` split, since
``DreamerPolicy.predict()``'s stateful signature cannot go through the
default ``BaseAlgorithm._eval_action`` path, see that class's docstring).

**Deviation from ``ModelBasedAlgorithm``'s documented hook contract**
(explicitly flagged, per model-based-base rules: the plan beats a base
class's docstring where the two conflict for this concrete case): that
base's ``_update_model`` docstring says "trains the world model" (implying
its own backward/optimizer step), matching TD-MPC2's convention. DreamerV3
instead needs exactly ONE backward pass over a single ``LaProp`` optimizer,
summing ``Σ scale_k · loss_k`` across the world-model losses AND the
imagination-based actor/critic losses AND the replay-based ``repval``
critic loss (r2dreamer ``dreamer.py``'s ``_cal_grad`` computes all of these
in one function before its single ``backward()`` call) -- ``_update_model``
here therefore only computes and stashes the (fp32-cast) world-model losses
on ``self._pending_model_losses`` (no backward, no optimizer step);
``_update_actor_critic`` computes the actor/critic/repval losses (via the
pure, side-effect-free ``_imagination_losses`` helper, split out 2026-09-14
so the actor/critic-vs-repval/model-loss gradient-isolation contract is
directly testable -- see that method's own docstring), adds them to the
stashed model losses, does the ONE combined backward, AGC-clips, and steps
the optimizer + LR scheduler.

**``contdisc``** (plan decision 2): default ``False`` here (constructor
kwarg, threaded straight into ``RSSM``, see that class's own docstring for
the exact behavioural difference from official JAX's default ``True``).

**``torch.compile``** (plan decision 5): DEFERRED, not used anywhere in
this module. Official JAX is jit-compiled end to end; r2dreamer wraps its
own ``_cal_grad`` in ``torch.compile(mode="reduce-overhead")``
(``dreamer.py:163``). Enabling ``torch.compile`` here is a follow-up, not
attempted in this commit -- also see ``docs/guides/dreamer-v3.md``.

**Discrete (``spaces.Discrete``) action spaces are out of scope for this
class** (``DreamerPolicy`` itself implements both action heads per plan
decision 4, but this algorithm currently only supports ``spaces.Box``):
wiring a discrete action through the replay buffer (whose storage shape is
literally ``action_space.shape`` -- ``()`` for ``Discrete``, not the
one-hot ``(n,)`` vector RSSM's dynamics core needs internally) and through
``RSSM.model_loss``'s ``batch.action`` (consumed directly, already assumed
one-hot) is a real design question the plan does not resolve, so this
class raises ``NotImplementedError`` for anything but ``Box`` rather than
guessing at an untested contract -- flagged in the port's final report as
an open question, not silently shipped half-working.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.model_based import ModelBasedAlgorithm
from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim.agc import clip_grad_agc_
from rl_garden.common.optim.laprop import LaProp, linear_warmup
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.config import EncoderConfig
from rl_garden.encoders.dreamer_conv import DreamerConvEncoder
from rl_garden.networks.returns import lambda_return
from rl_garden.observations import ObsGroups, normalize_observation_space
from rl_garden.observations.schema import ObservationSchema
from rl_garden.policies.dreamer_policy import DreamerPolicy
from rl_garden.world_models.imagine import clone_and_freeze, imagine
from rl_garden.world_models.rssm import RSSM, RSSM_SIZES, RSSMSize

_MODEL_LOSS_SCALE_KEYS = ("dyn", "rep", "rew", "con")


class DreamerV3(ModelBasedAlgorithm):
    _compatible_checkpoint_algorithms = ("DreamerV3",)
    # The RSSM is the sole encoder (world-model-owned representation, see
    # class docstring); there is no separate critic extractor to split
    # gradients between, so "shared" is the only supported value --
    # ObservationContractError rejects an asymmetric obs_groups or a
    # distinct critic_encoder_config at preflight (both would otherwise
    # resolve to "separate", which encoder_sharing_choices excludes).
    encoder_sharing = "shared"
    encoder_sharing_choices = ("shared",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        size: Literal["12M", "25M", "50M", "100M", "200M", "400M"] = "12M",
        stoch: int = 32,
        unimix: float = 0.01,
        blocks: int = 8,
        obs_layers: int = 1,
        img_layers: int = 2,
        dyn_layers: int = 1,
        decoder_layers: int = 3,
        reward_bins: int = 255,
        kl_free: float = 1.0,
        contdisc: bool = False,
        discount_horizon: float = 333.0,
        batch_size: int = 16,
        batch_length: int = 64,
        train_ratio: float = 512.0,
        imag_horizon: int = 15,
        lam: float = 0.95,
        act_entropy: float = 3e-4,
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        recon_scale: float = 1.0,
        rew_scale: float = 1.0,
        con_scale: float = 1.0,
        policy_scale: float = 1.0,
        value_scale: float = 1.0,
        repval_scale: float = 0.3,
        lr: float = 4e-5,
        warmup: int = 1_000,
        slow_target_fraction: float = 0.02,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 1_024,
        compute_dtype: Optional[Literal["float32", "bfloat16"]] = None,
        # Dict-obs (pixel) encoder config -- see class docstring: DreamerV3
        # builds its own DreamerConvEncoder directly from the RSSMSize
        # preset's depth/units, NOT through the generic
        # EncoderConfig-driven factory (rl_garden.encoders.dreamer_conv
        # .DreamerConvEncoder's own docstring). encoder_config is accepted
        # and recorded (checkpoint metadata, --print-config) for CLI/
        # contract uniformity with every other algorithm, and so
        # ObservationEncoderMixin's generic asymmetric/critic-encoder
        # preflight rejection works, but its field values (besides being
        # non-None) are otherwise unused.
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        # Test-only escape hatch: overrides the size preset's RSSMSize row
        # entirely (used by tests/test_dreamer_v3_smoke.py's tiny model);
        # never set by the CLI entrypoint.
        rssm_size: Optional[RSSMSize] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
    ) -> None:
        if not isinstance(env.single_action_space, spaces.Box):
            raise NotImplementedError(
                f"{type(self).__name__} only supports spaces.Box action spaces in this "
                f"port (see class docstring); got {type(env.single_action_space)}."
            )
        # Train-ratio -> (training_freq, utd) derivation (plan section F):
        # train_ratio is upstream's own definition -- "1 gradient update per
        # batch_steps/train_ratio env steps collected", batch_steps =
        # batch_size*batch_length (dreamer-code-survey.md section 6). Setting
        # utd = train_ratio / batch_steps directly reproduces that ratio
        # through OffPolicyAlgorithm's own grad_steps_per_iteration =
        # max(1, int(training_freq * utd)) formula for ANY training_freq;
        # training_freq = env.num_envs (one rollout step per env per
        # iteration, steps_per_env == 1) is chosen only for the smallest,
        # most frequent update granularity a vectorized rollout can offer,
        # not because it changes the ratio itself.
        utd = train_ratio / (batch_size * batch_length)
        training_freq = env.num_envs

        if learning_starts // env.num_envs <= batch_length:
            raise ValueError(
                f"{type(self).__name__} requires learning_starts // env.num_envs > "
                f"batch_length (a sampled window needs batch_length + 1 contiguous "
                f"rows per env, and by learning_starts each env has only "
                f"learning_starts // env.num_envs rows written): got "
                f"learning_starts={learning_starts}, env.num_envs={env.num_envs} "
                f"({learning_starts // env.num_envs} rows/env), "
                f"batch_length={batch_length}. Raise learning_starts, lower "
                f"batch_length, or lower env.num_envs."
            )

        super().__init__(
            env=env,
            eval_env=eval_env,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            learning_starts=learning_starts,
            batch_size=batch_size,
            training_freq=training_freq,
            utd=utd,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint,
        )

        self.size = size
        self._rssm_size = rssm_size if rssm_size is not None else RSSM_SIZES[f"size{size}"]
        self.stoch = stoch
        self.unimix = unimix
        self.blocks = blocks
        self.obs_layers = obs_layers
        self.img_layers = img_layers
        self.dyn_layers = dyn_layers
        self.decoder_layers = decoder_layers
        self.reward_bins = reward_bins
        self.kl_free = kl_free
        self.contdisc = contdisc
        self.discount_horizon = discount_horizon

        self.batch_length = batch_length
        self.train_ratio = train_ratio
        self.imag_horizon = imag_horizon
        self.lam = lam
        self.act_entropy = act_entropy

        self.dyn_scale = dyn_scale
        self.rep_scale = rep_scale
        self.recon_scale = recon_scale
        self.rew_scale = rew_scale
        self.con_scale = con_scale
        self.policy_scale = policy_scale
        self.value_scale = value_scale
        self.repval_scale = repval_scale

        self.lr = lr
        self.warmup = warmup
        self.slow_target_fraction = slow_target_fraction

        self.compute_dtype = compute_dtype or ("bfloat16" if self.device.type == "cuda" else "float32")
        self._autocast_kwargs = dict(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.compute_dtype == "bfloat16",
        )

        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        self.critic_encoder_config = critic_encoder_config

        self._setup_model()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        # Cheap (builds no encoder): raises ObservationContractError if an
        # asymmetric obs_groups or a critic_encoder_config was given (both
        # resolve to "separate", excluded by encoder_sharing_choices above)
        # -- see class docstring.
        self._resolve_encoder_sharing(self.env.single_observation_space)

        observation_space = self.env.single_observation_space
        normalized_space = normalize_observation_space(observation_space)
        schema = ObservationSchema.from_space(normalized_space)
        action_dim = int(np.prod(self.env.single_action_space.shape))
        size = self._rssm_size

        encoder = DreamerConvEncoder(observation_space, schema, depth=size.cnn_depth, units=size.units)
        world_model = RSSM(
            encoder,
            schema,
            action_dim,
            size,
            stoch=self.stoch,
            unimix=self.unimix,
            blocks=self.blocks,
            obs_layers=self.obs_layers,
            img_layers=self.img_layers,
            dyn_layers=self.dyn_layers,
            decoder_layers=self.decoder_layers,
            reward_bins=self.reward_bins,
            kl_free=self.kl_free,
            contdisc=self.contdisc,
            discount_horizon=self.discount_horizon,
        )
        self.policy = DreamerPolicy(
            observation_space,
            self.env.single_action_space,
            world_model,
            units=size.units,
            reward_bins=self.reward_bins,
            slow_target_fraction=self.slow_target_fraction,
        ).to(self.device)
        world_model = self.policy.world_model  # now on self.device

        # LaProp trains rssm + encoder/decoder + reward/continue heads +
        # actor + critic -- NOT slow_critic (an EMA copy, see
        # DreamerPolicy.slow_critic's docstring and _update_targets below).
        self._trainable_params = [
            p for name, p in self.policy.named_parameters() if not name.startswith("slow_critic.")
        ]
        self.optimizer = LaProp(self._trainable_params, lr=self.lr, betas=(0.9, 0.999), eps=1e-20)
        self._lr_scheduler = linear_warmup(self.optimizer, self.warmup)

        self.replay_buffer = SequenceReplayBuffer(
            observation_space=observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            cross_episode=True,
            priority=False,
            horizon=self.batch_length,
            carry_spec={
                "deter": (size.deter,),
                "stoch": (self.stoch * size.discrete,),
            },
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    # ------------------------------------------------------------------
    # Rollout: carry the RSSM state across env steps via observe() itself
    # (no explicit zeroing in the loop -- is_first masking inside observe()
    # does the reset, plan section F).
    # ------------------------------------------------------------------

    def _on_env_reset(self, obs) -> None:
        super()._on_env_reset(obs)
        self._rollout_state = self.world_model.initial_state(self.num_envs, self.device)
        self._rollout_is_first = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _rollout_action(self, obs, learning_has_started: bool):
        obs_device = self._obs_to_policy_device(obs)
        with torch.no_grad(), torch.autocast(**self._autocast_kwargs):
            action, new_state = self.policy.predict(
                obs_device, self._rollout_state, self._rollout_is_first, deterministic=False
            )
            if not learning_has_started:
                # Random-action prefill (plan section F): the RSSM carry is
                # still advanced through observe() above (needed regardless
                # of exploration phase -- it is a fact about obs, not about
                # which action gets taken from it); only the EXECUTED action
                # (and the "prev_action" fed to the next observe() call) is
                # substituted for a uniform-random one.
                action = self._explore_action(obs)
                new_state = {**new_state, "prev_action": action}
        action = action.float()
        self._rollout_state = new_state
        action_context = {"new_state": new_state}
        return action, action, action_context

    def _replay_buffer_add_kwargs(
        self, action_context, obs, next_obs, real_next_obs, infos, need_final_obs
    ) -> dict[str, Any]:
        del obs, next_obs, real_next_obs, infos, need_final_obs
        new_state = action_context["new_state"]
        carry = {
            "deter": new_state["deter"],
            "stoch": new_state["stoch"].reshape(new_state["stoch"].shape[0], -1),
        }
        return {"is_first": self._rollout_is_first, "carry": carry}

    def _post_rollout_step(self, action_context, terminations, truncations, infos) -> None:
        del action_context, infos
        # self._rollout_state was already advanced (through observe()) in
        # _rollout_action -- only the NEXT step's is_first flag is set here.
        # A CPU-backed env backend (e.g. mujoco.device="cpu") returns
        # terminations/truncations on the CPU -- move to self.device before
        # combining with anything else so the next predict() call's
        # torch.where(mask, ...) doesn't mix a CPU is_first with a
        # CUDA-resident rollout state (RSSM.observe's device requirement).
        self._rollout_is_first = (terminations | truncations).to(self.device)

    # ------------------------------------------------------------------
    # Eval: separate state from the rollout's (TDMPC2._rollout_action's
    # documented hazard -- _evaluate() interleaves with an in-progress
    # training episode within one learn() call).
    # ------------------------------------------------------------------

    def _eval_start_hook(self) -> None:
        self._eval_state = self.world_model.initial_state(self.eval_env.num_envs, self.device)
        self._eval_is_first = torch.ones(self.eval_env.num_envs, dtype=torch.bool, device=self.device)

    def _eval_action_and_critic_action(self, obs):
        obs_device = self._obs_to_policy_device(obs)
        with torch.no_grad(), torch.autocast(**self._autocast_kwargs):
            action, new_state = self.policy.predict(
                obs_device, self._eval_state, self._eval_is_first, deterministic=True
            )
        action = action.float()
        self._eval_state = new_state
        return action, action

    def _eval_step_hook(self, obs_before, critic_action, rewards, terminations, truncations, infos) -> None:
        del obs_before, critic_action, rewards, infos
        self._eval_is_first = (terminations | truncations).to(self.device)

    # ------------------------------------------------------------------
    # Gradient step: ModelBasedAlgorithm's three hooks. See class docstring
    # -- _update_model stashes losses (no backward/step); _update_actor_critic
    # does the ONE combined backward + AGC clip + optimizer/scheduler step.
    # ------------------------------------------------------------------

    def _update_model(self, batch) -> tuple[dict[str, float], Any]:
        with torch.autocast(**self._autocast_kwargs):
            losses, posterior = self.world_model.model_loss(batch)
        # fp32 reduction (plan section F): autocast keeps a bf16 .mean()
        # result in bf16; cast each scalar loss back to float32 before it
        # ever contributes to the single combined backward in
        # _update_actor_critic.
        losses = {k: v.float() for k, v in losses.items()}
        self._pending_model_losses = losses
        # Write the freshly-computed posterior deter/stoch back into the
        # buffer's per-step carry table (detached -- plan section F), so the
        # next sample() of these slots warm-starts from an up-to-date
        # posterior instead of the value stored when they were first added
        # (r2dreamer buffer.update / JAX Replay.update).
        carry = {
            "deter": posterior["deter"].detach(),
            "stoch": posterior["stoch"].detach().reshape(*posterior["stoch"].shape[:2], -1),
        }
        self.replay_buffer.write_back_carry(batch.indices, carry)
        metrics = {f"{k}_loss": float(v.detach()) for k, v in losses.items()}
        return metrics, posterior

    def _imagination_losses(self, batch, posterior) -> dict[str, Any]:
        """Pure (no backward, no optimizer step, no mutation of ``self``):
        computes the imagination-based actor/critic losses plus the
        replay-based ``repval`` critic loss from a fresh frozen-snapshot
        imagination rollout, and returns them alongside the diagnostics
        ``_update_actor_critic`` logs. Split out of that method (fixer pass,
        2026-09-14) so ``tests/test_dreamer_v3_smoke.py`` can exercise the
        actor/critic-vs-repval/model-loss gradient-isolation contract (the
        actor/critic losses below read only a FROZEN, no-grad snapshot of
        the RSSM -- ``frozen_rssm``/``feat_all`` are computed under
        ``torch.no_grad()`` -- so they carry zero gradient into the live
        RSSM/encoder/decoder; only ``repval_loss`` below, and the world-model
        losses ``_update_model`` computes separately, do) without a real
        backward/optimizer step. ``_update_actor_critic`` is the only
        caller in production; it owns the combined backward + AGC clip +
        optimizer/scheduler step (class docstring)."""
        world_model = self.world_model
        horizon, num_envs = posterior["deter"].shape[0], posterior["deter"].shape[1]
        is_discrete = self.policy.is_discrete
        disc = 1.0 if self.contdisc else 1.0 - 1.0 / self.discount_horizon

        start_state = {
            "deter": posterior["deter"].detach().reshape(horizon * num_envs, -1),
            "stoch": posterior["stoch"].detach().reshape(horizon * num_envs, *posterior["stoch"].shape[2:]),
            "logits": posterior["logits"].detach().reshape(horizon * num_envs, *posterior["logits"].shape[2:]),
        }

        with torch.autocast(**self._autocast_kwargs):
            frozen_rssm = clone_and_freeze(world_model)
            frozen_actor = clone_and_freeze(self.policy.actor)
            frozen_critic = clone_and_freeze(self.policy.critic)
            frozen_slow_critic = clone_and_freeze(self.policy.slow_critic)

            def policy_fn(state):
                feat = frozen_rssm.features(state)
                dist = frozen_actor(feat)
                return dist.sample() if is_discrete else dist.rsample()

            with torch.no_grad():
                traj = imagine(
                    frozen_rssm, policy_fn, start_state, horizon=self.imag_horizon, grad=False
                )
                feat_all = frozen_rssm.features(traj.states)  # (imag_horizon+1, horizon*num_envs, latent)
                imag_reward = frozen_rssm.reward_head(feat_all).mode
                imag_cont = frozen_rssm.cont_head(feat_all).mean
                imag_value = frozen_critic(feat_all).mode
                imag_slow_value = frozen_slow_critic(feat_all).mode

                weight = torch.cumprod(imag_cont * disc, dim=0)
                last = torch.zeros_like(imag_cont)
                term = 1.0 - imag_cont
                ret = lambda_return(
                    imag_reward, imag_value, imag_value, last=last, term=term, disc=disc, lam=self.lam
                )
                # boot for the replay-based repval return below: the
                # imagination return's own first row, reshaped back to the
                # replay batch's (horizon, num_envs, 1) layout -- r2dreamer
                # dreamer.py's `boot = ret[:, 0].reshape(B, T, 1)`.
                boot_replay = ret[0].reshape(horizon, num_envs, 1)

            ret_offset, ret_scale = self.policy.return_ema(ret)
            del ret_offset
            adv = (ret - imag_value[:-1]) / ret_scale

            feat_leading = feat_all[:-1]  # rows 0..imag_horizon-1, matches ret/traj.actions

            action_dist = self.policy.actor(feat_leading)
            logp = action_dist.log_prob(traj.actions).unsqueeze(-1)
            entropy = action_dist.entropy().unsqueeze(-1)
            weight_detached = weight[:-1].detach()
            # .float() before the final .mean() reduction (plan section F:
            # "losses reduced in fp32") -- autocast keeps every op above
            # in bf16 when enabled; only the scalar loss itself needs fp32.
            actor_loss = -(
                (logp * adv.detach() + self.act_entropy * entropy) * weight_detached
            ).float().mean()

            critic_dist = self.policy.critic(feat_leading)
            critic_loss = -(
                (
                    critic_dist.log_prob(ret.detach())
                    + critic_dist.log_prob(imag_slow_value[:-1].detach())
                )
                * weight_detached.squeeze(-1)
            ).float().mean()

            # repval: critic regression on the LIVE (gradient-attached
            # into rssm/encoder) replay posteriors -- r2dreamer dreamer.py's
            # "replay-based value learning" section.
            feat_live = world_model.features(posterior)
            with torch.no_grad():
                replay_value = frozen_critic(feat_live).mode
                replay_slow_value = frozen_slow_critic(feat_live).mode
                last_replay = batch.is_last.float().unsqueeze(-1)
                term_replay = batch.is_terminal.float().unsqueeze(-1)
                reward_replay = batch.reward.unsqueeze(-1)
                ret_replay = lambda_return(
                    reward_replay, replay_value, boot_replay,
                    last=last_replay, term=term_replay, disc=disc, lam=self.lam,
                )
                weight_replay = (1.0 - last_replay)[:-1].detach()

            live_value_dist = self.policy.critic(feat_live[:-1])
            repval_loss = -(
                (
                    live_value_dist.log_prob(ret_replay.detach())
                    + live_value_dist.log_prob(replay_slow_value[:-1].detach())
                )
                * weight_replay.squeeze(-1)
            ).float().mean()

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "repval_loss": repval_loss,
            "disc": disc,
            "ret": ret,
            "adv": adv,
            "ret_scale": ret_scale,
        }

    def _update_actor_critic(self, batch, posterior) -> dict[str, float]:
        imag = self._imagination_losses(batch, posterior)

        with torch.autocast(**self._autocast_kwargs):
            recon_loss = sum(
                v for k, v in self._pending_model_losses.items() if k not in _MODEL_LOSS_SCALE_KEYS
            )
            total_loss = (
                self.dyn_scale * self._pending_model_losses["dyn"]
                + self.rep_scale * self._pending_model_losses["rep"]
                + self.rew_scale * self._pending_model_losses["rew"]
                + self.con_scale * self._pending_model_losses["con"]
                + self.recon_scale * recon_loss
                + self.policy_scale * imag["actor_loss"]
                + self.value_scale * imag["critic_loss"]
                + self.repval_scale * imag["repval_loss"]
            )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        clip_grad_agc_(self._trainable_params, clip=0.3, pmin=1e-3)
        self.optimizer.step()
        self._lr_scheduler.step()

        metrics = {
            "actor_loss": float(imag["actor_loss"].detach()),
            "critic_loss": float(imag["critic_loss"].detach()),
            "repval_loss": float(imag["repval_loss"].detach()),
            "total_loss": float(total_loss.detach()),
            "ret_mean": float(imag["ret"].detach().mean()),
            "adv_mean": float(imag["adv"].detach().mean()),
            "return_ema_scale": float(imag["ret_scale"].detach()),
            "lr": float(self._lr_scheduler.get_last_lr()[0]),
        }
        del self._pending_model_losses
        return metrics

    def _update_targets(self) -> None:
        polyak_update(
            self.policy.critic.parameters(), self.policy.slow_critic.parameters(), self.slow_target_fraction
        )
        self.policy.slow_critic.eval()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("optimizer",)

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {**super()._extra_checkpoint_state(), "lr_scheduler_state": self._lr_scheduler.state_dict()}

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        sched_state = state.get("lr_scheduler_state")
        if sched_state is not None:
            self._lr_scheduler.load_state_dict(sched_state)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "size": self.size,
            "stoch": self.stoch,
            "unimix": self.unimix,
            "blocks": self.blocks,
            "obs_layers": self.obs_layers,
            "img_layers": self.img_layers,
            "dyn_layers": self.dyn_layers,
            "decoder_layers": self.decoder_layers,
            "reward_bins": self.reward_bins,
            "kl_free": self.kl_free,
            "contdisc": self.contdisc,
            "discount_horizon": self.discount_horizon,
            "batch_length": self.batch_length,
            "train_ratio": self.train_ratio,
            "imag_horizon": self.imag_horizon,
            "lam": self.lam,
            "act_entropy": self.act_entropy,
            "dyn_scale": self.dyn_scale,
            "rep_scale": self.rep_scale,
            "recon_scale": self.recon_scale,
            "rew_scale": self.rew_scale,
            "con_scale": self.con_scale,
            "policy_scale": self.policy_scale,
            "value_scale": self.value_scale,
            "repval_scale": self.repval_scale,
            "lr": self.lr,
            "warmup": self.warmup,
            "slow_target_fraction": self.slow_target_fraction,
            "compute_dtype": self.compute_dtype,
            "encoder_sharing": self.encoder_sharing,
            "encoder_sharing_origin": self.encoder_sharing_origin,
            "encoder_config": (
                dataclasses.asdict(self.encoder_config) if self.encoder_config is not None else None
            ),
            "obs_groups": (
                dataclasses.asdict(self.obs_groups) if self.obs_groups is not None else None
            ),
        }
