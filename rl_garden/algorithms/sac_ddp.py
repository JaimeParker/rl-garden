"""Single-node multi-GPU DDP variant of ``SAC``.

Every hook this class overrides is a no-op when
``rl_garden.common.ddp.is_ddp_active()`` is False, so constructing
``SACDDP`` outside an active ``torch.distributed`` process group behaves
identically to plain ``SAC`` -- e.g. loading a checkpoint saved by
``SACDDP`` back into a single-process eval script should construct
``SACDDP`` (not ``SAC``) for that load, and it will behave exactly like
``SAC`` there.

v1 scope: plain ``SAC`` only (not ``SequenceSAC``/``RecurrentSAC``/
``TransformerSAC``/``SACFlow``/``RLPD``/``RLPDHybrid``/``FlashSAC``/
``DDPG``/``TD3``). Never constructed directly from the CLI -- selected
transparently by ``rl_garden.training.online.sac.build_sac`` when
``is_ddp_active()``.

KNOWN LIMITATION under multi-GPU DDP: each rank's replay buffer is
independent and rank-local (by design, same shape as PPO's DDP support --
rank-local rollout data feeds a synchronized gradient update). This means
``buffer_size`` GPU (or CPU) memory is paid N times over, once per rank,
not shared -- worth checking before enabling DDP with a large
``buffer_size``/``mmap_dir``-backed buffer.
"""
from __future__ import annotations

from rl_garden.algorithms.sac import SAC
from rl_garden.common.ddp import allreduce_param_grads


class SACDDP(SAC):
    _compatible_checkpoint_algorithms = ("SAC", "SACDDP")

    def _sync_ddp_grads(self, params: list) -> None:
        allreduce_param_grads(params)

    def _ddp_extra_broadcast_modules(self) -> list:
        if self.alpha_tuner is not None:
            return [self.alpha_tuner]
        return []
