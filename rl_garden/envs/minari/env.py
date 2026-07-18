"""Minari-backed vectorized env: N ``recover_environment()`` copies batched
with ``gymnasium.vector.SyncVectorEnv``, wrapped in rl-garden's shared
``TorchVectorEnvAdapter`` (see ``rl_garden.envs.vector_env`` module
docstring for the ``final_observation``/``final_info`` contract).
"""

from __future__ import annotations

from rl_garden.envs.minari.config import MinariEnvConfig
from rl_garden.envs.vector_env import TorchVectorEnvAdapter


def make_minari_env(cfg: MinariEnvConfig) -> TorchVectorEnvAdapter:
    import gymnasium.vector
    import minari
    from gymnasium.wrappers import RecordEpisodeStatistics

    dataset = minari.load_dataset(cfg.dataset_id, download=cfg.download)

    def _make_sub_env():
        return RecordEpisodeStatistics(dataset.recover_environment(eval_env=cfg.eval_env))

    env_fns = [_make_sub_env for _ in range(cfg.num_envs)]
    vec_env = gymnasium.vector.SyncVectorEnv(
        env_fns, autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP
    )
    return TorchVectorEnvAdapter(vec_env, device=cfg.device)
