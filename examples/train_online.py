"""Online RL entry point: sac / drqv2 / flash_sac / ppo.

To add a new algorithm: create ``rl_garden/training/online/<name>.py`` with an Args
dataclass, a run function, and a package-local ``registry.register`` call.
No other file needs to change — all public modules in ``rl_garden/training/online/`` are
discovered and imported automatically at startup.

Pass ``--print-config`` anywhere on the command line to print the fully resolved
configuration as JSON and exit without creating training resources.

Usage:
    # SAC – state obs (default)
    python examples/train_online.py sac --env_id PickCube-v1

    # SAC – visual obs
    python examples/train_online.py sac --env_id PickCube-v1 \\
        --obs.rgb base_camera --encoder.backbone plain_conv

    # SAC – RoboTwin backend
    python examples/train_online.py sac --env_backend robotwin --env_id place_shoe

    # DrQ-v2 (requires image observations)
    python examples/train_online.py drqv2 --env_id PickCube-v1 --obs.rgb base_camera

    # DrQ-v2 with non-default ManiSkill backend options
    python examples/train_online.py drqv2 --obs.rgb base_camera \\
        --maniskill.reward-mode normalized_dense

    # FlashSAC
    python examples/train_online.py flash_sac --env_id PickCube-v1

    # PPO – visual obs
    python examples/train_online.py ppo --env_id PickCube-v1 --obs.rgb base_camera

    # PPO – state obs (default)
    python examples/train_online.py ppo --env_id PickCube-v1
"""
from __future__ import annotations

from rl_garden.training.online import registry


def main() -> None:
    registry.run_cli()


if __name__ == "__main__":
    main()
