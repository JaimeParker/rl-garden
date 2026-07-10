"""Reward-classifier data collection, porting HIL-SERL's own mechanism
(``3rd_party/hil-serl/examples/record_success_fail.py``): continuously step
the real robot (a human can intervene via ``TeleopInterventionWrapper``),
globally listen for the spacebar via ``pynput`` to mark the *most recently
collected* transition as a success, everything else defaults to failure.
Saves two pickle files under ``--output_dir`` on exit, matching
``rl_garden.models.reward.success.data.SuccessClassifierDataset``'s expected
``classifier_data/*_{success,failure}_*.pkl`` naming and ``{"obs": obs}``
entry shape.

Camera capture is caller-supplied (see
``rl_garden/envs/franka_real/env.py``'s module docstring -- camera SDK
dependencies are deliberately kept out of the env itself and out of the
``franka_real`` backend registration, which always passes
``camera_capture=None``): the CLI entrypoint below has no camera wired up,
run it as a library (``main(camera_capture=...)``) with a cell-specific
capture callable to actually record images.
"""
from __future__ import annotations

import datetime
import os
import pickle
from dataclasses import dataclass, field

import torch
import tyro

from robot_infra.teleop.examples.record_teleop_wsrl import index_tree


@dataclass
class Args:
    output_dir: str
    successes_needed: int = 200
    exp_name: str = "franka_real"

    bridge_url: str = "http://localhost:5000"
    action_scale_pos: float = 0.02
    action_scale_rot: float = 0.1
    gripper_threshold: float = 0.5
    max_episode_steps: int = 100
    camera_keys: tuple[str, ...] = field(default_factory=tuple)
    camera_height: int = 128
    camera_width: int = 128

    teleop_device: str = "pico"


def main(camera_capture=None) -> None:
    args = tyro.cli(Args)

    from pynput import keyboard

    from rl_garden.envs.franka_real import FrankaRealEnvConfig, make_franka_real_env
    from rl_garden.envs.wrappers.teleop_intervention import TeleopInterventionWrapper

    success_key = [False]

    def on_press(key):
        if str(key) == "Key.space":
            success_key[0] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    cfg = FrankaRealEnvConfig(
        bridge_url=args.bridge_url,
        action_scale=(args.action_scale_pos, args.action_scale_rot),
        gripper_threshold=args.gripper_threshold,
        max_episode_steps=args.max_episode_steps,
        camera_keys=args.camera_keys,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )
    env = make_franka_real_env(cfg, camera_capture=camera_capture)
    env = TeleopInterventionWrapper(env, device=args.teleop_device)
    image_keys = [k for k in env.single_observation_space.spaces if k != "state"]

    successes: list[dict] = []
    failures: list[dict] = []

    obs, _ = env.reset()
    try:
        while len(successes) < args.successes_needed:
            action = torch.zeros(env.action_space.shape)
            next_obs, _reward, terminated, truncated, info = env.step(action)

            image_obs = {k: index_tree(obs[k], 0) for k in image_keys}
            entry = {"obs": image_obs}
            if success_key[0]:
                successes.append(entry)
                success_key[0] = False
                print(f"success {len(successes)}/{args.successes_needed}", flush=True)
            else:
                failures.append(entry)

            obs = next_obs
            if bool(terminated) or bool(truncated):
                obs, _ = env.reset()
    finally:
        listener.stop()
        env.close()

    os.makedirs(args.output_dir, exist_ok=True)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_path = os.path.join(
        args.output_dir, f"{args.exp_name}_{args.successes_needed}_success_images_{uuid}.pkl"
    )
    failure_path = os.path.join(args.output_dir, f"{args.exp_name}_failure_images_{uuid}.pkl")
    with open(success_path, "wb") as f:
        pickle.dump(successes, f)
    with open(failure_path, "wb") as f:
        pickle.dump(failures, f)
    print(f"saved {len(successes)} successes to {success_path}", flush=True)
    print(f"saved {len(failures)} failures to {failure_path}", flush=True)


if __name__ == "__main__":
    main()
