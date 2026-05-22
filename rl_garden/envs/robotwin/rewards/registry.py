"""Task-specific RoboTwin reward factories.

Factories are adapted from RoboTwin's ``RLinf_support`` branch where dense
reward configs were embedded directly in task files. Keeping them here avoids
vendoring task implementations and lets rl-garden own reward tuning.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rl_garden.envs.robotwin.reward import (
    Contact,
    Endpose,
    Pick,
    Place,
    Reward,
    Stack,
    Success,
)

RewardBuilder = Callable[[Any], Any]


def _arm_from_x(x: float) -> str:
    return "right" if x > 0 else "left"


def build_adjust_bottle(task):
    target = [0.17 if getattr(task, "qpose_tag", 0) == 1 else -0.17, 0, 0.9]
    return Reward.build_top(
        {
            "type": "Serial",
            "subtasks": [
                Pick(base=task, max_reward=4, entity=task.bottle, dist=0.23, a_d=1, a_g=0.8),
                Place(
                    base=task,
                    max_reward=4,
                    entity=task.bottle,
                    target=target,
                    eef_dim=[1, 0, 1],
                    c_d=4,
                    c_g=0,
                ),
                Success(),
            ],
            "transition_rewards": [1, 3],
        }
    )


def build_beat_block_hammer(task):
    block_pose = task.block.get_functional_point(0, "pose").p
    place_target_pose = [block_pose[0], block_pose[1] + 0.05, block_pose[2]]
    return Reward.build_top(
        {
            "type": "Serial",
            "subtasks": [
                Pick(base=task, max_reward=4, entity=task.hammer, dist=0.28),
                Place(
                    base=task,
                    max_reward=4,
                    entity=task.hammer,
                    target=place_target_pose,
                    eef_dim=2,
                    is_function_point=0,
                ),
                Success(),
            ],
            "transition_rewards": [1, 3],
        }
    )


def build_click_bell(task):
    arm_tag = _arm_from_x(task.bell.get_pose().p[0])
    return Reward.build_top(
        {
            "type": "Serial",
            "subtasks": [
                Contact(
                    base=task,
                    max_reward=4,
                    entity=task.bell,
                    arm_tag=arm_tag,
                    entity_name="050_bell",
                    entity_idx=0,
                ),
                Success(),
            ],
            "transition_rewards": [1],
        }
    )


def build_handover_block(task):
    target = task.target_box.get_functional_point(1, "pose").p.tolist()
    return Reward.build_top(
        {
            "type": "Serial",
            "subtasks": [
                Pick(base=task, max_reward=4, entity=task.box, dist=0.20),
                Place(base=task, max_reward=4, entity=task.box, target=target, eef_dim=3),
                Success(),
            ],
            "transition_rewards": [1, 3],
        }
    )


def build_lift_pot(task):
    return Reward.build_top(
        {
            "type": "Serial",
            "subtasks": [
                Endpose(
                    base=task,
                    max_reward=4,
                    left_target=task.robot.get_left_tcp_pose(),
                    right_target=task.robot.get_right_tcp_pose(),
                ),
                Success(),
            ],
            "transition_rewards": [1],
        }
    )


def build_move_can_pot(task):
    target = task.target_pose.p.tolist() if hasattr(task.target_pose, "p") else list(task.target_pose)
    return Reward.build_top(
        {
            "type": "Serial",
            "subtasks": [
                Pick(base=task, max_reward=4, entity=task.can, dist=0.20, arm_tag=task.arm_tag),
                Place(base=task, max_reward=4, entity=task.can, target=target, eef_dim=3, arm_tag=task.arm_tag),
                Success(),
            ],
            "transition_rewards": [1, 3],
        }
    )


def build_pick_dual_bottles(task):
    return Reward.build_top(
        {
            "type": "Serial",
            "subtasks": [
                {
                    "type": "Parallel",
                    "subtasks": [
                        {
                            "type": "Serial",
                            "subtasks": [
                                Pick(base=task, max_reward=4, entity=task.bottle1, dist=0.19, arm_tag=0),
                                Place(
                                    base=task,
                                    max_reward=4,
                                    entity=task.bottle1,
                                    target=[task.left_target_pose[0], task.left_target_pose[1], 0.89],
                                    eef_dim=3,
                                    arm_tag=0,
                                ),
                            ],
                            "transition_rewards": [1],
                        },
                        {
                            "type": "Serial",
                            "subtasks": [
                                Pick(base=task, max_reward=4, entity=task.bottle2, dist=0.19, arm_tag=1),
                                Place(
                                    base=task,
                                    max_reward=4,
                                    entity=task.bottle2,
                                    target=[task.right_target_pose[0], task.right_target_pose[1], 0.89],
                                    eef_dim=3,
                                    arm_tag=1,
                                ),
                            ],
                            "transition_rewards": [1],
                        },
                    ],
                    "weights": [1, 1],
                },
                Success(),
            ],
            "transition_rewards": [3],
        }
    )


def build_place_container_plate(task):
    target = task.plate.get_pose().p.tolist()
    return Reward.build_top(
        {
            "type": "Serial",
            "subtasks": [
                Pick(base=task, max_reward=4, entity=task.container, dist=0.20),
                Place(base=task, max_reward=4, entity=task.container, target=target, eef_dim=3),
                Success(),
            ],
            "transition_rewards": [1, 3],
        }
    )


def build_place_empty_cup(task):
    coaster_pose = task.coaster.get_functional_point(0, "pose").p
    target = list(coaster_pose[:2]) + [coaster_pose[2] + 0.015] + [0.5, 0.5, -0.5, -0.5]
    return Reward.build_top(
        {
            "type": "Serial",
            "subtasks": [
                Pick(base=task, max_reward=4, entity=task.cup, dist=0.28),
                Place(base=task, max_reward=4, entity=task.cup, target=target),
                Success(),
            ],
            "transition_rewards": [1, 3],
        }
    )


def build_place_shoe(task):
    return Reward.build_top(
        {
            "type": "Serial",
            "subtasks": [
                Pick(base=task, max_reward=4, entity=task.shoe, dist=0.19),
                Place(
                    base=task,
                    max_reward=4,
                    entity=task.shoe,
                    target=[0, -0.08, 0.84, 0.5, 0.5, -0.5, -0.5],
                ),
                Success(),
            ],
            "transition_rewards": [1, 3],
        }
    )


def build_stack_bowls_three(task):
    return Reward.build_top(
        {
            "type": "Serial",
            "subtasks": [
                Stack(
                    task,
                    max_reward=8.0,
                    entities=[task.bowl1, task.bowl2, task.bowl3],
                    eps=[0.13, 0.03],
                    a_ds=[2, 2],
                    c_ds=[3, 5],
                    target_pose=[0, -0.1],
                ),
                Success(),
            ],
            "transition_rewards": [2.0],
        }
    )


REWARD_BUILDERS: dict[str, RewardBuilder] = {
    "adjust_bottle": build_adjust_bottle,
    "beat_block_hammer": build_beat_block_hammer,
    "click_bell": build_click_bell,
    "handover_block": build_handover_block,
    "lift_pot": build_lift_pot,
    "move_can_pot": build_move_can_pot,
    "pick_dual_bottles": build_pick_dual_bottles,
    "place_container_plate": build_place_container_plate,
    "place_empty_cup": build_place_empty_cup,
    "place_shoe": build_place_shoe,
    "stack_bowls_three": build_stack_bowls_three,
}


def build_task_reward(task_name: str, task: Any):
    try:
        builder = REWARD_BUILDERS[task_name]
    except KeyError as exc:
        raise KeyError(f"No RoboTwin dense reward factory registered for {task_name!r}.") from exc
    reward = builder(task)
    task.reward = reward
    return reward


def supported_reward_tasks() -> tuple[str, ...]:
    return tuple(sorted(REWARD_BUILDERS))
