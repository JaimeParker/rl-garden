"""Version-neutral task logic mixins.

Each mixin implements a task's observations/rewards/dones/reset jitter once,
written only against the accessors both ``RLGardenDirectRLEnv`` scaffolds
(``v2.direct_env``/``v3.direct_env``) provide: ``_joint_pos()``,
``_joint_vel()``, ``_apply_joint_effort(target, joint_ids)``,
``_camera_rgb()`` and the ``_sample_reset_state`` hook. Modules here import
only torch/math, never ``isaaclab``.
"""
