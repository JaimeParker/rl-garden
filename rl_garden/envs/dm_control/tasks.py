"""NCRL's DMC task set (NCRL reference implementation, ``src/envs.py::_DMControl_TASK_SET``).

Copied verbatim (without NCRL's ``dmc-`` prefix, which ``--env_id`` also
accepts). A ``-hard`` suffix selects the hard variant (see
``rl_garden.envs.dm_control.env.hard_task_reward``) only for the NCRL tasks
listed below that end in ``-hard``; any other ``*-hard`` name is a genuine
``dm_control.suite`` task (``reacher-hard`` -> ``reacher/hard``,
``finger-turn-hard`` -> ``finger/turn_hard``).
"""
from __future__ import annotations

# 22 dm control tasks in total, 5 embodiments
NCRL_DMC_TASKS = (
    'cartpole-balance', # 1
    'acrobot-swingup', 'acrobot-swingup-sparse', 'acrobot-swingup-hard', # 3
    'cheetah-run', 'cheetah-run-backwards', 'cheetah-jump', 'cheetah-run-back', 'cheetah-run-front', # 5
    'walker-stand', 'walker-walk', 'walker-walk-hard', 'walker-run', 'walker-walk-backwards', 'walker-run-backwards', 'walker-backflip', # 7
    'quadruped-stand', 'quadruped-walk', 'quadruped-run', 'quadruped-jump', 'quadruped-roll', 'quadruped-roll-fast', # 6
)

# The NCRL tasks registered by ``rl_garden.envs.dm_control.custom_tasks``
# rather than shipped by ``dm_control.suite`` itself.
NCRL_CUSTOM_DMC_TASKS = (
    'cheetah-run-backwards', 'cheetah-jump', 'cheetah-run-back', 'cheetah-run-front',
    'walker-walk-backwards', 'walker-run-backwards', 'walker-backflip',
    'quadruped-stand', 'quadruped-jump', 'quadruped-roll', 'quadruped-roll-fast',
)

_DOMAIN_ALIASES = {"cup": "ball_in_cup", "point": "point_mass"}
_HARD_SUFFIX = "-hard"
_NCRL_HARD_TASKS = frozenset(t for t in NCRL_DMC_TASKS if t.endswith(_HARD_SUFFIX))
_NCRL_PREFIX = "dmc-"


def parse_task_id(task_id: str) -> tuple[str, str, bool]:
    """``"walker-walk-hard"`` -> ``("walker", "walk", True)``.

    Same parsing as NCRL's ``DMControl.__init__``: the first ``-`` splits
    domain from task, remaining dashes become underscores
    (``"walker-run-backwards"`` -> ``("walker", "run_backwards", False)``).
    An NCRL-style ``dmc-`` prefix is accepted and dropped. ``-hard`` is only
    stripped (``hard=True``) for the NCRL hard tasks; ``reacher-hard`` stays
    ``("reacher", "hard", False)``. Use ``--dm_control.hard`` to wrap any task.
    """
    name = task_id[len(_NCRL_PREFIX):] if task_id.startswith(_NCRL_PREFIX) else task_id
    hard = name in _NCRL_HARD_TASKS
    if hard:
        name = name[: -len(_HARD_SUFFIX)]
    if "-" not in name:
        raise ValueError(
            f"dm_control: env_id {task_id!r} must look like '<domain>-<task>' "
            "(e.g. 'walker-walk', 'cheetah-run-backwards', 'walker-walk-hard')."
        )
    domain, task = name.split("-", 1)
    return _DOMAIN_ALIASES.get(domain, domain), task.replace("-", "_"), hard
