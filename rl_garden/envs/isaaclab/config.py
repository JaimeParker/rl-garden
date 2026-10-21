from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IsaacLabEnvConfig:
    env_id: str
    num_envs: int
    seed: int
    headless: bool = True
    sim_device: str = "cuda:0"
    # From EnvRequest.observation. IsaacLab tasks bake their own camera
    # set/resolution into the task registration (not runtime-selectable), so
    # this backend only distinguishes "some vision requested" from
    # state-only, plus whether to keep the "state" key when vision is on.
    is_visual: bool = False
    state: bool = True
    frame_stack: int = 1
    env_kwargs: dict = field(default_factory=dict)
    # Hydra-style preset selectors passed to isaaclab_tasks' parse_env_cfg,
    # e.g. ["physics=newton_mjwarp", "renderer=isaacsim_rtx"]. Must stay a
    # list: parse_env_cfg falls back to reading sys.argv when given None.
    overrides: list[str] = field(default_factory=list)
