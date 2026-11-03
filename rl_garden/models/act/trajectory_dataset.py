"""ACT training dataset over lists of transition dicts with dict observations.

A trajectory source is a list of transition dicts in time order, the format of pickled
demo lists: ``obs`` (a dict observation, or a bare state tensor), ``action``, ``done`` and,
optionally, ``next_obs`` and ``truncated``. Every value may carry a leading batch-of-1 axis
(what a single-env vector env returns). Observations follow the keys
``ACTBaseActionProvider`` reads at inference: ``state`` (``(D,)`` float) and, for a visual
policy, one ``rgb_<camera>`` image per camera (``(H, W, 3)`` uint8, the same size for all
cameras).

Episodes end at a transition whose ``done`` or ``truncated`` is set, and, unless disabled, where
a transition's ``next_obs`` differs from the following transition's ``obs``: recorders that
keep only some steps, or end an episode by truncation without storing it, leave ``done`` unset
at such a boundary.

Everything is moved to ``device`` once at construction; ``get_batch``/``sample`` index those
tensors with no host round trip.
"""
from __future__ import annotations

import glob
import pickle
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Optional

import torch

STATE_KEY = "state"
CAMERA_KEY_PREFIX = "rgb_"
# A channel-stacked image key ACTBaseActionProvider prefers over per-camera keys at inference.
STACKED_RGB_KEY = "rgb"
MIN_NORM_STD = 1e-2


class ACTBatch(NamedTuple):
    """One ACT training batch; ``state`` and ``action_chunk`` are normalised by the dataset's
    ``norm_stats``."""

    state: torch.Tensor  # (B, D) float32
    images: Optional[torch.Tensor]  # (B, num_cameras, 3, H, W) uint8, None for state-only
    action_chunk: torch.Tensor  # (B, num_queries, A) float32
    is_pad: torch.Tensor  # (B, num_queries) bool, True past the episode end


def load_transition_pkls(patterns: Sequence[str]) -> list[dict]:
    """Concatenate the pickled transition lists of every file matching ``patterns`` (globs,
    each expanded in sorted order). Raises ``FileNotFoundError`` when nothing matches."""

    paths = [path for pattern in patterns for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"no transition pkl matches {list(patterns)}")
    transitions: list[dict] = []
    for path in paths:
        with open(path, "rb") as stream:
            loaded = pickle.load(stream)
        if not isinstance(loaded, list):
            raise TypeError(f"{path} holds a {type(loaded).__name__}, expected a list of transition dicts")
        transitions.extend(loaded)
    return transitions


def _as_dict_observation(obs: Any) -> Mapping[str, Any]:
    return obs if isinstance(obs, Mapping) else {STATE_KEY: obs}


def _single_step(value: Any, *, ndim: int, name: str) -> torch.Tensor:
    """``value`` without its leading batch-of-1 axis, checked to have ``ndim`` dims."""

    tensor = torch.as_tensor(value)
    if tensor.ndim == ndim + 1 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dims (optionally with a leading batch of 1); "
            f"got shape {tuple(tensor.shape)}"
        )
    return tensor


def _flag(transition: Mapping[str, Any], key: str) -> bool:
    value = transition.get(key)
    return value is not None and bool(torch.as_tensor(value).any())


def _observations_equal(first: Any, second: Any) -> bool:
    first, second = _as_dict_observation(first), _as_dict_observation(second)
    if first.keys() != second.keys():
        return False
    return all(
        torch.equal(torch.as_tensor(first[key]).cpu(), torch.as_tensor(second[key]).cpu())
        for key in first
    )


def episode_end_indices(
    transitions: Sequence[Mapping[str, Any]], *, split_on_observation_gaps: bool = True
) -> list[int]:
    """One-past-last transition index of every episode, in order (the last is ``len``)."""

    ends: list[int] = []
    for index, transition in enumerate(transitions):
        is_last = index == len(transitions) - 1
        ends_here = is_last or _flag(transition, "done") or _flag(transition, "truncated")
        if not ends_here and split_on_observation_gaps and "next_obs" in transition:
            ends_here = not _observations_equal(transition["next_obs"], transitions[index + 1]["obs"])
        if ends_here:
            ends.append(index + 1)
    return ends


def _camera_keys(obs: Mapping[str, Any]) -> tuple[str, ...]:
    """Per-camera keys in ``ACTBaseActionProvider``'s camera order (sorted)."""

    return tuple(sorted(key for key in obs if key.startswith(CAMERA_KEY_PREFIX)))


def _norm_mean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0, keepdim=True)
    if values.shape[0] > 1:
        std = values.std(dim=0, keepdim=True)
    else:
        std = torch.ones_like(mean)
    return mean, std.clamp_min(MIN_NORM_STD)


class ACTTrajectoryDataset:
    """Action-chunk samples over a list of transition dicts, for ACT training.

    Sample ``t`` pairs the observation at ``t`` with the ``num_queries`` actions from ``t``
    within its episode; past the episode end the chunk repeats the last action and ``is_pad``
    marks those entries (the ACT loss ignores them). ``norm_stats`` holds
    ``state_mean``/``state_std`` ``(1, D)`` and ``action_mean``/``action_std`` ``(1, A)``
    (std clamped to ``MIN_NORM_STD``), the layout ``ACTBaseActionProvider`` reads from a
    checkpoint's ``norm_stats``.
    """

    def __init__(
        self,
        transitions: Sequence[Mapping[str, Any]],
        *,
        num_queries: int,
        device: torch.device | str = "cpu",
        use_cameras: bool = True,
        split_on_observation_gaps: bool = True,
    ) -> None:
        if not transitions:
            raise ValueError("ACTTrajectoryDataset needs at least one transition")
        if num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}")
        self.num_queries = int(num_queries)
        self.device = torch.device(device)

        observations = [_as_dict_observation(transition["obs"]) for transition in transitions]
        first_obs = observations[0]
        if STATE_KEY not in first_obs:
            raise ValueError(f"observations need a {STATE_KEY!r} key; got keys {sorted(first_obs)}")
        self.camera_keys: tuple[str, ...] = _camera_keys(first_obs) if use_cameras else ()
        if self.camera_keys and STACKED_RGB_KEY in first_obs:
            raise ValueError(
                f"observations hold both {STACKED_RGB_KEY!r} and {list(self.camera_keys)}; "
                f"ACTBaseActionProvider would read {STACKED_RGB_KEY!r} at inference, so keep "
                "only the per-camera keys"
            )

        states = torch.stack(
            [
                _single_step(obs[STATE_KEY], ndim=1, name=STATE_KEY).to(self.device, torch.float32)
                for obs in observations
            ]
        )
        actions = torch.stack(
            [
                _single_step(transition["action"], ndim=1, name="action").to(self.device, torch.float32)
                for transition in transitions
            ]
        )
        self.images: Optional[torch.Tensor] = self._stack_images(observations) if self.camera_keys else None

        ends = episode_end_indices(transitions, split_on_observation_gaps=split_on_observation_gaps)
        starts = [0, *ends[:-1]]
        self.episode_lengths: list[int] = [end - start for start, end in zip(starts, ends)]
        self._episode_end = torch.cat(
            [torch.full((end - start,), end, dtype=torch.long) for start, end in zip(starts, ends)]
        ).to(self.device)
        self._chunk_offsets = torch.arange(self.num_queries, device=self.device)

        state_mean, state_std = _norm_mean_std(states)
        action_mean, action_std = _norm_mean_std(actions)
        self._norm_stats = {
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
        }
        self.states = (states - state_mean) / state_std
        self.actions = (actions - action_mean) / action_std

    def _stack_images(self, observations: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        per_camera = []
        for key in self.camera_keys:
            frames = []
            for obs in observations:
                if key not in obs:
                    raise ValueError(f"an observation lacks camera key {key!r}; every step needs all cameras")
                frame = _single_step(obs[key], ndim=3, name=key)
                if frame.shape[-1] != 3 or frame.dtype != torch.uint8:
                    raise ValueError(
                        f"{key} must be (H, W, 3) uint8; got shape {tuple(frame.shape)} {frame.dtype}"
                    )
                frames.append(frame.to(self.device))
            per_camera.append(torch.stack(frames).permute(0, 3, 1, 2))
        sizes = {tuple(camera.shape[-2:]) for camera in per_camera}
        if len(sizes) != 1:
            raise ValueError(f"all cameras must share one image size; got {sorted(sizes)}")
        return torch.stack(per_camera, dim=1).contiguous()

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def num_episodes(self) -> int:
        return len(self.episode_lengths)

    @property
    def norm_stats(self) -> dict[str, torch.Tensor]:
        """CPU copies of the normalisation statistics, for a checkpoint."""

        return {key: value.detach().cpu().clone() for key, value in self._norm_stats.items()}

    def get_batch(self, indices: torch.Tensor) -> ACTBatch:
        indices = indices.to(self.device, torch.long)
        episode_end = self._episode_end[indices].unsqueeze(1)
        chunk_indices = indices.unsqueeze(1) + self._chunk_offsets
        is_pad = chunk_indices >= episode_end
        chunk_indices = torch.minimum(chunk_indices, episode_end - 1)
        return ACTBatch(
            state=self.states[indices],
            images=None if self.images is None else self.images[indices],
            action_chunk=self.actions[chunk_indices],
            is_pad=is_pad,
        )

    def sample(self, batch_size: int, *, generator: Optional[torch.Generator] = None) -> ACTBatch:
        """A batch of ``batch_size`` steps drawn uniformly with replacement."""

        indices = torch.randint(
            len(self), (batch_size,), device=self.device, generator=generator
        )
        return self.get_batch(indices)
