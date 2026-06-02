from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC
from rl_garden.common.types import ReplayBufferSample
from rl_garden.encoders import BaseFeaturesExtractor
from rl_garden.networks.actor_critic import _PARAM_PREFIX, _safe_name


class DummyVecEnv:
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box) -> None:
        self.num_envs = 1
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = action_space


class TrainableBoxExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: spaces.Box, features_dim: int = 7
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.proj = torch.nn.Linear(int(np.prod(observation_space.shape)), features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(obs.float().flatten(start_dim=1)))


class TrainableDictExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: spaces.Dict, features_dim: int = 7
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.state_proj = torch.nn.Linear(5, features_dim)
        self.rgb_proj = torch.nn.Linear(3, features_dim)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        state = obs["state"].float()
        rgb = obs["rgb"].float().mean(dim=(1, 2)) / 255.0
        return torch.tanh(self.state_proj(state) + self.rgb_proj(rgb))


def _box_env() -> DummyVecEnv:
    return DummyVecEnv(
        spaces.Box(low=-10.0, high=10.0, shape=(5,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )


def _dict_env() -> DummyVecEnv:
    return DummyVecEnv(
        spaces.Dict(
            {
                "state": spaces.Box(low=-10.0, high=10.0, shape=(5,), dtype=np.float32),
                "rgb": spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )


def _make_agent(
    env: DummyVecEnv,
    critic_impl: str,
    extractor_cls: type[BaseFeaturesExtractor],
    *,
    device: str = "cpu",
) -> SAC:
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    return SAC(
        env=env,
        device=device,
        buffer_device=device,
        buffer_size=32,
        batch_size=8,
        learning_starts=0,
        gamma=0.8,
        tau=0.01,
        training_freq=1,
        utd=1.0,
        policy_lr=3e-4,
        q_lr=3e-4,
        ent_coef="auto",
        target_entropy="auto",
        net_arch={"pi": [16, 16], "qf": [16, 16]},
        n_critics=2,
        critic_impl=critic_impl,
        policy_kwargs={
            "features_extractor_class": extractor_cls,
            "features_extractor_kwargs": {"features_dim": 7},
        },
        eval_freq=0,
        save_final_checkpoint=False,
    )


def _legacy_q_state_as_stacked(legacy_critic, dotted: str) -> torch.Tensor:
    return torch.stack([q.state_dict()[dotted] for q in legacy_critic.q_nets], dim=0)


def _assert_critic_close(vmap_critic, legacy_critic, *, atol: float, rtol: float) -> None:
    for dotted in vmap_critic._dotted_param_names:
        vmap_value = getattr(vmap_critic, _safe_name(dotted, _PARAM_PREFIX)).detach()
        legacy_value = _legacy_q_state_as_stacked(legacy_critic, dotted).to(vmap_value.device)
        torch.testing.assert_close(vmap_value, legacy_value, atol=atol, rtol=rtol)


def _assert_module_params_close(
    lhs: torch.nn.Module, rhs: torch.nn.Module, *, atol: float, rtol: float
) -> None:
    for lp, rp in zip(lhs.parameters(), rhs.parameters()):
        torch.testing.assert_close(lp.detach(), rp.detach(), atol=atol, rtol=rtol)


def _assert_agents_close(vmap: SAC, legacy: SAC, *, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    _assert_module_params_close(
        vmap.policy.features_extractor,
        legacy.policy.features_extractor,
        atol=atol,
        rtol=rtol,
    )
    _assert_module_params_close(vmap.policy.actor, legacy.policy.actor, atol=atol, rtol=rtol)
    _assert_critic_close(vmap.policy.critic, legacy.policy.critic, atol=atol, rtol=rtol)
    _assert_critic_close(
        vmap.policy.critic_target,
        legacy.policy.critic_target,
        atol=atol,
        rtol=rtol,
    )
    if vmap.log_alpha is not None:
        torch.testing.assert_close(vmap.log_alpha, legacy.log_alpha, atol=atol, rtol=rtol)


def _box_batch(device: str) -> ReplayBufferSample:
    gen = torch.Generator(device=device)
    gen.manual_seed(456)
    obs = torch.randn(8, 5, generator=gen, device=device)
    next_obs = torch.randn(8, 5, generator=gen, device=device)
    actions = torch.randn(8, 2, generator=gen, device=device).clamp(-0.9, 0.9)
    rewards = torch.randn(8, generator=gen, device=device)
    dones = torch.randint(0, 2, (8,), generator=gen, device=device).float()
    return ReplayBufferSample(obs, next_obs, actions, rewards, dones)


def _dict_batch(device: str) -> ReplayBufferSample:
    gen = torch.Generator(device=device)
    gen.manual_seed(789)
    obs = {
        "state": torch.randn(8, 5, generator=gen, device=device),
        "rgb": torch.randint(0, 256, (8, 8, 8, 3), generator=gen, device=device, dtype=torch.uint8),
    }
    next_obs = {
        "state": torch.randn(8, 5, generator=gen, device=device),
        "rgb": torch.randint(0, 256, (8, 8, 8, 3), generator=gen, device=device, dtype=torch.uint8),
    }
    actions = torch.randn(8, 2, generator=gen, device=device).clamp(-0.9, 0.9)
    rewards = torch.randn(8, generator=gen, device=device)
    dones = torch.randint(0, 2, (8,), generator=gen, device=device).float()
    return ReplayBufferSample(obs, next_obs, actions, rewards, dones)


def _install_fixed_sampler(agent: SAC, batch: ReplayBufferSample) -> None:
    def _sample_train_batch(self, batch_size: int) -> ReplayBufferSample:
        assert batch_size == 8
        return batch

    agent._sample_train_batch = types.MethodType(_sample_train_batch, agent)


def _get_cuda_rng(device: str) -> torch.Tensor | None:
    if not str(device).startswith("cuda"):
        return None
    return torch.cuda.get_rng_state(torch.device(device))


def _set_cuda_rng(device: str, rng: torch.Tensor | None) -> None:
    if rng is not None:
        torch.cuda.set_rng_state(rng, torch.device(device))


def _run_matched_train_step(
    vmap: SAC,
    legacy: SAC,
    cpu_rng: torch.Tensor,
    cuda_rng: torch.Tensor | None,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    torch.set_rng_state(cpu_rng)
    _set_cuda_rng(device, cuda_rng)
    legacy_info = legacy.train(1, compute_info=True)
    legacy_rng_after = torch.get_rng_state().clone()
    legacy_cuda_rng_after = _get_cuda_rng(device)

    torch.set_rng_state(cpu_rng)
    _set_cuda_rng(device, cuda_rng)
    vmap_info = vmap.train(1, compute_info=True)
    vmap_rng_after = torch.get_rng_state().clone()
    vmap_cuda_rng_after = _get_cuda_rng(device)

    assert legacy_info.keys() == vmap_info.keys()
    for key in legacy_info:
        assert legacy_info[key] == pytest.approx(vmap_info[key], abs=1e-6, rel=1e-6)
    assert torch.equal(legacy_rng_after, vmap_rng_after)
    if legacy_cuda_rng_after is not None:
        assert vmap_cuda_rng_after is not None
        assert torch.equal(legacy_cuda_rng_after, vmap_cuda_rng_after)
    return legacy_rng_after, legacy_cuda_rng_after


@pytest.mark.parametrize(
    "env_factory,extractor_cls",
    [(_box_env, TrainableBoxExtractor), (_dict_env, TrainableDictExtractor)],
)
def test_sac_policy_init_vmap_matches_legacy(env_factory, extractor_cls):
    legacy = _make_agent(env_factory(), "legacy", extractor_cls)
    legacy_rng = torch.get_rng_state().clone()
    legacy_sample = torch.randint(0, 100000, (16,))

    vmap = _make_agent(env_factory(), "vmap", extractor_cls)
    vmap_rng = torch.get_rng_state().clone()
    vmap_sample = torch.randint(0, 100000, (16,))

    assert torch.equal(vmap_rng, legacy_rng)
    assert torch.equal(vmap_sample, legacy_sample)
    _assert_agents_close(vmap, legacy)


@pytest.mark.parametrize(
    "env_factory,extractor_cls,batch_factory",
    [
        (_box_env, TrainableBoxExtractor, _box_batch),
        (_dict_env, TrainableDictExtractor, _dict_batch),
    ],
)
def test_sac_update_vmap_matches_legacy_on_fixed_batch(
    env_factory, extractor_cls, batch_factory
):
    legacy = _make_agent(env_factory(), "legacy", extractor_cls)
    vmap = _make_agent(env_factory(), "vmap", extractor_cls)
    _install_fixed_sampler(legacy, batch_factory("cpu"))
    _install_fixed_sampler(vmap, batch_factory("cpu"))

    _assert_agents_close(vmap, legacy)
    cpu_rng = torch.manual_seed(2026).get_state()
    cuda_rng = None
    for _ in range(8):
        cpu_rng, cuda_rng = _run_matched_train_step(
            vmap, legacy, cpu_rng, cuda_rng, device="cpu"
        )
        _assert_agents_close(vmap, legacy, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sac_update_vmap_matches_legacy_on_fixed_batch_cuda():
    legacy = _make_agent(_box_env(), "legacy", TrainableBoxExtractor, device="cuda")
    vmap = _make_agent(_box_env(), "vmap", TrainableBoxExtractor, device="cuda")
    _install_fixed_sampler(legacy, _box_batch("cuda"))
    _install_fixed_sampler(vmap, _box_batch("cuda"))

    _assert_agents_close(vmap, legacy)
    cpu_rng = torch.manual_seed(2026).get_state()
    torch.cuda.manual_seed_all(2026)
    cuda_rng = _get_cuda_rng("cuda")
    for _ in range(8):
        cpu_rng, cuda_rng = _run_matched_train_step(
            vmap, legacy, cpu_rng, cuda_rng, device="cuda"
        )
        _assert_agents_close(vmap, legacy, atol=2e-6, rtol=2e-6)


def test_alpha_min_clamps_current_alpha():
    """alpha_min prevents _current_alpha() from going below the floor."""
    agent = _make_agent(_box_env(), "vmap", TrainableBoxExtractor)
    agent.alpha_min = 1e-4

    # Drive log_alpha far below alpha_min.
    agent.log_alpha.data.fill_(-20.0)  # exp(-20) ≈ 2e-9

    alpha = agent._current_alpha()
    # Compare against the float32 value of 1e-4 to avoid platform precision issues.
    assert alpha.detach().item() >= torch.tensor(1e-4).item() - 1e-10, (
        f"Expected alpha >= 1e-4, got {alpha.item()}"
    )

    # Default alpha_min=0.0 must not clamp.
    agent_no_min = _make_agent(_box_env(), "vmap", TrainableBoxExtractor)
    agent_no_min.log_alpha.data.fill_(-20.0)
    alpha_no_min = agent_no_min._current_alpha()
    assert alpha_no_min.detach().item() < 1e-4, "alpha_min=0 should not clamp"
