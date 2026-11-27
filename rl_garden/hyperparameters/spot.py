"""SPOT learning hyperparameters, shared by constructors and training Args."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rl_garden.hyperparameters.td3_bc import TD3CoreParams


@dataclass
class SPOTParams(TD3CoreParams):
    """Defaults matching the SPOT learning core."""

    # Actor learning rate.
    actor_lr: float = 1e-4

    # Behavior VAE learning rate.
    vae_lr: float = 1e-3

    # Behavior VAE hidden layer width.
    vae_hidden_dim: int = 750

    # Behavior VAE latent width; None selects twice the action dimension.
    vae_latent_dim: Optional[int] = None

    # Number of behavior VAE pretraining updates.
    vae_iterations: int = 100_000

    # Behavior VAE KL penalty coefficient.
    beta: float = 0.5

    # Support constraint weight.
    lambd: float = 1.0

    # Number of samples for the support constraint estimate.
    num_samples: int = 1

    # Use the importance-weighted VAE likelihood estimate.
    iwae: bool = False

    # Cool the support constraint during online updates.
    lambd_cool: bool = False

    # Minimum online support constraint weight fraction.
    lambd_end: float = 0.2

    # Online exploration noise standard deviation.
    expl_noise: float = 0.1


@dataclass
class Off2OnSPOTParams(SPOTParams):
    """SPOT rollout-shell hyperparameters."""

    # Discount applied at the offline-to-online switch.
    online_discount: float = 0.995
