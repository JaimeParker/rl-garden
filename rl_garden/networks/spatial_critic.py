"""Spatial Q-critic for token-structured features.

These modules consume flat feature vectors whose layout is declared via
``BaseFeaturesExtractor.structured_feature_config()``.  They reconstruct the
token grid internally from the flat representation, so they are independent
of any specific encoder class.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.networks.mlp import KernelInit, create_mlp


class SpatialEmbQHead(nn.Module):
    """Single spatial Q-head operating on token-structured features.

    Parameters
    ----------
    num_patches:
        Total number of ViT / token patches concatenated in the flat feature.
    patch_dim:
        Embedding dimension per patch.
    prop_dim:
        Proprioceptive / state dimension appended after the token block.
    action_dim:
        Dimension of the action vector.
    hidden_dims:
        Hidden layer sizes for the final Q MLP.
    spatial_emb_dim:
        Dimension of the spatial projection.  Set to 0 to skip spatial
        projection and fall back to flattened-token + prop + action input.
    use_layer_norm:
        Apply LayerNorm inside the spatial projection.
    kernel_init:
        Optional weight initializer for the Q MLP.
    """

    def __init__(
        self,
        num_patches: int,
        patch_dim: int,
        prop_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        *,
        spatial_emb_dim: int = 1024,
        use_layer_norm: bool = True,
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.spatial_emb_dim = spatial_emb_dim
        self.token_dim = num_patches * patch_dim

        if spatial_emb_dim > 0:
            # Input to spatial_proj: for each of the patch_dim "channels",
            # concatenate: that channel's value across patches (num_patches),
            # plus prop_dim, plus action_dim.
            token_channel_input_dim = num_patches + prop_dim + action_dim
            self.spatial_proj = nn.Sequential(
                nn.Linear(token_channel_input_dim, spatial_emb_dim),
                nn.LayerNorm(spatial_emb_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
            )
            self.spatial_weight = nn.Parameter(
                torch.empty(1, patch_dim, spatial_emb_dim)
            )
            nn.init.xavier_uniform_(self.spatial_weight)
            head_input_dim = spatial_emb_dim + prop_dim + action_dim
        else:
            self.spatial_proj = None
            self.spatial_weight = None
            head_input_dim = self.token_dim + prop_dim + action_dim

        self.q = create_mlp(
            head_input_dim,
            1,
            hidden_dims,
            use_layer_norm=use_layer_norm,
            kernel_init=kernel_init,
        )

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Reconstruct token grid and prop from the flat feature vector.
        tokens = features[:, : self.token_dim].reshape(
            features.shape[0], self.num_patches, self.patch_dim
        )
        prop = features[:, self.token_dim :]

        if self.spatial_proj is None:
            z = torch.cat([tokens.flatten(1), prop, actions], dim=-1)
            return self.q(z)

        b, _p, d = tokens.shape
        # token_channels: (B, patch_dim, num_patches) — transpose patch/channel axes
        token_channels = tokens.transpose(1, 2)
        action_rep = actions.unsqueeze(1).expand(b, d, self.action_dim)
        if self.prop_dim:
            prop_rep = prop.unsqueeze(1).expand(b, d, self.prop_dim)
            spatial_input = torch.cat([token_channels, prop_rep, action_rep], dim=-1)
        else:
            spatial_input = torch.cat([token_channels, action_rep], dim=-1)

        y = self.spatial_proj(spatial_input)
        assert self.spatial_weight is not None
        z = (self.spatial_weight * y).sum(dim=1)
        return self.q(torch.cat([z, prop, actions], dim=-1))


class SpatialEmbQEnsemble(nn.Module):
    """Ensemble of ``SpatialEmbQHead`` critics.

    Validates that the declared layout is consistent with ``features_dim`` at
    construction time so misconfiguration is caught early.
    """

    def __init__(
        self,
        num_patches: int,
        patch_dim: int,
        prop_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int],
        *,
        n_critics: int = 2,
        spatial_emb_dim: int = 1024,
        use_layer_norm: bool = True,
        kernel_init: Optional[KernelInit] = None,
        features_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if features_dim is not None:
            expected = num_patches * patch_dim + prop_dim
            if features_dim != expected:
                raise ValueError(
                    f"SpatialEmbQEnsemble: features_dim={features_dim} does not match "
                    f"num_patches*patch_dim + prop_dim = {expected}."
                )
        action_dim = int(np.prod(action_space.shape))
        self.n_critics = n_critics
        self.q_nets = nn.ModuleList(
            [
                SpatialEmbQHead(
                    num_patches=num_patches,
                    patch_dim=patch_dim,
                    prop_dim=prop_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    spatial_emb_dim=spatial_emb_dim,
                    use_layer_norm=use_layer_norm,
                    kernel_init=kernel_init,
                )
                for _ in range(n_critics)
            ]
        )

    def forward(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return tuple(q(features, actions) for q in self.q_nets)

    def forward_all(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack([q(features, actions) for q in self.q_nets], dim=0)
