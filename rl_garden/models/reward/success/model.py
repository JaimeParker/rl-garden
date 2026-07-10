"""Binary success classifier for :class:`~rl_garden.envs.wrappers.reward_classifier.RewardClassifierWrapper`,
architecture-matched to HIL-SERL's own reward classifier (frozen pretrained
ResNet encoder + small MLP head, image-only input -- no proprio/state).

Training entrypoint: :mod:`rl_garden.models.reward.success.train`. Data
collection: :mod:`rl_garden.models.reward.success.collect_data`. Both agree
with this module on the checkpoint format (a plain ``state_dict()`` for
:class:`SuccessClassifier`).
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
from gymnasium import spaces

ClassifierFn = Callable[[dict], torch.Tensor]


class SuccessClassifier(nn.Module):
    """``forward(obs) -> logit`` (raw, pre-sigmoid) -- matches HIL-SERL's
    ``BinaryClassifier`` convention (trained with
    ``sigmoid_binary_cross_entropy``, sigmoid applied by the caller)."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        image_keys: Sequence[str],
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
        pretrained_weights: str | None = "resnet10-imagenet",
    ) -> None:
        super().__init__()
        from rl_garden.encoders.combined import CombinedExtractor
        from rl_garden.encoders.resnet import resnet_encoder_factory

        factory = resnet_encoder_factory(
            pretrained_weights=pretrained_weights,
            freeze_resnet_encoder=True,
            # Matches HIL-SERL's own PreTrainedResNetEncoder (spatial-learned
            # -embedding pooling); also avoids a pre-existing bug in
            # SpatialSoftmax where its pos_x/pos_y buffers are unreduced
            # broadcast views that torch's load_state_dict refuses to copy
            # into on a freshly-constructed instance.
            pooling_method="spatial_learned_embeddings",
        )
        self.encoder = CombinedExtractor(
            observation_space,
            image_keys=image_keys,
            use_proprio=False,
            image_encoder_factory=factory,
            fusion_mode="per_key",
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.features_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        return self.head(self.encoder(obs)).squeeze(-1)


def load_classifier_fn(
    checkpoint_path: str,
    observation_space: spaces.Dict,
    image_keys: Sequence[str],
    device: str | torch.device = "cpu",
) -> ClassifierFn:
    # pretrained_weights=None: the checkpoint's state_dict already contains
    # every parameter (frozen submodules included -- freezing only stops
    # gradient updates, it doesn't exclude a module from state_dict()), so
    # fetching pretrained weights here would be immediately overwritten by
    # load_state_dict below and would needlessly require the pretrained
    # checkpoint file to exist just to load an already-trained classifier.
    model = SuccessClassifier(observation_space, image_keys, pretrained_weights=None)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    def classifier_fn(obs: dict) -> torch.Tensor:
        image_obs = {k: obs[k] for k in image_keys}
        with torch.no_grad():
            logit = model(image_obs)
        return torch.sigmoid(logit)

    return classifier_fn
