"""ResNet-based binary classifier model."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResNetBinaryClassifier(nn.Module):
    """ResNet-based binary classifier."""

    def __init__(
        self,
        resnet_type: str = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
        from torchvision.models import resnet18, resnet34, resnet50

        model_map = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}
        weights_map = {
            "resnet18": ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            "resnet34": ResNet34_Weights.IMAGENET1K_V1 if pretrained else None,
            "resnet50": ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
        }

        if resnet_type not in model_map:
            raise ValueError(f"Unsupported resnet_type: {resnet_type}")

        backbone = model_map[resnet_type](weights=weights_map.get(resnet_type))
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.classifier(x))
