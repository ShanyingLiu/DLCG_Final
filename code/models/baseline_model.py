# baseline: predict lighting with only visuals (no material head)

import torch
import torch.nn as nn
import torchvision.models as models


BACKBONE_FEATURES = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


class BaselineLightingNet(nn.Module):
    """Single-task model: ResNet backbone → lighting regression head only."""

    def __init__(self, config):
        super().__init__()
        backbone_name = config.backbone
        feat_dim = BACKBONE_FEATURES[backbone_name]

        # Load pretrained backbone and strip the final FC
        backbone_fn = getattr(models, backbone_name)
        weights = "IMAGENET1K_V1" if config.pretrained else None
        backbone = backbone_fn(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool

        # Lighting head: 27-dim SH coefficient regression
        self.lighting_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, config.sh_dim),
        )

    def forward(self, x):
        feat = self.features(x)          # (B, feat_dim, 1, 1)
        feat = torch.flatten(feat, 1)    # (B, feat_dim)
        lighting = self.lighting_head(feat)
        return lighting
