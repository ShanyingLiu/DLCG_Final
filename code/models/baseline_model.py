# baseline: predict envmap with only visuals (no material head)

import torch.nn as nn
import torchvision.models as models

from models.envmap_decoder import EnvmapDecoder


BACKBONE_FEATURES = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


class BaselineLightingNet(nn.Module):
    """Single-task model: ResNet backbone -> EnvmapDecoder -> HDR envmap."""

    def __init__(self, config):
        super().__init__()
        backbone_name = config.backbone
        feat_dim = BACKBONE_FEATURES[backbone_name]

        backbone_fn = getattr(models, backbone_name)
        weights = "IMAGENET1K_V1" if config.pretrained else None
        backbone = backbone_fn(weights=weights)
        # Strip avgpool + fc: keep spatial feats from layer4 (4x4 @ image=128).
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        if config.pretrained:
            for name, param in self.features.named_parameters():
                if not name.startswith('7'):  # layer4 is child index 7
                    param.requires_grad = False

        self.decoder = EnvmapDecoder(feat_dim)

    def forward(self, x):
        feat = self.features(x)        # (B, feat_dim, 4, 4)
        return self.decoder(feat)      # (B, 3, 64, 128)
