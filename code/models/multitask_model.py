# 2-head model predicting material and lighting

import torch
import torch.nn as nn
import torchvision.models as models


BACKBONE_FEATURES = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


class MaterialAwareLightingNet(nn.Module):
    """Multi-task model: shared ResNet backbone → lighting head + material head."""

    def __init__(self, config):
        super().__init__()
        backbone_name = config.backbone
        feat_dim = BACKBONE_FEATURES[backbone_name]

        # Load pretrained backbone and strip the final FC
        backbone_fn = getattr(models, backbone_name)
        weights = "IMAGENET1K_V1" if config.pretrained else None
        backbone = backbone_fn(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool

        # Freeze backbone except the last block (layer4)
        if config.pretrained:
            for name, param in self.features.named_parameters():
                if not name.startswith('7'):  # layer4 is child index 7
                    param.requires_grad = False

        # Material branch: hidden features used for both classification and conditioning
        self.material_encoder = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
        )
        self.material_dropout = nn.Dropout(0.3)
        self.material_classifier = nn.Linear(128, config.num_material_classes)

        # Lighting head: conditioned on material features
        self.lighting_head = nn.Sequential(
            nn.Linear(feat_dim + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, config.sh_dim),
        )

    def forward(self, x):
        feat = self.features(x)          # (B, feat_dim, 1, 1)
        feat = torch.flatten(feat, 1)    # (B, feat_dim)

        # Material branch
        mat_features = self.material_encoder(feat)          # (B, 128)
        material = self.material_classifier(self.material_dropout(mat_features))

        # Lighting branch conditioned on material features
        lighting_input = torch.cat([feat, mat_features], dim=1)
        lighting = self.lighting_head(lighting_input)

        return lighting, material
