# 2-head model predicting material parameters and envmap, with material
# features tiled into the lighting decoder's spatial input.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.envmap_decoder import EnvmapDecoder


BACKBONE_FEATURES = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}

MATERIAL_HIDDEN = 128


class MaterialAwareLightingNet(nn.Module):
    """Multi-task model: shared spatial features -> material regression head
    + envmap decoder. Material hidden vector (B, 128) is tiled across the
    4x4 spatial grid and concatenated with backbone feats before decoding."""

    def __init__(self, config):
        super().__init__()
        backbone_name = config.backbone
        feat_dim = BACKBONE_FEATURES[backbone_name]

        backbone_fn = getattr(models, backbone_name)
        weights = "IMAGENET1K_V1" if config.pretrained else None
        backbone = backbone_fn(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        if config.pretrained:
            for name, param in self.features.named_parameters():
                if not name.startswith('7'):
                    param.requires_grad = False

        self.material_encoder = nn.Sequential(
            nn.Linear(feat_dim, MATERIAL_HIDDEN),
            nn.ReLU(inplace=True),
        )
        self.material_dropout = nn.Dropout(0.3)
        self.material_regressor = nn.Linear(MATERIAL_HIDDEN, config.num_material_params)

        self.decoder = EnvmapDecoder(feat_dim + MATERIAL_HIDDEN)

    def forward(self, x):
        feat = self.features(x)                              # (B, C, 4, 4)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)   # (B, C)

        mat_features = self.material_encoder(pooled)         # (B, 128)
        material = torch.sigmoid(
            self.material_regressor(self.material_dropout(mat_features))
        )

        # Tile material features across the spatial grid and concat.
        B, _, H, W = feat.shape
        mat_tiled = mat_features.view(B, MATERIAL_HIDDEN, 1, 1).expand(
            B, MATERIAL_HIDDEN, H, W
        )
        decoder_in = torch.cat([feat, mat_tiled], dim=1)     # (B, C+128, 4, 4)

        envmap = self.decoder(decoder_in)                    # (B, 3, 64, 128)
        return envmap, material
