import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Combined lighting (SH MSE) + material parameter regression (MSE) loss."""

    def __init__(self, config):
        super().__init__()
        self.lighting_weight = config.lighting_loss_weight
        self.material_weight = config.material_loss_weight
        self.lighting_loss = nn.MSELoss()
        self.material_loss = nn.MSELoss()

    def forward(self, pred_lighting, pred_material,
                target_lighting, target_material):
        loss_lighting = self.lighting_loss(pred_lighting, target_lighting)
        loss_material = self.material_loss(pred_material, target_material)

        total = (self.lighting_weight * loss_lighting +
                 self.material_weight * loss_material)

        return total, loss_lighting, loss_material
