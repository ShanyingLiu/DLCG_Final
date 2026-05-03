import torch
import torch.nn as nn
import torch.nn.functional as F


def log_hdr_mse(pred, target, eps: float = 1.0):
    """MSE in log-HDR space: ((log(pred+eps) - log(target+eps))^2).mean().

    eps=1.0 yields effectively log1p, which is well-behaved across HDR
    range [0, ~10k] and at exactly zero. pred is assumed non-negative
    (e.g. Softplus output); target is non-negative HDR irradiance.
    """
    return F.mse_loss(torch.log(pred + eps), torch.log(target + eps))


class MultiTaskLoss(nn.Module):
    """Combined lighting (log-HDR MSE) + material parameter regression (MSE)."""

    def __init__(self, config):
        super().__init__()
        self.lighting_weight = config.lighting_loss_weight
        self.material_weight = config.material_loss_weight
        self.log_eps = float(getattr(config, "log_eps", 1.0))
        self.material_loss = nn.MSELoss()

    def forward(self, pred_lighting, pred_material,
                target_lighting, target_material):
        loss_lighting = log_hdr_mse(pred_lighting, target_lighting,
                                    eps=self.log_eps)
        loss_material = self.material_loss(pred_material, target_material)

        total = (self.lighting_weight * loss_lighting +
                 self.material_weight * loss_material)
        return total, loss_lighting, loss_material
