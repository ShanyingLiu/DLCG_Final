import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _peak_weight(target, peak_lambda: float):
    """Per-pixel weight = 1 + lambda * log1p(target). Sun pixels (target~1e3)
    weigh ~7-9x more than dark sky (target~0), so MSE/L1 gradients are not
    dominated by the 99% of low-radiance pixels.
    """
    if peak_lambda == 0.0:
        return None
    return 1.0 + peak_lambda * torch.log1p(target.clamp_min(0))


def log_hdr_mse(pred, target, eps: float = 1.0, peak_lambda: float = 0.0):
    """MSE in log-HDR space: ((log(pred+eps) - log(target+eps))^2).mean().

    eps=1.0 yields effectively log1p, which is well-behaved across HDR
    range [0, ~10k] and at exactly zero. pred is assumed non-negative
    (e.g. Softplus output); target is non-negative HDR irradiance.

    peak_lambda > 0 applies a per-pixel weight that emphasizes bright pixels
    (see `_peak_weight`).
    """
    sq = (torch.log(pred + eps) - torch.log(target + eps)).pow(2)
    w = _peak_weight(target, peak_lambda)
    if w is None:
        return sq.mean()
    return (sq * w).sum() / w.sum().clamp_min(1e-8)


def log_hdr_l1(pred, target, eps: float = 1.0, peak_lambda: float = 0.0):
    """L1 in log-HDR space. Less sensitive to bright outliers (e.g. sun pixels)
    than MSE, so it doesn't blur the prediction toward the mean as aggressively.

    peak_lambda > 0 applies a per-pixel weight that emphasizes bright pixels.
    """
    abs_d = (torch.log(pred + eps) - torch.log(target + eps)).abs()
    w = _peak_weight(target, peak_lambda)
    if w is None:
        return abs_d.mean()
    return (abs_d * w).sum() / w.sum().clamp_min(1e-8)


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords -= (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return g


def ssim(pred, target, window_size: int = 11, sigma: float = 1.5,
         data_range: float = 1.0):
    """Mean SSIM over a (B, C, H, W) batch using a 2D Gaussian window.

    Returns a scalar in [0, 1] (1 = identical). Loss should be `1 - ssim`.
    Inputs are expected to live on roughly the same scale as `data_range`.
    """
    if pred.dim() != 4:
        raise ValueError(f"ssim expects (B, C, H, W); got shape {tuple(pred.shape)}")

    C = pred.shape[1]
    device, dtype = pred.device, pred.dtype

    g1d = _gaussian_window(window_size, sigma, device, dtype)
    window_2d = (g1d[:, None] * g1d[None, :])  # (W, W)
    window = window_2d.expand(C, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu_p = F.conv2d(pred, window, padding=pad, groups=C)
    mu_t = F.conv2d(target, window, padding=pad, groups=C)

    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p2 = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_p2
    sigma_t2 = F.conv2d(target * target, window, padding=pad, groups=C) - mu_t2
    sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_pt

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2))
    return ssim_map.mean()


def log_hdr_ssim_loss(pred, target, eps: float = 1.0,
                      window_size: int = 11, sigma: float = 1.5):
    """`1 - SSIM` computed on log-HDR-mapped images. Working in log space lets
    SSIM see structure across the full HDR range (esp. the sun disk) instead
    of saturating on bright pixels.

    Targets are clamped to >=0 before log to be robust to numerical noise.
    """
    log_pred = torch.log(pred + eps)
    log_target = torch.log(target.clamp_min(0) + eps)
    # Empirical data_range for envmaps in log1p space. Max-pooled targets
    # reach log(~148k) ~ 12, so log(1e6+eps) ~ 13.8 keeps SSIM constants
    # accurate without saturating on sun-disk structure.
    data_range = math.log(1e6 + eps)
    return 1.0 - ssim(log_pred, log_target, window_size=window_size,
                      sigma=sigma, data_range=data_range)


class MultiTaskLoss(nn.Module):
    """Combined lighting (log-HDR MSE + L1 + SSIM) + material MSE."""

    def __init__(self, config):
        super().__init__()
        self.lighting_weight = config.lighting_loss_weight
        self.material_weight = config.material_loss_weight
        self.log_eps = float(getattr(config, "log_eps", 1.0))

        # Per-term weights inside the lighting loss. Defaults are picked so
        # MSE remains dominant while L1/SSIM nudge toward sharper structure.
        self.mse_weight = float(getattr(config, "lighting_mse_weight", 1.0))
        self.l1_weight = float(getattr(config, "lighting_l1_weight", 0.5))
        self.ssim_weight = float(getattr(config, "lighting_ssim_weight", 0.2))
        self.peak_lambda = float(getattr(config, "lighting_peak_weight_lambda", 0.0))

        self.material_loss = nn.MSELoss()

    def forward(self, pred_lighting, pred_material,
                target_lighting, target_material):
        loss_lighting = lighting_loss(
            pred_lighting, target_lighting,
            eps=self.log_eps,
            mse_weight=self.mse_weight,
            l1_weight=self.l1_weight,
            ssim_weight=self.ssim_weight,
            peak_lambda=self.peak_lambda,
        )
        loss_material = self.material_loss(pred_material, target_material)

        total = (self.lighting_weight * loss_lighting +
                 self.material_weight * loss_material)
        return total, loss_lighting, loss_material


def lighting_loss(pred, target, eps: float = 1.0,
                  mse_weight: float = 1.0,
                  l1_weight: float = 0.5,
                  ssim_weight: float = 0.2,
                  peak_lambda: float = 0.0):
    """Combined log-HDR lighting loss: weighted MSE + L1 + (1 - SSIM).

    SSIM is window-based and is left unweighted; the per-pixel `peak_lambda`
    boost only applies to the MSE/L1 terms.
    """
    total = pred.new_zeros(())
    if mse_weight != 0.0:
        total = total + mse_weight * log_hdr_mse(pred, target, eps=eps,
                                                 peak_lambda=peak_lambda)
    if l1_weight != 0.0:
        total = total + l1_weight * log_hdr_l1(pred, target, eps=eps,
                                               peak_lambda=peak_lambda)
    if ssim_weight != 0.0:
        total = total + ssim_weight * log_hdr_ssim_loss(pred, target, eps=eps)
    return total
