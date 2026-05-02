# Evaluation metrics for lighting estimation and material classification

import math
import numpy as np


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

# Normal approximation: with our test sets (N ~ 1500) the t-distribution is
# indistinguishable from normal, so we avoid a scipy dependency by using the
# standard-normal critical value 1.96 for 95% CIs and the erf-based normal
# CDF for two-sided p-values.
_Z_95 = 1.959964


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mean_std_ci(values):
    """Return mean, sample std (ddof=1), and 95% CI half-width on the mean.

    Args:
        values: 1-D iterable of floats.

    Returns:
        dict with keys: mean, std, ci95_half, n. Half-width is `1.96 * sem`,
        so the 95% CI is `[mean - ci95_half, mean + ci95_half]`. For n <= 1
        the std and CI are reported as 0.0 (point estimate only).
    """
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "std": 0.0, "ci95_half": 0.0, "n": 0}
    m = float(arr.mean())
    if n < 2:
        return {"mean": m, "std": 0.0, "ci95_half": 0.0, "n": n}
    s = float(arr.std(ddof=1))
    sem = s / math.sqrt(n)
    return {"mean": m, "std": s, "ci95_half": _Z_95 * sem, "n": n}


def paired_ttest(values_a, values_b):
    """Two-sided paired t-test on per-sample arrays a and b.

    Tests whether the mean of (a - b) differs from zero. Uses the normal
    approximation for the p-value, which is essentially exact for n > ~50.

    Args:
        values_a, values_b: equal-length iterables of per-sample scores
            (must be paired by sample index).

    Returns:
        dict with: n, mean_diff, std_diff, sem, t_stat, p_value, ci95_half_diff.
        Convention: mean_diff = mean(a) - mean(b). For angular error, passing
        baseline as `a` and multitask as `b` yields a positive mean_diff when
        the multitask model is more accurate.
    """
    a = np.asarray(list(values_a), dtype=np.float64)
    b = np.asarray(list(values_b), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have the same shape, "
                         f"got {a.shape} vs {b.shape}")
    diff = a - b
    n = int(diff.size)
    if n < 2:
        return {"n": n, "mean_diff": float(diff.mean()) if n else 0.0,
                "std_diff": 0.0, "sem": 0.0,
                "t_stat": 0.0, "p_value": 1.0, "ci95_half_diff": 0.0}
    md = float(diff.mean())
    sd = float(diff.std(ddof=1))
    sem = sd / math.sqrt(n)
    t = md / sem if sem > 0 else 0.0
    p = float(2.0 * (1.0 - _normal_cdf(abs(t))))
    return {
        "n": n,
        "mean_diff": md,
        "std_diff": sd,
        "sem": sem,
        "t_stat": float(t),
        "p_value": p,
        "ci95_half_diff": _Z_95 * sem,
    }


def dominant_light_direction(sh_coeffs):
    """Extract dominant light direction from order-1 SH coefficients.

    Args:
        sh_coeffs: array of shape (27,) laid out as [R0..R8, G0..G8, B0..B8].

    Returns:
        Unit-length direction vector (3,) derived from the L=1 band.
    """
    sh = np.asarray(sh_coeffs, dtype=np.float64).reshape(3, 9)  # (RGB, 9 SH)
    # Luminance-weight the three channels
    lum = 0.2126 * sh[0] + 0.7152 * sh[1] + 0.0722 * sh[2]

    # L=1 basis ordering (real SH): Y1^-1, Y1^0, Y1^+1  → indices 1,2,3
    # These correspond to y, z, x directions respectively.
    direction = np.array([lum[3], lum[1], lum[2]])  # (x, y, z)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.zeros(3)
    return direction / norm


def angular_error(pred_sh, target_sh):
    """Angular error (degrees) between dominant light directions.

    Args:
        pred_sh: predicted SH coefficients, shape (27,).
        target_sh: ground-truth SH coefficients, shape (27,).

    Returns:
        Angle in degrees between the two dominant directions.
    """
    d_pred = dominant_light_direction(pred_sh)
    d_target = dominant_light_direction(target_sh)

    cos_angle = np.clip(np.dot(d_pred, d_target), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def relative_intensity_error(pred_sh, target_sh):
    """Relative error of overall lighting intensity (L0 band = ambient).

    Computes |pred_intensity - gt_intensity| / (gt_intensity + eps).

    Args:
        pred_sh: predicted SH coefficients, shape (27,).
        target_sh: ground-truth SH coefficients, shape (27,).

    Returns:
        Scalar relative intensity error.
    """
    pred = np.asarray(pred_sh, dtype=np.float64).reshape(3, 9)
    target = np.asarray(target_sh, dtype=np.float64).reshape(3, 9)

    # L0 coefficient (index 0) for each channel → luminance
    pred_l0 = 0.2126 * pred[0, 0] + 0.7152 * pred[1, 0] + 0.0722 * pred[2, 0]
    target_l0 = 0.2126 * target[0, 0] + 0.7152 * target[1, 0] + 0.0722 * target[2, 0]

    eps = 1e-6
    return float(abs(pred_l0 - target_l0) / (abs(target_l0) + eps))


def sh_mse(pred_sh, target_sh):
    """Mean squared error over all 27 SH coefficients."""
    pred = np.asarray(pred_sh, dtype=np.float64)
    target = np.asarray(target_sh, dtype=np.float64)
    return float(np.mean((pred - target) ** 2))


def material_param_mae(pred_params, target_params, param_names=None):
    """Mean absolute error for material parameter regression.

    Both inputs are expected to be in the same normalized space the model
    was trained in (params already in [0,1], with ior pre-normalized).

    Args:
        pred_params: (N, P) predicted material parameters.
        target_params: (N, P) ground-truth material parameters.
        param_names: optional list of length P naming each parameter; used
                     for the per-parameter breakdown.

    Returns:
        Dict with keys:
            'mean': float, MAE averaged over all params and samples.
            'per_param': dict mapping param_name -> mean absolute error.
    """
    pred = np.asarray(pred_params, dtype=np.float64)
    target = np.asarray(target_params, dtype=np.float64)
    abs_err = np.abs(pred - target)              # (N, P)

    per_param_mae = abs_err.mean(axis=0)         # (P,)
    if param_names is None:
        param_names = [f"param_{i}" for i in range(per_param_mae.shape[0])]

    return {
        "mean": float(abs_err.mean()),
        "per_param": {name: float(v)
                      for name, v in zip(param_names, per_param_mae)},
    }


def lighting_metrics_by_bucket(pred_sh_all, target_sh_all, buckets):
    """Compute lighting metrics for each named subset of test samples.

    Args:
        pred_sh_all: (N, 27) predicted SH coefficients.
        target_sh_all: (N, 27) ground-truth SH coefficients.
        buckets: dict mapping bucket_name -> 1-D boolean mask of length N.

    Returns:
        Dict mapping bucket_name -> {angular_error, intensity_error, sh_mse, count}.
    """
    results = {}
    n_total = len(pred_sh_all)
    for name, mask in buckets.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != n_total:
            raise ValueError(f"bucket '{name}' mask length {mask.shape[0]} "
                             f"does not match N={n_total}")
        count = int(mask.sum())
        if count == 0:
            results[name] = {"angular_error": None, "angular_error_std": None,
                             "angular_error_ci95_half": None,
                             "intensity_error": None,
                             "sh_mse": None, "count": 0}
            continue
        idxs = np.where(mask)[0]
        ang = [angular_error(pred_sh_all[i], target_sh_all[i]) for i in idxs]
        inten = [relative_intensity_error(pred_sh_all[i], target_sh_all[i])
                 for i in idxs]
        mses = [sh_mse(pred_sh_all[i], target_sh_all[i]) for i in idxs]
        ang_stats = mean_std_ci(ang)
        results[name] = {
            "angular_error": ang_stats["mean"],
            "angular_error_std": ang_stats["std"],
            "angular_error_ci95_half": ang_stats["ci95_half"],
            "intensity_error": float(np.mean(inten)),
            "sh_mse": float(np.mean(mses)),
            "count": count,
        }
    return results


def default_param_buckets(material_params, param_names):
    """Build the default continuous-parameter buckets used by the multitask
    evaluator. Returns a dict[name -> boolean mask] suitable for
    lighting_metrics_by_bucket.

    Buckets (using the normalized [0,1] target parameters):
        metallic_yes / metallic_no       (split at 0.5)
        transmissive / opaque            (split at 0.25)
        rough_low / rough_mid / rough_high  (cuts at 0.3 and 0.7)
    """
    p = np.asarray(material_params, dtype=np.float32)
    idx = {name: i for i, name in enumerate(param_names)}
    metallic = p[:, idx["metallic"]]
    transmission = p[:, idx["transmission"]]
    roughness = p[:, idx["roughness"]]

    return {
        "metallic_yes":   metallic > 0.5,
        "metallic_no":    metallic <= 0.5,
        "transmissive":   transmission > 0.25,
        "opaque":         transmission <= 0.25,
        "rough_low":      roughness < 0.3,
        "rough_mid":      (roughness >= 0.3) & (roughness < 0.7),
        "rough_high":     roughness >= 0.7,
    }


def per_material_metrics(pred_sh_all, target_sh_all, material_labels, num_classes=5):
    """Compute lighting metrics broken down by material category.

    Args:
        pred_sh_all: (N, 27) predicted SH coefficients.
        target_sh_all: (N, 27) ground-truth SH coefficients.
        material_labels: (N,) ground-truth material labels.
        num_classes: number of material categories.

    Returns:
        Dict mapping material label → {angular_error, intensity_error, sh_mse, count}.
    """
    material_names = ["diffuse", "glossy", "metallic", "rough_metallic", "dielectric"]
    results = {}

    for label in range(num_classes):
        mask = np.asarray(material_labels) == label
        count = int(mask.sum())
        if count == 0:
            results[material_names[label]] = {
                "angular_error": None,
                "angular_error_std": None,
                "angular_error_ci95_half": None,
                "intensity_error": None,
                "sh_mse": None,
                "count": 0,
            }
            continue

        ang_errors = [
            angular_error(pred_sh_all[i], target_sh_all[i])
            for i in range(len(mask)) if mask[i]
        ]
        int_errors = [
            relative_intensity_error(pred_sh_all[i], target_sh_all[i])
            for i in range(len(mask)) if mask[i]
        ]
        mses = [
            sh_mse(pred_sh_all[i], target_sh_all[i])
            for i in range(len(mask)) if mask[i]
        ]
        ang_stats = mean_std_ci(ang_errors)

        results[material_names[label]] = {
            "angular_error": ang_stats["mean"],
            "angular_error_std": ang_stats["std"],
            "angular_error_ci95_half": ang_stats["ci95_half"],
            "intensity_error": float(np.mean(int_errors)),
            "sh_mse": float(np.mean(mses)),
            "count": count,
        }

    return results
