# Evaluation metrics for envmap lighting estimation and material regression

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
    """Return mean, sample std (ddof=1), and 95% CI half-width on the mean."""
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

    Convention: mean_diff = mean(a) - mean(b). For an error metric (lower is
    better), passing baseline as `a` and multitask as `b` yields a positive
    mean_diff when the multitask model is more accurate.
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


# ---------------------------------------------------------------------------
# Envmap metrics
# ---------------------------------------------------------------------------

def linear_mse(pred_env, target_env):
    """MSE in linear HDR space, per-sample. Inputs (3, H, W) or (H, W, 3)."""
    p = np.asarray(pred_env, dtype=np.float64).ravel()
    t = np.asarray(target_env, dtype=np.float64).ravel()
    return float(np.mean((p - t) ** 2))


def log_mse(pred_env, target_env, eps: float = 1.0):
    """MSE in log-HDR space, per-sample. Mirrors training loss."""
    p = np.log(np.asarray(pred_env, dtype=np.float64) + eps).ravel()
    t = np.log(np.asarray(target_env, dtype=np.float64) + eps).ravel()
    return float(np.mean((p - t) ** 2))


def psnr_log(pred_env, target_env, eps: float = 1.0):
    """PSNR computed in log-HDR space.

    Treats log(env+eps) as the signal with peak 1.0. Comparable across runs
    of the same dataset; not a calibrated photographic PSNR.
    """
    mse = log_mse(pred_env, target_env, eps=eps)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


# ---------------------------------------------------------------------------
# Material parameter regression
# ---------------------------------------------------------------------------

def material_param_mae(pred_params, target_params, param_names=None):
    """Mean absolute error for material parameter regression.

    Both inputs are expected to be in the same normalized space the model
    was trained in (params already in [0,1], with ior pre-normalized).
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


# ---------------------------------------------------------------------------
# Bucketed and per-material aggregations
# ---------------------------------------------------------------------------

def _summarize_bucket(pred_env_all, target_env_all, idxs):
    """Compute log_mse (with CI), linear_mse, psnr_log over a set of indices."""
    log_mses = [log_mse(pred_env_all[i], target_env_all[i]) for i in idxs]
    lin_mses = [linear_mse(pred_env_all[i], target_env_all[i]) for i in idxs]
    psnrs    = [psnr_log(pred_env_all[i], target_env_all[i])  for i in idxs]
    s = mean_std_ci(log_mses)
    return {
        "log_mse": s["mean"],
        "log_mse_std": s["std"],
        "log_mse_ci95_half": s["ci95_half"],
        "linear_mse": float(np.mean(lin_mses)),
        "psnr_log": float(np.mean(psnrs)),
        "count": len(idxs),
    }


def _empty_bucket():
    return {"log_mse": None, "log_mse_std": None, "log_mse_ci95_half": None,
            "linear_mse": None, "psnr_log": None, "count": 0}


def lighting_metrics_by_bucket(pred_env_all, target_env_all, buckets):
    """Compute envmap metrics for each named subset of test samples."""
    results = {}
    n_total = len(pred_env_all)
    for name, mask in buckets.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != n_total:
            raise ValueError(f"bucket '{name}' mask length {mask.shape[0]} "
                             f"does not match N={n_total}")
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            results[name] = _empty_bucket()
            continue
        results[name] = _summarize_bucket(pred_env_all, target_env_all, idxs)
    return results


def default_param_buckets(material_params, param_names):
    """Build the default continuous-parameter buckets used by the multitask
    evaluator. Returns a dict[name -> boolean mask].
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


def per_material_metrics(pred_env_all, target_env_all, material_labels, num_classes=5):
    """Compute envmap metrics broken down by material category."""
    material_names = ["diffuse", "glossy", "metallic", "rough_metallic", "dielectric"]
    results = {}

    for label in range(num_classes):
        mask = np.asarray(material_labels) == label
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            results[material_names[label]] = _empty_bucket()
            continue
        results[material_names[label]] = _summarize_bucket(
            pred_env_all, target_env_all, idxs
        )
    return results
