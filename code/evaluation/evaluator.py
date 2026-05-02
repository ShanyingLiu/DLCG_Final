# Run trained models on a test set and compare baseline vs multi-task

import torch
import numpy as np
from torch.utils.data import DataLoader

from evaluation.metrics import (
    angular_error,
    relative_intensity_error,
    sh_mse,
    material_param_mae,
    per_material_metrics,
    lighting_metrics_by_bucket,
    default_param_buckets,
    mean_std_ci,
    paired_ttest,
)


def evaluate_model(model, dataloader, device, is_multitask=True):
    """Run a model over a dataloader and collect predictions.

    Args:
        model: trained nn.Module. Should return (pred_lighting,) for baseline
               or (pred_lighting, pred_material_params) for multitask.
        dataloader: DataLoader yielding dicts with 'image', 'lighting',
                    'material_params', 'material_label'.
        device: torch device.
        is_multitask: if True, model returns (lighting, material params);
                      else just lighting.

    Returns:
        Dict with arrays: pred_sh, target_sh, target_material_label,
                          pred_material_params (or None), target_material_params.
    """
    model.eval()

    all_pred_sh = []
    all_target_sh = []
    all_pred_params = []
    all_target_params = []
    all_target_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            target_sh = batch['lighting'].numpy()
            target_params = batch['material_params'].numpy()
            target_label = batch['material_label'].numpy()

            if is_multitask:
                pred_lighting, pred_params = model(images)
                all_pred_params.append(pred_params.cpu().numpy())
            else:
                pred_lighting = model(images)

            all_pred_sh.append(pred_lighting.cpu().numpy())
            all_target_sh.append(target_sh)
            all_target_params.append(target_params)
            all_target_labels.append(target_label)

    results = {
        "pred_sh": np.concatenate(all_pred_sh),
        "target_sh": np.concatenate(all_target_sh),
        "target_material_params": np.concatenate(all_target_params),
        "target_material_label": np.concatenate(all_target_labels),
        "pred_material_params": (np.concatenate(all_pred_params)
                                 if all_pred_params else None),
    }
    return results


def compute_all_metrics(results, num_classes=5,
                        material_param_names=None):
    """Compute full evaluation metrics from evaluate_model output.

    Args:
        results: dict from evaluate_model.
        num_classes: number of material categories (for per-material grouping).
        material_param_names: list of param names for per-param MAE reporting.

    Returns:
        Dict of aggregate and per-material metrics.
    """
    pred_sh = results["pred_sh"]
    target_sh = results["target_sh"]
    target_label = results["target_material_label"]
    pred_params = results["pred_material_params"]
    target_params = results["target_material_params"]

    n = len(pred_sh)

    # Per-sample lighting metrics (kept for paired statistical comparison
    # between models — same dataloader order = aligned indices).
    ang_errors = [angular_error(pred_sh[i], target_sh[i]) for i in range(n)]
    int_errors = [relative_intensity_error(pred_sh[i], target_sh[i]) for i in range(n)]
    mses = [sh_mse(pred_sh[i], target_sh[i]) for i in range(n)]

    ang_stats = mean_std_ci(ang_errors)
    int_stats = mean_std_ci(int_errors)
    mse_stats = mean_std_ci(mses)

    metrics = {
        "aggregate": {
            "angular_error_mean": ang_stats["mean"],
            "angular_error_std": ang_stats["std"],
            "angular_error_ci95_half": ang_stats["ci95_half"],
            "angular_error_median": float(np.median(ang_errors)),
            "intensity_error_mean": int_stats["mean"],
            "intensity_error_std": int_stats["std"],
            "intensity_error_ci95_half": int_stats["ci95_half"],
            "sh_mse_mean": mse_stats["mean"],
            "sh_mse_std": mse_stats["std"],
            "sh_mse_ci95_half": mse_stats["ci95_half"],
            "n_samples": n,
        },
        "per_sample": {
            "angular_error": [float(v) for v in ang_errors],
        },
        "per_material": per_material_metrics(
            pred_sh, target_sh, target_label, num_classes
        ),
    }

    # Continuous-parameter buckets (always available — uses ground-truth
    # target_material_params, which are present even for the baseline since
    # they come from metadata).
    if material_param_names is not None and target_params is not None:
        buckets = default_param_buckets(target_params, material_param_names)
        metrics["per_param_bucket"] = lighting_metrics_by_bucket(
            pred_sh, target_sh, buckets
        )

    # Material parameter regression metrics (multitask only)
    if pred_params is not None:
        metrics["material_param_mae"] = material_param_mae(
            pred_params, target_params, material_param_names
        )

    return metrics


def compare_models(baseline_metrics, multitask_metrics,
                   material_param_names=None):
    """Print a side-by-side comparison of baseline vs multitask results.

    Args:
        baseline_metrics: dict from compute_all_metrics for the baseline.
        multitask_metrics: dict from compute_all_metrics for the multitask model.
        material_param_names: list of material parameter names for the
                              per-parameter MAE table.
    """
    def _fmt_with_ci(stats_dict, key_mean, key_ci_half):
        m = stats_dict.get(key_mean)
        c = stats_dict.get(key_ci_half)
        if m is None:
            return "N/A"
        if c is None:
            return f"{m:.2f}"
        return f"{m:.2f} ± {c:.2f}"

    print("=" * 78)
    print(f"{'Metric':<35} {'Baseline':>20} {'Multitask':>20}")
    print("-" * 78)

    ba = baseline_metrics["aggregate"]
    ma = multitask_metrics["aggregate"]

    for label, mean_key, ci_key in [
        ("angular_error (mean ± 95% CI)",
         "angular_error_mean", "angular_error_ci95_half"),
        ("intensity_error (mean ± 95% CI)",
         "intensity_error_mean", "intensity_error_ci95_half"),
        ("sh_mse (mean ± 95% CI)",
         "sh_mse_mean", "sh_mse_ci95_half"),
    ]:
        b_str = _fmt_with_ci(ba, mean_key, ci_key)
        m_str = _fmt_with_ci(ma, mean_key, ci_key)
        print(f"  {label:<33} {b_str:>20} {m_str:>20}")

    # Median doesn't have a CI in our output — keep its single-value format.
    print(f"  {'angular_error_median':<33} "
          f"{ba['angular_error_median']:>20.2f} "
          f"{ma['angular_error_median']:>20.2f}")

    if "material_param_mae" in multitask_metrics:
        mae = multitask_metrics["material_param_mae"]
        print(f"  {'material_param_mae (mean)':<33} {'N/A':>20} "
              f"{mae['mean']:>20.4f}")

    # Paired t-test on per-sample angular errors
    if ("per_sample" in baseline_metrics
            and "per_sample" in multitask_metrics):
        b_ang = baseline_metrics["per_sample"]["angular_error"]
        m_ang = multitask_metrics["per_sample"]["angular_error"]
        if len(b_ang) == len(m_ang):
            tt = paired_ttest(b_ang, m_ang)  # baseline - multitask
            sign = "multitask better" if tt["mean_diff"] > 0 else "baseline better"
            print()
            print("Paired t-test (baseline - multitask) on per-sample angular error:")
            print(f"  n              = {tt['n']}")
            print(f"  mean diff      = {tt['mean_diff']:+.4f} deg "
                  f"(95% CI ± {tt['ci95_half_diff']:.4f}) — {sign}")
            print(f"  std of diffs   = {tt['std_diff']:.4f}")
            print(f"  t-statistic    = {tt['t_stat']:+.4f}")
            print(f"  p-value (two-sided, normal approx) = {tt['p_value']:.4g}")
        else:
            print("\n[paired t-test skipped: per-sample arrays have different "
                  f"lengths {len(b_ang)} vs {len(m_ang)}]")

    print()
    print("Per-material angular error (mean ± 95% CI):")
    print(f"  {'Material':<20} {'Baseline':>20} {'Multitask':>20} {'Delta':>10}")
    print("  " + "-" * 75)

    for mat_name in ["diffuse", "glossy", "metallic", "rough_metallic", "dielectric"]:
        b_dict = baseline_metrics["per_material"].get(mat_name, {})
        m_dict = multitask_metrics["per_material"].get(mat_name, {})
        b_val = b_dict.get("angular_error")
        m_val = m_dict.get("angular_error")
        b_str = _fmt_with_ci(b_dict, "angular_error", "angular_error_ci95_half")
        m_str = _fmt_with_ci(m_dict, "angular_error", "angular_error_ci95_half")
        if b_val is not None and m_val is not None:
            d_str = f"{m_val - b_val:+.2f}"
        else:
            d_str = "—"
        print(f"  {mat_name:<20} {b_str:>20} {m_str:>20} {d_str:>10}")

    # Per-parameter-bucket breakdown (continuous-property buckets)
    if ("per_param_bucket" in baseline_metrics
            and "per_param_bucket" in multitask_metrics):
        print()
        print("Per-parameter-bucket angular error (mean ± 95% CI):")
        print(f"  {'Bucket':<20} {'Baseline':>20} {'Multitask':>20} {'Delta':>10}")
        print("  " + "-" * 75)
        bucket_order = ["metallic_yes", "metallic_no", "transmissive", "opaque",
                        "rough_low", "rough_mid", "rough_high"]
        for name in bucket_order:
            b = baseline_metrics["per_param_bucket"].get(name, {})
            m = multitask_metrics["per_param_bucket"].get(name, {})
            b_val = b.get("angular_error")
            m_val = m.get("angular_error")
            b_str = _fmt_with_ci(b, "angular_error", "angular_error_ci95_half")
            m_str = _fmt_with_ci(m, "angular_error", "angular_error_ci95_half")
            if b_val is not None and m_val is not None:
                d_str = f"{m_val - b_val:+.2f}"
            else:
                d_str = "—"
            count = b.get("count", 0)
            print(f"  {name:<20} {b_str:>20} {m_str:>20} {d_str:>10}  (n={count})")

    # Per-parameter MAE breakdown for the multitask model
    if "material_param_mae" in multitask_metrics:
        mae = multitask_metrics["material_param_mae"]
        names = material_param_names or list(mae["per_param"].keys())
        print()
        print("Multitask per-parameter MAE:")
        for name in names:
            v = mae["per_param"].get(name)
            if v is not None:
                print(f"  {name:<20} {v:.4f}")

    print("=" * 70)
