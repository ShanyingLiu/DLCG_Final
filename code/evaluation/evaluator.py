# Run trained models on a test set and compare baseline vs multi-task

import torch
import numpy as np

from evaluation.metrics import (
    log_mse,
    linear_mse,
    psnr_log,
    material_param_mae,
    per_material_metrics,
    lighting_metrics_by_bucket,
    default_param_buckets,
    mean_std_ci,
    paired_ttest,
)


def evaluate_model(model, dataloader, device, is_multitask=True):
    """Run a model over a dataloader and collect predictions.

    Returns:
        Dict with arrays: pred_env (N,3,H,W), target_env (N,3,H,W),
                          target_material_label, target_material_params,
                          pred_material_params (or None).
    """
    model.eval()

    all_pred_env = []
    all_target_env = []
    all_pred_params = []
    all_target_params = []
    all_target_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            target_env = batch['lighting'].numpy()
            target_params = batch['material_params'].numpy()
            target_label = batch['material_label'].numpy()

            if is_multitask:
                pred_env, pred_params = model(images)
                all_pred_params.append(pred_params.cpu().numpy())
            else:
                pred_env = model(images)

            all_pred_env.append(pred_env.cpu().numpy())
            all_target_env.append(target_env)
            all_target_params.append(target_params)
            all_target_labels.append(target_label)

    results = {
        "pred_env": np.concatenate(all_pred_env),
        "target_env": np.concatenate(all_target_env),
        "target_material_params": np.concatenate(all_target_params),
        "target_material_label": np.concatenate(all_target_labels),
        "pred_material_params": (np.concatenate(all_pred_params)
                                 if all_pred_params else None),
    }
    return results


def compute_all_metrics(results, num_classes=5,
                        material_param_names=None):
    """Compute aggregate, per-sample, per-material, and per-bucket metrics."""
    pred_env = results["pred_env"]
    target_env = results["target_env"]
    target_label = results["target_material_label"]
    pred_params = results["pred_material_params"]
    target_params = results["target_material_params"]

    n = len(pred_env)

    # Per-sample arrays (kept for paired statistical comparison between models).
    log_mses = [log_mse(pred_env[i], target_env[i]) for i in range(n)]
    lin_mses = [linear_mse(pred_env[i], target_env[i]) for i in range(n)]
    psnrs    = [psnr_log(pred_env[i], target_env[i])  for i in range(n)]

    log_stats = mean_std_ci(log_mses)
    lin_stats = mean_std_ci(lin_mses)
    psnr_stats = mean_std_ci(psnrs)

    metrics = {
        "aggregate": {
            "log_mse_mean": log_stats["mean"],
            "log_mse_std": log_stats["std"],
            "log_mse_ci95_half": log_stats["ci95_half"],
            "linear_mse_mean": lin_stats["mean"],
            "linear_mse_std": lin_stats["std"],
            "linear_mse_ci95_half": lin_stats["ci95_half"],
            "psnr_log_mean": psnr_stats["mean"],
            "psnr_log_std": psnr_stats["std"],
            "psnr_log_ci95_half": psnr_stats["ci95_half"],
            "n_samples": n,
        },
        "per_sample": {
            "log_mse": [float(v) for v in log_mses],
        },
        "per_material": per_material_metrics(
            pred_env, target_env, target_label, num_classes
        ),
    }

    if material_param_names is not None and target_params is not None:
        buckets = default_param_buckets(target_params, material_param_names)
        metrics["per_param_bucket"] = lighting_metrics_by_bucket(
            pred_env, target_env, buckets
        )

    if pred_params is not None:
        metrics["material_param_mae"] = material_param_mae(
            pred_params, target_params, material_param_names
        )

    return metrics


def compare_models(baseline_metrics, multitask_metrics,
                   material_param_names=None):
    """Print a side-by-side comparison of baseline vs multitask results."""
    def _fmt_with_ci(stats_dict, key_mean, key_ci_half):
        m = stats_dict.get(key_mean)
        c = stats_dict.get(key_ci_half)
        if m is None:
            return "N/A"
        if c is None:
            return f"{m:.4f}"
        return f"{m:.4f} ± {c:.4f}"

    print("=" * 78)
    print(f"{'Metric':<35} {'Baseline':>20} {'Multitask':>20}")
    print("-" * 78)

    ba = baseline_metrics["aggregate"]
    ma = multitask_metrics["aggregate"]

    for label, mean_key, ci_key in [
        ("log_mse (mean ± 95% CI)",
         "log_mse_mean", "log_mse_ci95_half"),
        ("linear_mse (mean ± 95% CI)",
         "linear_mse_mean", "linear_mse_ci95_half"),
        ("psnr_log dB (mean ± 95% CI)",
         "psnr_log_mean", "psnr_log_ci95_half"),
    ]:
        b_str = _fmt_with_ci(ba, mean_key, ci_key)
        m_str = _fmt_with_ci(ma, mean_key, ci_key)
        print(f"  {label:<33} {b_str:>20} {m_str:>20}")

    if "material_param_mae" in multitask_metrics:
        mae = multitask_metrics["material_param_mae"]
        print(f"  {'material_param_mae (mean)':<33} {'N/A':>20} "
              f"{mae['mean']:>20.4f}")

    # Paired t-test on per-sample log_mse
    if ("per_sample" in baseline_metrics
            and "per_sample" in multitask_metrics):
        b_log = baseline_metrics["per_sample"]["log_mse"]
        m_log = multitask_metrics["per_sample"]["log_mse"]
        if len(b_log) == len(m_log):
            tt = paired_ttest(b_log, m_log)  # baseline - multitask
            sign = "multitask better" if tt["mean_diff"] > 0 else "baseline better"
            print()
            print("Paired t-test (baseline - multitask) on per-sample log_mse:")
            print(f"  n              = {tt['n']}")
            print(f"  mean diff      = {tt['mean_diff']:+.4f} "
                  f"(95% CI ± {tt['ci95_half_diff']:.4f}) — {sign}")
            print(f"  std of diffs   = {tt['std_diff']:.4f}")
            print(f"  t-statistic    = {tt['t_stat']:+.4f}")
            print(f"  p-value (two-sided, normal approx) = {tt['p_value']:.4g}")
        else:
            print("\n[paired t-test skipped: per-sample arrays have different "
                  f"lengths {len(b_log)} vs {len(m_log)}]")

    print()
    print("Per-material log_mse (mean ± 95% CI):")
    print(f"  {'Material':<20} {'Baseline':>20} {'Multitask':>20} {'Delta':>10}")
    print("  " + "-" * 75)

    for mat_name in ["diffuse", "glossy", "metallic", "rough_metallic", "dielectric"]:
        b_dict = baseline_metrics["per_material"].get(mat_name, {})
        m_dict = multitask_metrics["per_material"].get(mat_name, {})
        b_val = b_dict.get("log_mse")
        m_val = m_dict.get("log_mse")
        b_str = _fmt_with_ci(b_dict, "log_mse", "log_mse_ci95_half")
        m_str = _fmt_with_ci(m_dict, "log_mse", "log_mse_ci95_half")
        if b_val is not None and m_val is not None:
            d_str = f"{m_val - b_val:+.4f}"
        else:
            d_str = "—"
        print(f"  {mat_name:<20} {b_str:>20} {m_str:>20} {d_str:>10}")

    if ("per_param_bucket" in baseline_metrics
            and "per_param_bucket" in multitask_metrics):
        print()
        print("Per-parameter-bucket log_mse (mean ± 95% CI):")
        print(f"  {'Bucket':<20} {'Baseline':>20} {'Multitask':>20} {'Delta':>10}")
        print("  " + "-" * 75)
        bucket_order = ["metallic_yes", "metallic_no", "transmissive", "opaque",
                        "rough_low", "rough_mid", "rough_high"]
        for name in bucket_order:
            b = baseline_metrics["per_param_bucket"].get(name, {})
            m = multitask_metrics["per_param_bucket"].get(name, {})
            b_val = b.get("log_mse")
            m_val = m.get("log_mse")
            b_str = _fmt_with_ci(b, "log_mse", "log_mse_ci95_half")
            m_str = _fmt_with_ci(m, "log_mse", "log_mse_ci95_half")
            if b_val is not None and m_val is not None:
                d_str = f"{m_val - b_val:+.4f}"
            else:
                d_str = "—"
            count = b.get("count", 0)
            print(f"  {name:<20} {b_str:>20} {m_str:>20} {d_str:>10}  (n={count})")

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
