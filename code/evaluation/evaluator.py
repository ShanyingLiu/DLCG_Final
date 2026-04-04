# Run trained models on a test set and compare baseline vs multi-task

import torch
import numpy as np
from torch.utils.data import DataLoader

from evaluation.metrics import (
    angular_error,
    relative_intensity_error,
    sh_mse,
    material_accuracy,
    per_material_metrics,
)


def evaluate_model(model, dataloader, device, is_multitask=True):
    """Run a model over a dataloader and collect predictions.

    Args:
        model: trained nn.Module. Should return (pred_lighting,) for baseline
               or (pred_lighting, pred_material) for multitask.
        dataloader: DataLoader yielding dicts with 'image', 'lighting', 'material'.
        device: torch device.
        is_multitask: if True, model returns (lighting, material); else just lighting.

    Returns:
        Dict with arrays: pred_sh, target_sh, pred_material (or None), target_material.
    """
    model.eval()

    all_pred_sh = []
    all_target_sh = []
    all_pred_mat = []
    all_target_mat = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            target_sh = batch['lighting'].numpy()
            target_mat = batch['material'].numpy()

            if is_multitask:
                pred_lighting, pred_material = model(images)
                pred_mat = pred_material.argmax(dim=1).cpu().numpy()
                all_pred_mat.append(pred_mat)
            else:
                pred_lighting = model(images)

            all_pred_sh.append(pred_lighting.cpu().numpy())
            all_target_sh.append(target_sh)
            all_target_mat.append(target_mat)

    results = {
        "pred_sh": np.concatenate(all_pred_sh),
        "target_sh": np.concatenate(all_target_sh),
        "target_material": np.concatenate(all_target_mat),
        "pred_material": np.concatenate(all_pred_mat) if all_pred_mat else None,
    }
    return results


def compute_all_metrics(results, num_classes=5):
    """Compute full evaluation metrics from evaluate_model output.

    Args:
        results: dict from evaluate_model.
        num_classes: number of material categories.

    Returns:
        Dict of aggregate and per-material metrics.
    """
    pred_sh = results["pred_sh"]
    target_sh = results["target_sh"]
    target_mat = results["target_material"]
    pred_mat = results["pred_material"]

    n = len(pred_sh)

    # Aggregate lighting metrics
    ang_errors = [angular_error(pred_sh[i], target_sh[i]) for i in range(n)]
    int_errors = [relative_intensity_error(pred_sh[i], target_sh[i]) for i in range(n)]
    mses = [sh_mse(pred_sh[i], target_sh[i]) for i in range(n)]

    metrics = {
        "aggregate": {
            "angular_error_mean": float(np.mean(ang_errors)),
            "angular_error_median": float(np.median(ang_errors)),
            "intensity_error_mean": float(np.mean(int_errors)),
            "sh_mse_mean": float(np.mean(mses)),
            "n_samples": n,
        },
        "per_material": per_material_metrics(
            pred_sh, target_sh, target_mat, num_classes
        ),
    }

    # Material classification accuracy (multitask only)
    if pred_mat is not None:
        metrics["material_accuracy"] = material_accuracy(pred_mat, target_mat)

    return metrics


def compare_models(baseline_metrics, multitask_metrics):
    """Print a side-by-side comparison of baseline vs multitask results.

    Args:
        baseline_metrics: dict from compute_all_metrics for the baseline.
        multitask_metrics: dict from compute_all_metrics for the multitask model.
    """
    print("=" * 70)
    print(f"{'Metric':<35} {'Baseline':>15} {'Multitask':>15}")
    print("-" * 70)

    ba = baseline_metrics["aggregate"]
    ma = multitask_metrics["aggregate"]

    for key in ["angular_error_mean", "angular_error_median",
                "intensity_error_mean", "sh_mse_mean"]:
        print(f"  {key:<33} {ba[key]:>15.4f} {ma[key]:>15.4f}")

    if "material_accuracy" in multitask_metrics:
        print(f"  {'material_accuracy':<33} {'N/A':>15} "
              f"{multitask_metrics['material_accuracy']:>15.4f}")

    print()
    print("Per-material angular error:")
    print(f"  {'Material':<20} {'Baseline':>15} {'Multitask':>15} {'Delta':>15}")
    print("  " + "-" * 65)

    for mat_name in ["diffuse", "glossy", "metallic", "rough_metallic", "dielectric"]:
        b_val = baseline_metrics["per_material"].get(mat_name, {}).get("angular_error")
        m_val = multitask_metrics["per_material"].get(mat_name, {}).get("angular_error")

        b_str = f"{b_val:.2f}" if b_val is not None else "N/A"
        m_str = f"{m_val:.2f}" if m_val is not None else "N/A"

        if b_val is not None and m_val is not None:
            delta = m_val - b_val
            d_str = f"{delta:+.2f}"
        else:
            d_str = "—"

        print(f"  {mat_name:<20} {b_str:>15} {m_str:>15} {d_str:>15}")

    print("=" * 70)
