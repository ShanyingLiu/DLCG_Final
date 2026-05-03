"""Visualize predicted vs ground-truth HDR envmaps and training curves."""

import os
import numpy as np
import matplotlib.pyplot as plt


def _tonemap(x):
    """Reinhard-style tonemap for HDR display. x assumed non-negative."""
    return np.clip(x / (1.0 + x), 0.0, 1.0)


def _to_hwc(x):
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return x


def plot_envmap_comparison(pred_env, target_env, save_path, title=None):
    """3-panel figure: GT envmap, predicted envmap, abs-error heatmap.

    Args:
        pred_env, target_env: arrays shape (3, H, W) OR (H, W, 3).
        save_path: output PNG path.
    """
    p = _to_hwc(pred_env)
    t = _to_hwc(target_env)

    p_disp = _tonemap(p)
    t_disp = _tonemap(t)
    err = np.abs(p - t).mean(axis=-1)  # (H, W) per-pixel mean abs error

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].imshow(t_disp)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    axes[1].imshow(p_disp)
    axes[1].set_title("Predicted")
    axes[1].axis("off")
    im = axes[2].imshow(err, cmap="magma")
    axes[2].set_title("Abs error (mean over RGB)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _grouped_bar_with_ci(ax, group_names, baseline_means, baseline_cis,
                         multitask_means, multitask_cis,
                         xlabel, ylabel, title):
    """Helper: paired bars per group with optional 95% CI whiskers."""
    x = np.arange(len(group_names))
    width = 0.35
    ax.bar(x - width / 2, baseline_means, width, yerr=baseline_cis,
           label="Baseline", color="#5588cc",
           capsize=4, error_kw={"ecolor": "#1f3b66", "linewidth": 1.0})
    ax.bar(x + width / 2, multitask_means, width, yerr=multitask_cis,
           label="Multitask", color="#cc7755",
           capsize=4, error_kw={"ecolor": "#5e2d18", "linewidth": 1.0})
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=15)
    ax.legend()


def _collect_means_cis(per_group_dict, group_names,
                       mean_key="log_mse", ci_key="log_mse_ci95_half"):
    """Pull (means, ci-half-widths) lists in `group_names` order. Replaces
    missing entries with 0 so the bar chart still renders cleanly."""
    means, cis = [], []
    for name in group_names:
        d = per_group_dict.get(name, {}) or {}
        m = d.get(mean_key)
        c = d.get(ci_key)
        means.append(m if m is not None else 0.0)
        cis.append(c if c is not None else 0.0)
    return means, cis


def plot_per_material_comparison(baseline_metrics, multitask_metrics, save_path):
    """Grouped bar chart of log_mse by discrete material category, with
    95% CI whiskers from the per-material per-sample spread.
    """
    material_names = ["diffuse", "glossy", "metallic", "rough_metallic", "dielectric"]
    b_means, b_cis = _collect_means_cis(
        baseline_metrics["per_material"], material_names)
    m_means, m_cis = _collect_means_cis(
        multitask_metrics["per_material"], material_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar_with_ci(
        ax, material_names, b_means, b_cis, m_means, m_cis,
        xlabel="Material Category",
        ylabel="Log-HDR MSE",
        title="Lighting Log-HDR MSE by Material Category (95% CI)",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_param_bucket_comparison(baseline_metrics, multitask_metrics,
                                     save_path):
    """Grouped bar chart over continuous-attribute buckets:
    metallic_yes/no, transmissive/opaque, rough_low/mid/high.

    Complements plot_per_material_comparison by slicing the test set along
    individual BSDF attributes rather than the discrete category label.
    """
    bucket_order = ["metallic_yes", "metallic_no",
                    "transmissive", "opaque",
                    "rough_low", "rough_mid", "rough_high"]
    b_dict = baseline_metrics.get("per_param_bucket", {})
    m_dict = multitask_metrics.get("per_param_bucket", {})
    if not b_dict or not m_dict:
        return  # nothing to plot

    b_means, b_cis = _collect_means_cis(b_dict, bucket_order)
    m_means, m_cis = _collect_means_cis(m_dict, bucket_order)

    # Show sample counts under each bucket so reviewers can spot thin buckets.
    counts = [(b_dict.get(name, {}) or {}).get("count", 0)
              for name in bucket_order]
    labels_with_n = [f"{name}\n(n={c})" for name, c in zip(bucket_order, counts)]

    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bar_with_ci(
        ax, labels_with_n, b_means, b_cis, m_means, m_cis,
        xlabel="Material Attribute Bucket",
        ylabel="Log-HDR MSE",
        title="Lighting Log-HDR MSE by Material Attribute (95% CI)",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_param_mae(multitask_metrics, save_path,
                       param_names=None):
    """Bar chart of multitask material-parameter MAE (one bar per BSDF param)."""
    mae = multitask_metrics.get("material_param_mae")
    if not mae:
        return  # multitask wasn't run

    per_param = mae["per_param"]
    if param_names is None:
        param_names = list(per_param.keys())
    values = [per_param[n] for n in param_names]

    x = np.arange(len(param_names))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, values, color="#cc7755")
    ax.axhline(mae["mean"], color="#444", linewidth=1, linestyle="--",
               label=f"mean MAE = {mae['mean']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.set_xlabel("BSDF Parameter (normalized to [0, 1])")
    ax.set_ylabel("MAE")
    ax.set_title("Multitask Material-Parameter Regression MAE")
    ax.set_ylim(0, max(0.01, max(values) * 1.15))
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_log_mse_distribution(baseline_metrics, multitask_metrics, save_path):
    """Overlaid histogram of per-sample log_mse for baseline vs multitask,
    with vertical lines at each model's mean. Visualizes the spread that
    backs the paired t-test.
    """
    b = baseline_metrics.get("per_sample", {}).get("log_mse")
    m = multitask_metrics.get("per_sample", {}).get("log_mse")
    if not b or not m:
        return

    b = np.asarray(b)
    m = np.asarray(m)
    lo = float(min(b.min(), m.min()))
    hi = float(max(b.max(), m.max()))
    bins = np.linspace(lo, hi, 40)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(b, bins=bins, color="#5588cc", alpha=0.55, label="Baseline")
    ax.hist(m, bins=bins, color="#cc7755", alpha=0.55, label="Multitask")
    ax.axvline(b.mean(), color="#1f3b66", linewidth=1.5, linestyle="--",
               label=f"baseline mean = {b.mean():.4f}")
    ax.axvline(m.mean(), color="#5e2d18", linewidth=1.5, linestyle="--",
               label=f"multitask mean = {m.mean():.4f}")
    ax.set_xlabel("Per-sample Log-HDR MSE")
    ax.set_ylabel("# test samples")
    ax.set_title(f"Per-sample Log-HDR MSE Distribution (N = {len(b)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, save_path, title=None):
    """Plot training and validation loss curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss_lighting"], label="Train")
    axes[1].plot(epochs, history["val_loss_lighting"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Lighting Loss (log-HDR MSE)")
    axes[1].set_title("Lighting Loss")
    axes[1].legend()

    suptitle = title or "Training Curves"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
