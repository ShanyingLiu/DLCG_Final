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


def plot_per_material_comparison(baseline_metrics, multitask_metrics, save_path):
    """Grouped bar chart of log_mse by material: baseline vs multitask."""
    material_names = ["diffuse", "glossy", "metallic", "rough_metallic", "dielectric"]
    baseline_vals = []
    multitask_vals = []

    for name in material_names:
        b = baseline_metrics["per_material"].get(name, {}).get("log_mse")
        m = multitask_metrics["per_material"].get(name, {}).get("log_mse")
        baseline_vals.append(b if b is not None else 0)
        multitask_vals.append(m if m is not None else 0)

    x = np.arange(len(material_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#5588cc")
    ax.bar(x + width / 2, multitask_vals, width, label="Multitask", color="#cc7755")
    ax.set_ylabel("Log-HDR MSE")
    ax.set_xlabel("Material Category")
    ax.set_title("Lighting Log-HDR MSE by Material Category")
    ax.set_xticks(x)
    ax.set_xticklabels(material_names, rotation=15)
    ax.legend()
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
