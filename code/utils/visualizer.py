"""Visualize predicted vs ground truth lighting by rendering a sphere"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# SH evaluation on a sphere (bands 0-2, 9 basis functions)
# ---------------------------------------------------------------------------

def _eval_sh_basis(normals):
    """Evaluate the 9 real SH basis functions (l=0,1,2) at given directions.

    Args:
        normals: (N, 3) unit vectors.

    Returns:
        (N, 9) array of basis function values.
    """
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    basis = np.zeros((len(normals), 9))

    # Band 0
    basis[:, 0] = 0.282095  # 0.5 * sqrt(1/pi)
    # Band 1
    basis[:, 1] = 0.488603 * y
    basis[:, 2] = 0.488603 * z
    basis[:, 3] = 0.488603 * x
    # Band 2
    basis[:, 4] = 1.092548 * x * y
    basis[:, 5] = 1.092548 * y * z
    basis[:, 6] = 0.315392 * (3.0 * z * z - 1.0)
    basis[:, 7] = 1.092548 * x * z
    basis[:, 8] = 0.546274 * (x * x - y * y)

    return basis


def _sh_to_color(sh_coeffs, normals):
    """Compute Lambertian-shaded RGB from SH coefficients and surface normals.

    Args:
        sh_coeffs: (27,) flattened as [R0..R8, G0..G8, B0..B8].
        normals: (N, 3) unit normals.

    Returns:
        (N, 3) RGB values, clipped to [0, 1].
    """
    sh = np.asarray(sh_coeffs, dtype=np.float64).reshape(3, 9)  # (3, 9)
    basis = _eval_sh_basis(normals)  # (N, 9)

    # Cosine-lobe transfer coefficients for Lambertian diffuse
    Al = np.array([
        np.pi,                                  # l=0
        2.0 * np.pi / 3.0,                      # l=1 (x3)
        2.0 * np.pi / 3.0,
        2.0 * np.pi / 3.0,
        np.pi / 4.0,                            # l=2 (x5)
        np.pi / 4.0,
        np.pi / 4.0,
        np.pi / 4.0,
        np.pi / 4.0,
    ])

    weighted_basis = basis * Al[np.newaxis, :]  # (N, 9)

    rgb = np.zeros((len(normals), 3))
    for c in range(3):
        rgb[:, c] = weighted_basis @ sh[c]

    # Normalize to [0, 1] range for display
    max_val = rgb.max()
    if max_val > 0:
        rgb = rgb / max_val
    return np.clip(rgb, 0.0, 1.0)


def _make_sphere_image(sh_coeffs, resolution=128):
    """Render a Lambertian sphere lit by given SH coefficients.

    Returns:
        (resolution, resolution, 3) RGB image as uint8, black background.
    """
    y_coords = np.linspace(1, -1, resolution)
    x_coords = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)
    r2 = xx ** 2 + yy ** 2
    mask = r2 <= 1.0

    # Surface normals on the sphere
    zz = np.sqrt(np.maximum(1.0 - r2, 0.0))
    normals = np.stack([xx[mask], yy[mask], zz[mask]], axis=-1)

    colors = _sh_to_color(sh_coeffs, normals)

    image = np.zeros((resolution, resolution, 3))
    image[mask] = colors
    return (image * 255).astype(np.uint8), mask


# ---------------------------------------------------------------------------
# Public visualization functions
# ---------------------------------------------------------------------------

def render_sphere_comparison(pred_sh, target_sh, save_path, title=None):
    """Render spheres under predicted and GT lighting side-by-side.

    Args:
        pred_sh: (27,) predicted SH coefficients.
        target_sh: (27,) ground-truth SH coefficients.
        save_path: path to save the output image.
    """
    pred_img, _ = _make_sphere_image(pred_sh)
    gt_img, _ = _make_sphere_image(target_sh)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(gt_img)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    axes[1].imshow(pred_img)
    axes[1].set_title("Predicted")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sh_coefficients(pred_sh, target_sh, save_path, title=None):
    """Bar chart comparing predicted vs GT SH coefficients per channel.

    Args:
        pred_sh: (27,) predicted SH coefficients.
        target_sh: (27,) ground-truth SH coefficients.
        save_path: path to save the output image.
    """
    pred = np.asarray(pred_sh).reshape(3, 9)
    target = np.asarray(target_sh).reshape(3, 9)
    channels = ["Red", "Green", "Blue"]
    colors_gt = ["#cc4444", "#44aa44", "#4444cc"]
    colors_pred = ["#ff8888", "#88dd88", "#8888ff"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    x = np.arange(9)
    labels = [f"L{l}M{m}" for l in range(3) for m in range(-l, l + 1)]

    for c, ax in enumerate(axes):
        ax.bar(x - 0.15, target[c], 0.3, label="GT", color=colors_gt[c])
        ax.bar(x + 0.15, pred[c], 0.3, label="Pred", color=colors_pred[c])
        ax.set_ylabel(channels[c])
        ax.legend(fontsize=8)
        ax.set_xticks(x)

    axes[-1].set_xticklabels(labels, fontsize=8)
    axes[-1].set_xlabel("SH Basis Function")
    suptitle = title or "SH Coefficient Comparison"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_material_comparison(baseline_metrics, multitask_metrics, save_path):
    """Grouped bar chart of angular error by material: baseline vs multitask.

    Args:
        baseline_metrics: dict from compute_all_metrics for baseline.
        multitask_metrics: dict from compute_all_metrics for multitask.
        save_path: path to save the output image.
    """
    material_names = ["diffuse", "glossy", "metallic", "rough_metallic", "dielectric"]
    baseline_vals = []
    multitask_vals = []

    for name in material_names:
        b = baseline_metrics["per_material"].get(name, {}).get("angular_error")
        m = multitask_metrics["per_material"].get(name, {}).get("angular_error")
        baseline_vals.append(b if b is not None else 0)
        multitask_vals.append(m if m is not None else 0)

    x = np.arange(len(material_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#5588cc")
    ax.bar(x + width / 2, multitask_vals, width, label="Multitask", color="#cc7755")
    ax.set_ylabel("Angular Error (degrees)")
    ax.set_xlabel("Material Category")
    ax.set_title("Lighting Angular Error by Material Category")
    ax.set_xticks(x)
    ax.set_xticklabels(material_names, rotation=15)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, save_path, title=None):
    """Plot training and validation loss curves over epochs.

    Args:
        history: dict with keys 'train_loss', 'val_loss',
                 'train_loss_lighting', 'val_loss_lighting',
                 and optionally 'train_loss_material', 'val_loss_material'.
        save_path: path to save the output image.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Total loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()

    # Lighting loss
    axes[1].plot(epochs, history["train_loss_lighting"], label="Train")
    axes[1].plot(epochs, history["val_loss_lighting"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Lighting Loss (MSE)")
    axes[1].set_title("Lighting Loss")
    axes[1].legend()

    suptitle = title or "Training Curves"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
