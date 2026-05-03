"""Render Utah-teapot insertions for a few test samples and compare:
  - GT: lit by the original HDRI from metadata (with rotation + strength)
  - Pred (per model): lit by the model's predicted envmap saved as .hdr

Spawns Blender via subprocess for each render and composes a side-by-side
PNG per sample.
"""

import json
import os
import shutil
import subprocess
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def find_blender_bin() -> Optional[str]:
    """Locate a Blender executable. Honors $BLENDER_BIN, then PATH, then the
    macOS default install location."""
    env = os.environ.get("BLENDER_BIN")
    if env and os.path.exists(env):
        return env
    p = shutil.which("blender")
    if p:
        return p
    mac = "/Applications/Blender.app/Contents/MacOS/Blender"
    if os.path.exists(mac):
        return mac
    return None


def save_envmap_as_hdr(arr: np.ndarray, path: str) -> None:
    """Write a (3, H, W) or (H, W, 3) float32 RGB envmap as a Radiance .hdr.

    cv2 expects BGR; we transpose channels accordingly. Radiance HDR is
    used (rather than EXR) because cv2 supports it without optional deps.
    """
    if cv2 is None:
        raise RuntimeError("cv2 is required to write .hdr envmaps")
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    bgr = a[..., ::-1].astype(np.float32)
    bgr = np.ascontiguousarray(bgr)
    if not path.lower().endswith(".hdr"):
        path = path + ".hdr"
    ok = cv2.imwrite(path, bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed writing {path}")


def _find_hdri_on_disk(hdri_root: str, filename: str) -> Optional[str]:
    """Mirror data_gen.discover_hdris layout: dataset/hdris/{puresky,scene}/."""
    for sub in ("puresky", "scene", ""):
        p = os.path.join(hdri_root, sub, filename)
        if os.path.exists(p):
            return p
    return None


def _run_blender(blender_bin, script_path, teapot, hdri, output,
                 rotation_deg, strength, resolution, samples) -> bool:
    cmd = [
        blender_bin, "--background", "--python", script_path, "--",
        "--teapot", teapot,
        "--hdri", hdri,
        "--output", output,
        "--rotation_z_deg", f"{float(rotation_deg)}",
        "--strength", f"{float(strength)}",
        "--resolution", str(int(resolution)),
        "--samples", str(int(samples)),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        print(f"[teapot] Blender failed (rc={proc.returncode}) for {output}\n{tail}")
        return False
    return os.path.exists(output)


def _compose_strip(out_dir, sample_idx, panels, suptitle):
    """panels: list of (title, png_path). Writes sample_<i>_compare.png."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    panels = [(t, p) for t, p in panels if p and os.path.exists(p)]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, path) in zip(axes, panels):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"sample_{sample_idx}_compare.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_teapot_comparisons(config,
                           baseline_results,
                           multitask_results,
                           vis_dir,
                           n_samples: int = 3,
                           render_resolution: int = 256,
                           render_samples: int = 32,
                           teapot_path: str = "dataset/renders/utah_teapot.obj",
                           hdri_root: str = "dataset/hdris",
                           blender_script: str = "code/render_teapot.py"):
    """For up to n_samples test images, render: GT lit by original HDRI, and
    one prediction render per available model. Save individual PNGs and a
    side-by-side comparison.

    Skips silently (with a printed reason) if Blender, the OBJ, or the
    render script is missing — main() should remain functional without them.
    """
    blender_bin = find_blender_bin()
    if blender_bin is None:
        print("[teapot] Blender not found (set $BLENDER_BIN or install). "
              "Skipping teapot comparison renders.")
        return
    if not os.path.exists(teapot_path):
        print(f"[teapot] OBJ not found at {teapot_path}; skipping.")
        return
    if not os.path.exists(blender_script):
        print(f"[teapot] Render script missing at {blender_script}; skipping.")
        return

    sample_source = multitask_results or baseline_results
    if sample_source is None or "target_indices" not in sample_source:
        print("[teapot] No target_indices in eval results; skipping.")
        return

    metadata_path = os.path.join(config.metadata_root, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    indices = sample_source["target_indices"]
    n = min(int(n_samples), len(indices))
    if n == 0:
        print("[teapot] No samples available; skipping.")
        return

    out_dir = os.path.join(vis_dir, "teapot_renders")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[teapot] Rendering {n} test sample(s) at "
          f"{render_resolution}px × {render_samples}spp via {blender_bin}")
    print(f"[teapot] Output dir: {out_dir}")

    for i in range(n):
        idx = int(indices[i])
        if idx >= len(metadata):
            print(f"[teapot] sample {i}: metadata index {idx} out of range; skip")
            continue
        meta = metadata[idx]

        hdri_disk = _find_hdri_on_disk(hdri_root, meta["hdri_file"])
        if hdri_disk is None:
            print(f"[teapot] sample {i}: HDRI {meta['hdri_file']} not found; skip")
            continue

        rot_deg = float(meta["rotation_z_deg"])
        strength = float(meta["world_strength"])

        # Ground-truth teapot render
        gt_path = os.path.join(out_dir, f"sample_{i}_gt.png")
        ok_gt = _run_blender(blender_bin, blender_script, teapot_path,
                             hdri_disk, gt_path, rot_deg, strength,
                             render_resolution, render_samples)

        # Predicted renders, one per available model
        panels = [("Ground Truth (orig HDRI)", gt_path if ok_gt else None)]
        for tag, results in [("baseline", baseline_results),
                             ("multitask", multitask_results)]:
            if results is None:
                continue
            pred_env = results["pred_env"][i]                 # (3, H, W)
            hdr_path = os.path.join(out_dir, f"sample_{i}_{tag}_pred.hdr")
            try:
                save_envmap_as_hdr(pred_env, hdr_path)
            except Exception as exc:
                print(f"[teapot] sample {i} ({tag}): save_hdr failed: {exc}")
                continue
            png_path = os.path.join(out_dir, f"sample_{i}_{tag}_pred.png")
            ok = _run_blender(blender_bin, blender_script, teapot_path,
                              hdr_path, png_path,
                              0.0, 1.0,           # rotation/strength baked in
                              render_resolution, render_samples)
            label = f"{tag.title()} predicted envmap"
            panels.append((label, png_path if ok else None))

        suptitle = (f"Sample {i} — {meta.get('material_type','?')} "
                    f"{meta.get('color_name','')} | HDRI={meta.get('hdri_short_name','?')} "
                    f"rot={meta.get('rotation_z_deg','?')}° "
                    f"strength={meta.get('world_strength','?')}")
        compare = _compose_strip(out_dir, i, panels, suptitle)
        if compare:
            print(f"[teapot] sample {i} -> {compare}")

    print(f"[teapot] Done. See {out_dir}/sample_*_compare.png")
