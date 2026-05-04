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


def copy_sample_inputs(images_root: str, metadata, target_indices,
                       out_dir: str, n_samples: int) -> None:
    """Copy each sample's source sphere render to <out_dir>/sample_<i>_input.png.

    Lets viewers see, alongside the envmap-comparison and teapot-render
    figures, the actual photo the model received as input. Skips silently
    if the source PNG is missing.
    """
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    n = min(int(n_samples), len(target_indices))
    for i in range(n):
        idx = int(target_indices[i])
        if idx >= len(metadata):
            continue
        src = os.path.join(images_root, metadata[idx]["filename"])
        if not os.path.exists(src):
            print(f"[input-copy] sample {i}: {src} missing; skip")
            continue
        dst = os.path.join(out_dir, f"sample_{i}_input.png")
        shutil.copy2(src, dst)


def _find_hdri_on_disk(hdri_root: str, filename: str) -> Optional[str]:
    """Mirror data_gen.discover_hdris layout: dataset/hdris/{puresky,scene}/."""
    for sub in ("puresky", "scene", ""):
        p = os.path.join(hdri_root, sub, filename)
        if os.path.exists(p):
            return p
    return None


def _run_blender(blender_bin, script_path, teapot, hdri, output,
                 rotation_deg, strength, resolution, samples,
                 transparent_bg: bool = False) -> bool:
    # BPY image/file APIs do not reliably resolve relative paths against the
    # subprocess CWD, so pass absolute paths for everything Blender opens.
    script_abs = os.path.abspath(script_path)
    teapot_abs = os.path.abspath(teapot)
    hdri_abs   = os.path.abspath(hdri)
    output_abs = os.path.abspath(output)
    os.makedirs(os.path.dirname(output_abs) or ".", exist_ok=True)
    cmd = [
        blender_bin, "--background", "--python", script_abs, "--",
        "--teapot", teapot_abs,
        "--hdri", hdri_abs,
        "--output", output_abs,
        "--rotation_z_deg", f"{float(rotation_deg)}",
        "--strength", f"{float(strength)}",
        "--resolution", str(int(resolution)),
        "--samples", str(int(samples)),
    ]
    if transparent_bg:
        cmd.append("--transparent-bg")
    cmd += ["--cycles-seed", "0"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        print(f"[teapot] Blender failed (rc={proc.returncode}) for {output_abs}\n{tail}")
        return False
    if not os.path.exists(output_abs):
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        print(f"[teapot] Blender exited 0 but no output at {output_abs}\n{tail}")
        return False
    return True


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
        render_resolution: int = 512,
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

    # Pick a fresh random subset of eval positions each run so the showcase
    # renders aren't always the same first-n samples. Uses entropy-based
    # seeding (no fixed seed) for genuine variety run-to-run.
    rng = np.random.default_rng()
    positions = sorted(rng.choice(len(indices), n, replace=False).tolist())
    picked_indices = [int(indices[p]) for p in positions]

    out_dir = os.path.join(vis_dir, "teapot_renders")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[teapot] Rendering {n} test sample(s) at "
          f"{render_resolution}px × {render_samples}spp via {blender_bin}")
    print(f"[teapot] Eval positions: {positions}  "
          f"(metadata indices: {picked_indices})")
    print(f"[teapot] Output dir: {out_dir}")

    # Drop the source sphere image into teapot_renders/ so each sample's
    # inputs and renders sit next to each other.
    copy_sample_inputs(config.images_root, metadata, picked_indices,
                       out_dir, n)

    for i, pos in enumerate(positions):
        idx = int(indices[pos])
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
            pred_env = results["pred_env"][pos]               # (3, H, W)
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


# ---------------------------------------------------------------------------
# Eval-mode rendering + LPIPS on rendered teapots
# ---------------------------------------------------------------------------

def render_for_lpips(config,
                     baseline_results,
                     multitask_results,
                     subset_indices,
                     out_dir: str,
                     render_resolution: int = 256,
                     render_samples: int = 16,
                     teapot_path: str = "dataset/renders/utah_teapot.obj",
                     hdri_root: str = "dataset/hdris",
                     blender_script: str = "code/render_teapot.py"):
    """Render GT + per-model teapots (transparent envmap background) for each
    sample listed in subset_indices. Returns a dict with paths and the
    metadata index for each rendered sample. Skips renders that already exist.

    subset_indices: positional indices into the eval results arrays (NOT the
    metadata.json indices). I.e. they index into pred_env / target_indices.
    """
    blender_bin = find_blender_bin()
    if blender_bin is None:
        print("[teapot-lpips] Blender not found; skipping.")
        return None
    if not os.path.exists(teapot_path) or not os.path.exists(blender_script):
        print("[teapot-lpips] Teapot OBJ or render script missing; skipping.")
        return None
    if multitask_results is None or baseline_results is None:
        print("[teapot-lpips] Need both baseline and multitask results.")
        return None

    metadata_path = os.path.join(config.metadata_root, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    indices = multitask_results["target_indices"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"[teapot-lpips] Rendering {len(subset_indices)} sample(s) at "
          f"{render_resolution}px × {render_samples}spp (transparent bg).")

    records = []
    for k, eval_i in enumerate(subset_indices):
        eval_i = int(eval_i)
        meta_idx = int(indices[eval_i])
        if meta_idx >= len(metadata):
            continue
        meta = metadata[meta_idx]
        hdri_disk = _find_hdri_on_disk(hdri_root, meta["hdri_file"])
        if hdri_disk is None:
            print(f"[teapot-lpips] eval {eval_i}: HDRI {meta['hdri_file']} "
                  f"missing; skip")
            continue

        rot_deg = float(meta["rotation_z_deg"])
        strength = float(meta["world_strength"])

        gt_path = os.path.join(out_dir, f"eval_{eval_i:05d}_gt.png")
        b_path  = os.path.join(out_dir, f"eval_{eval_i:05d}_baseline.png")
        m_path  = os.path.join(out_dir, f"eval_{eval_i:05d}_multitask.png")

        # GT render (cached)
        if not os.path.exists(gt_path):
            ok = _run_blender(blender_bin, blender_script, teapot_path,
                              hdri_disk, gt_path, rot_deg, strength,
                              render_resolution, render_samples,
                              transparent_bg=True)
            if not ok:
                continue

        # Baseline pred
        if not os.path.exists(b_path):
            hdr_b = os.path.join(out_dir, f"eval_{eval_i:05d}_baseline.hdr")
            try:
                save_envmap_as_hdr(baseline_results["pred_env"][eval_i], hdr_b)
            except Exception as exc:
                print(f"[teapot-lpips] save_hdr failed (baseline {eval_i}): {exc}")
                continue
            ok = _run_blender(blender_bin, blender_script, teapot_path,
                              hdr_b, b_path, 0.0, 1.0,
                              render_resolution, render_samples,
                              transparent_bg=True)
            if not ok:
                continue

        # Multitask pred
        if not os.path.exists(m_path):
            hdr_m = os.path.join(out_dir, f"eval_{eval_i:05d}_multitask.hdr")
            try:
                save_envmap_as_hdr(multitask_results["pred_env"][eval_i], hdr_m)
            except Exception as exc:
                print(f"[teapot-lpips] save_hdr failed (multitask {eval_i}): {exc}")
                continue
            ok = _run_blender(blender_bin, blender_script, teapot_path,
                              hdr_m, m_path, 0.0, 1.0,
                              render_resolution, render_samples,
                              transparent_bg=True)
            if not ok:
                continue

        records.append({"eval_i": eval_i, "meta_idx": meta_idx,
                        "gt": gt_path, "baseline": b_path, "multitask": m_path})
        if (k + 1) % 10 == 0:
            print(f"[teapot-lpips] rendered {k + 1}/{len(subset_indices)}")

    print(f"[teapot-lpips] Rendered {len(records)} complete triplet(s).")
    return records


def _load_rgba_over_gray(path: str, gray: float = 0.5,
                         target_hw=None) -> np.ndarray:
    """Load an RGBA PNG, alpha-composite over a neutral gray, return (3,H,W)
    in [-1, 1] for LPIPS. If target_hw=(H,W), resize before compositing so
    mixed-resolution caches stack cleanly."""
    if cv2 is None:
        raise RuntimeError("cv2 is required to read rendered PNGs")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read {path}")
    if target_hw is not None and (img.shape[0], img.shape[1]) != target_hw:
        img = cv2.resize(img, (target_hw[1], target_hw[0]),
                         interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[..., :3]
        a = img[..., 3:4]
    else:
        bgr = img[..., :3] if img.ndim == 3 else np.repeat(img[..., None], 3, -1)
        a = np.ones_like(bgr[..., :1])
    rgb = bgr[..., ::-1]                                # BGR -> RGB
    comp = a * rgb + (1.0 - a) * gray                   # over neutral gray
    comp = np.clip(comp, 0.0, 1.0) * 2.0 - 1.0          # to [-1, 1]
    return np.transpose(comp, (2, 0, 1)).astype(np.float32)


def lpips_on_renders(records, device: str = "cpu", batch_size: int = 8):
    """Compute paired LPIPS(GT, baseline) and LPIPS(GT, multitask) over a
    list of render records (from render_for_lpips). Returns
    (b_lpips, m_lpips) as lists of equal length, or None on failure.
    """
    if not records:
        return None
    try:
        import lpips as _lpips_mod  # noqa: F401
        import torch
        net = _lpips_mod.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
    except ImportError:
        print("[teapot-lpips] lpips package not installed; skipping.")
        return None

    # Pick a common target size (smallest H/W across cached renders) so
    # mixed-resolution caches (e.g. earlier 128px + later 256px) coexist.
    target_hw = None
    for r in records:
        for k in ("gt", "baseline", "multitask"):
            im = cv2.imread(r[k], cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            hw = (im.shape[0], im.shape[1])
            target_hw = hw if target_hw is None else (
                min(target_hw[0], hw[0]), min(target_hw[1], hw[1]))

    arrs = {"gt": [], "baseline": [], "multitask": []}
    for r in records:
        for k in ("gt", "baseline", "multitask"):
            arrs[k].append(_load_rgba_over_gray(r[k], target_hw=target_hw))
    g = np.stack(arrs["gt"])
    b = np.stack(arrs["baseline"])
    m = np.stack(arrs["multitask"])

    def _batch(a, b):
        out = np.empty(a.shape[0], dtype=np.float32)
        with torch.no_grad():
            for s in range(0, a.shape[0], batch_size):
                e = min(s + batch_size, a.shape[0])
                ta = torch.from_numpy(a[s:e]).to(device)
                tb = torch.from_numpy(b[s:e]).to(device)
                d = net(ta, tb).view(-1).cpu().numpy()
                out[s:e] = d
        return out

    b_lp = _batch(g, b).tolist()
    m_lp = _batch(g, m).tolist()
    return [float(v) for v in b_lp], [float(v) for v in m_lp]
