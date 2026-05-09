"""Unified 4-model evaluation: baseline + multitask + EMLight + LuxDiT.

For each model with predictions over the same test split (positionally
aligned), we compute:
  - envmap log_mse, LPIPS (your existing metric stack)
  - Utah teapot renders + paired teapot-LPIPS vs GT (downstream metric)

External model predictions are provided as .npz files written by
emlight_infer.py / luxdit_postprocess.py:
    pred_env       : (N, 3, H, W)  float32 HDR linear
    target_indices : (N,)          int64    metadata indices

Usage:
    python -m code.external.run_external_compare \
        --emlight_npz result/material_aware_lighting/pred_env_emlight.npz \
        --luxdit_npz  result/material_aware_lighting/pred_env_luxdit.npz \
        --out_dir result/material_aware_lighting/external_compare \
        --n_teapot 50
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from config import config
from dataset.data import SphereDataset
from models.baseline_model import BaselineLightingNet
from models.multitask_model import MaterialAwareLightingNet
from evaluation.evaluator import evaluate_model, compute_all_metrics
from evaluation.metrics import lpips_per_sample, mean_std_ci, paired_ttest
from utils.teapot_compare import (
    find_blender_bin, _run_blender, save_envmap_as_hdr, _find_hdri_on_disk,
    _load_rgba_over_gray,
)


# ---------------------------------------------------------------------------
# Loading user models
# ---------------------------------------------------------------------------

def load_user_results(device):
    """Run baseline + multitask models over the test set and return their
    results dicts (same shape as evaluator.evaluate_model)."""
    tfm = T.Compose([T.Resize((config.image_size, config.image_size)),
                     T.ToTensor()])
    test_ds = SphereDataset("test", config, transform=tfm)
    loader = DataLoader(test_ds, batch_size=config.batch_size,
                        shuffle=False, num_workers=0)

    out = {}
    for tag, ctor, is_mt in [
        ("baseline", BaselineLightingNet, False),
        ("multitask", MaterialAwareLightingNet, True),
    ]:
        model = ctor(config).to(device)
        ckpt_path = os.path.join(
            config.save_dir,
            f"{config.experiment_name}_{tag}_best.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"[compare] loaded {tag} from {ckpt_path}")
        out[tag] = evaluate_model(model, loader, device, is_multitask=is_mt)
    return out


def load_external_npz(path, ref_target_indices):
    """Load a .npz produced by emlight_infer / luxdit_postprocess and return
    a results dict matching the user-model result shape, *positionally aligned*
    to the reference order.
    """
    z = np.load(path, allow_pickle=True)
    pred_env = z["pred_env"]
    ti = z["target_indices"]
    if not np.array_equal(ti, ref_target_indices):
        # Reorder to match the user models' eval order.
        order = {int(v): i for i, v in enumerate(ti)}
        try:
            perm = np.array([order[int(v)] for v in ref_target_indices])
        except KeyError as e:
            raise RuntimeError(
                f"{path}: missing metadata index {e} present in user eval order")
        pred_env = pred_env[perm]
        ti = ti[perm]
    return {
        "pred_env": pred_env.astype(np.float32),
        "target_env": None,                  # filled by caller from user dict
        "target_indices": ti,
        "target_material_label": None,
        "target_material_params": None,
        "pred_material_params": None,
    }


# ---------------------------------------------------------------------------
# Metrics over arbitrary models
# ---------------------------------------------------------------------------

def compute_envmap_metrics(pred_env, target_env, device="cpu"):
    from evaluation.metrics import log_mse
    n = len(pred_env)
    log_mses = [log_mse(pred_env[i], target_env[i]) for i in range(n)]
    log_stats = mean_std_ci(log_mses)
    out = {
        "log_mse_mean": log_stats["mean"],
        "log_mse_ci95_half": log_stats["ci95_half"],
        "log_mse_per_sample": [float(v) for v in log_mses],
        "n": n,
    }
    lp = lpips_per_sample(pred_env, target_env, device=device)
    if lp is not None:
        lp_stats = mean_std_ci(lp)
        out["envmap_lpips_mean"] = lp_stats["mean"]
        out["envmap_lpips_ci95_half"] = lp_stats["ci95_half"]
        out["envmap_lpips_per_sample"] = [float(v) for v in lp]
    return out


# ---------------------------------------------------------------------------
# Multi-model teapot rendering + LPIPS
# ---------------------------------------------------------------------------

def render_teapots_multi(model_preds, target_indices, subset_positions,
                         out_dir, render_resolution=256, render_samples=16,
                         teapot_path="dataset/renders/utah_teapot.obj",
                         hdri_root="dataset/hdris",
                         blender_script="code/render_teapot.py"):
    """For each eval position in subset_positions, render the GT teapot once
    and a per-model teapot under each model's predicted envmap. Returns a
    list of records:
        {"eval_i": int, "meta_idx": int, "gt": path,
         "<model_tag>": path, ...}
    """
    blender_bin = find_blender_bin()
    if blender_bin is None:
        print("[teapot-multi] Blender not found; skipping.")
        return []
    if not os.path.exists(teapot_path) or not os.path.exists(blender_script):
        print("[teapot-multi] OBJ or render script missing; skipping.")
        return []

    metadata_path = os.path.join(config.metadata_root, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    records = []
    for k, pos in enumerate(subset_positions):
        pos = int(pos)
        meta_idx = int(target_indices[pos])
        if meta_idx >= len(metadata):
            continue
        meta = metadata[meta_idx]
        hdri_disk = _find_hdri_on_disk(hdri_root, meta["hdri_file"])
        if hdri_disk is None:
            continue
        rot_deg = float(meta["rotation_z_deg"])
        strength = float(meta["world_strength"])

        rec = {"eval_i": pos, "meta_idx": meta_idx}

        gt_path = os.path.join(out_dir, f"eval_{pos:05d}_gt.png")
        if not os.path.exists(gt_path):
            ok = _run_blender(blender_bin, blender_script, teapot_path,
                              hdri_disk, gt_path, rot_deg, strength,
                              render_resolution, render_samples,
                              transparent_bg=True)
            if not ok:
                continue
        rec["gt"] = gt_path

        skip = False
        for tag, pred_env in model_preds.items():
            png = os.path.join(out_dir, f"eval_{pos:05d}_{tag}.png")
            if not os.path.exists(png):
                hdr = os.path.join(out_dir, f"eval_{pos:05d}_{tag}.hdr")
                try:
                    save_envmap_as_hdr(pred_env[pos], hdr)
                except Exception as exc:
                    print(f"[teapot-multi] save_hdr {tag} {pos}: {exc}")
                    skip = True
                    break
                ok = _run_blender(blender_bin, blender_script, teapot_path,
                                  hdr, png, 0.0, 1.0,
                                  render_resolution, render_samples,
                                  transparent_bg=True)
                if not ok:
                    skip = True
                    break
            rec[tag] = png
        if skip:
            continue
        records.append(rec)
        if (k + 1) % 5 == 0:
            print(f"[teapot-multi] {k+1}/{len(subset_positions)}")

    return records


def lpips_on_records(records, model_tags, device="cpu", batch_size=8):
    """For each model_tag, compute paired LPIPS(GT, model) across records.
    Returns dict tag -> list[float] of length len(records).
    """
    if not records:
        return {}
    import lpips as _lpips_mod
    import cv2
    net = _lpips_mod.LPIPS(net="alex", verbose=False).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    # Find common target HW across all renders to handle mixed caches.
    target_hw = None
    keys = ["gt"] + list(model_tags)
    for r in records:
        for k in keys:
            im = cv2.imread(r[k], cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            hw = (im.shape[0], im.shape[1])
            target_hw = hw if target_hw is None else (
                min(target_hw[0], hw[0]), min(target_hw[1], hw[1]))

    arrs = {k: [] for k in keys}
    for r in records:
        for k in keys:
            arrs[k].append(_load_rgba_over_gray(r[k], target_hw=target_hw))
    g = np.stack(arrs["gt"])

    def _batch(g, p):
        out = np.empty(g.shape[0], dtype=np.float32)
        with torch.no_grad():
            for s in range(0, g.shape[0], batch_size):
                e = min(s + batch_size, g.shape[0])
                ta = torch.from_numpy(g[s:e]).to(device)
                tb = torch.from_numpy(p[s:e]).to(device)
                out[s:e] = net(ta, tb).view(-1).cpu().numpy()
        return out.tolist()

    return {tag: _batch(g, np.stack(arrs[tag])) for tag in model_tags}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(per_model_metrics, teapot_lpips_per_model):
    print()
    print("=" * 88)
    print(f"{'Model':<14}{'log_mse':>20}{'envmap LPIPS':>22}{'teapot LPIPS':>22}")
    print("-" * 88)
    for tag, m in per_model_metrics.items():
        lm = f"{m['log_mse_mean']:.4f} ± {m['log_mse_ci95_half']:.4f}"
        if "envmap_lpips_mean" in m:
            el = f"{m['envmap_lpips_mean']:.4f} ± {m['envmap_lpips_ci95_half']:.4f}"
        else:
            el = "N/A"
        tl_vals = teapot_lpips_per_model.get(tag)
        if tl_vals:
            tl_stats = mean_std_ci(tl_vals)
            tl = f"{tl_stats['mean']:.4f} ± {tl_stats['ci95_half']:.4f}"
        else:
            tl = "N/A"
        print(f"{tag:<14}{lm:>20}{el:>22}{tl:>22}")
    print("=" * 88)

    # Pairwise paired t-tests on teapot LPIPS vs your multitask (the headline
    # comparison: does your in-domain model beat OOD baselines on the
    # downstream render?)
    if "multitask" in teapot_lpips_per_model:
        ref = teapot_lpips_per_model["multitask"]
        print()
        print("Paired t-test on teapot LPIPS  (ref = multitask):")
        for tag, vals in teapot_lpips_per_model.items():
            if tag == "multitask" or not vals:
                continue
            tt = paired_ttest(vals, ref)
            sign = "multitask better" if tt["mean_diff"] > 0 else f"{tag} better"
            print(f"  {tag:<14} mean_diff(other - mt)={tt['mean_diff']:+.4f} "
                  f"± {tt['ci95_half_diff']:.4f}  t={tt['t_stat']:+.3f} "
                  f"p={tt['p_value']:.3g}  ({sign})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emlight_npz", default=None)
    ap.add_argument("--luxdit_npz", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_teapot", type=int, default=50,
                    help="number of test samples to render teapots for")
    ap.add_argument("--render_resolution", type=int, default=256)
    ap.add_argument("--render_samples", type=int, default=16)
    ap.add_argument("--lpips_device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(config.device)

    # --- 1) user models
    user = load_user_results(device)
    target_env = user["multitask"]["target_env"]   # GT envmaps (N, 3, H, W)
    target_indices = user["multitask"]["target_indices"]

    model_preds = {
        "baseline":  user["baseline"]["pred_env"],
        "multitask": user["multitask"]["pred_env"],
    }

    # --- 2) external models
    if args.emlight_npz:
        em = load_external_npz(args.emlight_npz, target_indices)
        model_preds["emlight"] = em["pred_env"]
    if args.luxdit_npz:
        lx = load_external_npz(args.luxdit_npz, target_indices)
        model_preds["luxdit"] = lx["pred_env"]

    # --- 3) envmap metrics per model
    per_model_metrics = {}
    for tag, pe in model_preds.items():
        print(f"[compare] envmap metrics for {tag} ...")
        per_model_metrics[tag] = compute_envmap_metrics(
            pe, target_env, device=args.lpips_device)

    # --- 4) teapot renders for a deterministic subset
    n = len(target_indices)
    rng = np.random.RandomState(args.seed)
    n_teapot = min(args.n_teapot, n)
    subset_positions = sorted(rng.choice(n, size=n_teapot, replace=False).tolist())
    print(f"[compare] teapot subset: {n_teapot} samples")

    teapot_dir = os.path.join(args.out_dir, "teapot_renders")
    records = render_teapots_multi(
        model_preds=model_preds,
        target_indices=target_indices,
        subset_positions=subset_positions,
        out_dir=teapot_dir,
        render_resolution=args.render_resolution,
        render_samples=args.render_samples,
    )
    print(f"[compare] usable teapot triplets: {len(records)}")

    # --- 5) LPIPS on the rendered teapots, per model
    teapot_lpips = lpips_on_records(
        records, model_tags=list(model_preds.keys()),
        device=args.lpips_device)

    # --- 6) report + persist
    print_table(per_model_metrics, teapot_lpips)

    summary = {
        "per_model": {
            tag: {k: v for k, v in m.items()
                  if not k.endswith("_per_sample")}
            for tag, m in per_model_metrics.items()
        },
        "teapot_lpips_mean": {
            tag: float(np.mean(v)) if v else None
            for tag, v in teapot_lpips.items()
        },
        "n_teapot_records": len(records),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(args.out_dir, "per_sample.npz"),
        target_indices=target_indices,
        **{f"log_mse_{tag}": np.asarray(m["log_mse_per_sample"])
           for tag, m in per_model_metrics.items()},
        **{f"envmap_lpips_{tag}": np.asarray(m.get("envmap_lpips_per_sample", []))
           for tag, m in per_model_metrics.items()},
        **{f"teapot_lpips_{tag}": np.asarray(v)
           for tag, v in teapot_lpips.items()},
    )
    print(f"[compare] wrote {args.out_dir}/summary.json + per_sample.npz")


if __name__ == "__main__":
    main()
