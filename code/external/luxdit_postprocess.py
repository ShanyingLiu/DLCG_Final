"""Read LuxDiT's merged HDR EXRs and pack them as a (.npz) aligned to the
test-split eval order, ready for run_external_compare.py.

Resizes envmaps to your project's target shape (envmap_height, envmap_width).
"""

import argparse
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from config import config


def _load_hdr(path: str) -> np.ndarray:
    """Return (H, W, 3) float32 RGB linear from .exr or .hdr."""
    # Prefer imageio-freeimage for EXR (LuxDiT default); fall back to OpenCV
    # for .hdr.
    try:
        import imageio.v3 as iio
        img = iio.imread(path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img.astype(np.float32)
    except Exception:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH
                         | cv2.IMREAD_ANYCOLOR)
        if img is None:
            raise RuntimeError(f"failed to read {path}")
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img[..., ::-1]  # BGR -> RGB
        return img.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdr_dir", required=True,
                    help="dir containing LuxDiT-merged *.exr (or *.hdr)")
    ap.add_argument("--eval_index", required=True,
                    help="eval_index.json from luxdit_prep.py")
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()

    with open(args.eval_index) as f:
        idx_map = json.load(f)  # {"00000": meta_idx, ...}
    n = len(idx_map)

    target_h, target_w = config.envmap_height, config.envmap_width
    pred_env = np.zeros((n, 3, target_h, target_w), dtype=np.float32)
    target_indices = np.zeros((n,), dtype=np.int64)

    found = 0
    missing = []
    import cv2
    for eval_pos in range(n):
        stem = f"{eval_pos:05d}"
        # LuxDiT names outputs after the input stem; tolerate .exr or .hdr
        cands = (glob.glob(os.path.join(args.hdr_dir, stem + ".exr"))
                 + glob.glob(os.path.join(args.hdr_dir, stem + ".hdr"))
                 + glob.glob(os.path.join(args.hdr_dir, stem + "*.exr"))
                 + glob.glob(os.path.join(args.hdr_dir, stem + "*.hdr")))
        target_indices[eval_pos] = int(idx_map[str(eval_pos)])
        if not cands:
            missing.append(eval_pos)
            continue
        hwc = _load_hdr(cands[0])
        # resize to project's envmap target
        small = cv2.resize(hwc, (target_w, target_h),
                           interpolation=cv2.INTER_AREA)
        pred_env[eval_pos] = np.transpose(small, (2, 0, 1))
        found += 1

    if missing:
        print(f"[luxdit-post] WARNING: {len(missing)} samples missing HDR; "
              f"first few: {missing[:10]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        pred_env=pred_env,
        target_indices=target_indices,
        model="luxdit",
        notes=("LuxDiT image-DiT + hdr_merge_mlp; merged HDR centered on "
               "input camera direction; no z-roll applied."),
    )
    print(f"[luxdit-post] wrote {args.out_npz}: pred_env={pred_env.shape}, "
          f"resolved={found}/{n}")


if __name__ == "__main__":
    main()
