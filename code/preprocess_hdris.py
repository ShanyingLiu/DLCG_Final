"""Preprocess HDR/EXR envmaps to cached 64x128 .npy arrays.

Walks dataset/hdris/{puresky,scene}/, loads each .hdr/.exr (skipping
duplicate downloads like 'name (1).hdr'), downsamples to (64, 128, 3)
float32 via area-averaging, and saves under dataset/envmaps/<stem>.npy.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio

try:
    import cv2
    _RESIZE_BACKEND = "cv2"
except ImportError:
    from skimage.transform import resize as _sk_resize
    _RESIZE_BACKEND = "skimage"


HDRI_ROOT = Path("dataset/hdris")
OUT_ROOT = Path("dataset/envmaps")
TARGET_H = 64
TARGET_W = 128


def discover_hdris(root: Path):
    """Mirror data_gen.discover_hdris: walk puresky+scene, skip '(1)' dupes."""
    found = []
    for category in ("puresky", "scene"):
        cat_dir = root / category
        if not cat_dir.exists():
            continue
        for f in sorted(cat_dir.iterdir()):
            if f.suffix.lower() not in (".hdr", ".exr"):
                continue
            if re.search(r"\(\d+\)", f.stem):
                continue
            found.append(f)
    return found


def downsample(envmap: np.ndarray) -> np.ndarray:
    """Area-averaged downsample to TARGET_H x TARGET_W. Preserves HDR energy."""
    if _RESIZE_BACKEND == "cv2":
        # cv2.resize takes (W, H)
        out = cv2.resize(envmap, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    else:
        out = _sk_resize(
            envmap, (TARGET_H, TARGET_W),
            order=1, anti_aliasing=True, preserve_range=True,
        )
    return out.astype(np.float32)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    files = discover_hdris(HDRI_ROOT)
    if not files:
        print(f"[preprocess_hdris] No HDR/EXR files found under {HDRI_ROOT}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[preprocess_hdris] backend={_RESIZE_BACKEND}, "
          f"found {len(files)} HDRI files")

    for i, src in enumerate(files):
        out_path = OUT_ROOT / f"{src.stem}.npy"
        if out_path.exists():
            continue
        env = iio.imread(src).astype(np.float32)
        if env.ndim != 3 or env.shape[2] < 3:
            print(f"[preprocess_hdris] SKIP {src.name}: shape {env.shape}",
                  file=sys.stderr)
            continue
        env = env[..., :3]  # drop alpha if present
        small = downsample(env)
        np.save(out_path, small)
        if (i + 1) % 25 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {src.name} -> {out_path.name} "
                  f"shape={small.shape} min={small.min():.4f} max={small.max():.2f}")

    print(f"[preprocess_hdris] done. Wrote envmaps to {OUT_ROOT}")


if __name__ == "__main__":
    main()
