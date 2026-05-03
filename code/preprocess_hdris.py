"""Preprocess HDR/EXR envmaps to cached 64x128 .npy arrays.

Walks dataset/hdris/{puresky,scene}/, loads each .hdr/.exr (skipping
duplicate downloads like 'name (1).hdr'), downsamples to (64, 128, 3)
float32 via area-averaging, and saves under dataset/envmaps/<stem>.npy.

Loading uses cv2.imread with IMREAD_ANYDEPTH | IMREAD_ANYCOLOR so the
real HDR float32 values are preserved (imageio's default plugin path
silently quantizes .hdr to LDR ~[0, 255], which destroys sun/sky
intensities and forces predicted envmaps toward grayscale).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
    _RESIZE_BACKEND = "cv2"
except ImportError:
    cv2 = None
    from skimage.transform import resize as _sk_resize
    _RESIZE_BACKEND = "skimage"

# imageio is the fallback HDR reader when cv2 is unavailable.
try:
    import imageio.v3 as iio
except ImportError:
    iio = None


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


def load_hdr(path: Path) -> np.ndarray:
    """Load .hdr/.exr as float32 RGB preserving full HDR dynamic range.

    Returns array shape (H, W, 3), float32, RGB order.
    """
    if cv2 is not None:
        flags = cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
        img = cv2.imread(str(path), flags)
        if img is None:
            raise IOError(f"cv2 failed to read {path}")
        # cv2 returns BGR; envmap convention is RGB.
        if img.ndim == 3 and img.shape[2] >= 3:
            img = img[..., :3][..., ::-1]
        return np.ascontiguousarray(img.astype(np.float32))

    if iio is not None:
        # The "HDR-FI" plugin in imageio returns true float32 HDR.
        img = iio.imread(str(path), plugin="HDR-FI")
        return np.ascontiguousarray(img.astype(np.float32))

    raise RuntimeError("Neither cv2 nor imageio is available for HDR loading.")


def downsample(envmap: np.ndarray) -> np.ndarray:
    """Peak-preserving downsample to TARGET_H x TARGET_W via block max-pool.

    Linear area averaging from a 2K HDRI (~16x16 source pixels per output
    pixel) smears single-pixel suns into surrounding sky, leaving the cached
    target with very few bright pixels. Max-pool keeps the brightest source
    pixel in each block so the supervision actually contains the spotlights
    we want the model to predict. Energy is not conserved, but the cached
    .npy is only used as a regression target -- renders use the original .hdr.
    """
    H_src, W_src = envmap.shape[:2]
    bh, bw = H_src // TARGET_H, W_src // TARGET_W
    if bh < 1 or bw < 1:
        # Source smaller than target along some axis: fall back to area resize.
        if _RESIZE_BACKEND == "cv2":
            out = cv2.resize(envmap, (TARGET_W, TARGET_H),
                             interpolation=cv2.INTER_AREA)
        else:
            out = _sk_resize(envmap, (TARGET_H, TARGET_W),
                             order=1, anti_aliasing=True, preserve_range=True)
        return out.astype(np.float32)
    H_use, W_use = bh * TARGET_H, bw * TARGET_W
    # Center-crop residual rows/cols so block reshape is exact.
    y0 = (H_src - H_use) // 2
    x0 = (W_src - W_use) // 2
    cropped = envmap[y0:y0 + H_use, x0:x0 + W_use]
    blocks = cropped.reshape(TARGET_H, bh, TARGET_W, bw, -1)
    out = blocks.max(axis=(1, 3))
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing .npy files (use this after "
                             "fixing the HDR loader to regenerate stale caches)")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    files = discover_hdris(HDRI_ROOT)
    if not files:
        print(f"[preprocess_hdris] No HDR/EXR files found under {HDRI_ROOT}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[preprocess_hdris] backend={_RESIZE_BACKEND}, "
          f"found {len(files)} HDRI files, force={args.force}")

    suspicious = 0
    for i, src in enumerate(files):
        out_path = OUT_ROOT / f"{src.stem}.npy"
        if out_path.exists() and not args.force:
            continue
        env = load_hdr(src)
        if env.ndim != 3 or env.shape[2] < 3:
            print(f"[preprocess_hdris] SKIP {src.name}: shape {env.shape}",
                  file=sys.stderr)
            continue
        env = env[..., :3]  # drop alpha if present
        small = downsample(env)
        # Sanity check: real HDR sky should have a long tail above 1.0; a hard
        # ceiling at exactly 255 is a strong sign the loader quantized to LDR.
        if small.max() <= 255.0 + 1e-6 and (small == small.max()).mean() > 1e-3:
            suspicious += 1
        np.save(out_path, small)
        if (i + 1) % 25 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {src.name} -> {out_path.name} "
                  f"shape={small.shape} min={small.min():.4f} max={small.max():.2f}")

    print(f"[preprocess_hdris] done. Wrote envmaps to {OUT_ROOT}")
    if suspicious:
        print(f"[preprocess_hdris] WARNING: {suspicious} file(s) look LDR-clipped "
              f"(max <= 255 with a saturation plateau). Verify the loader.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
