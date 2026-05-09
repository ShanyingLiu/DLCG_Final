"""Run EMLight's pretrained DenseNet over the test split and save predicted
envmaps as a (.npz) file aligned positionally with the dataloader order.

Why this script (rather than EMLight/RegressionNetwork/test.py): EMLight's
test.py expects Laval Indoor crops, hardcodes paths, and pulls in util.py
which currently has unresolved git merge-conflict markers. We avoid all of
that — we import only DenseNet (clean) and re-implement the small panorama
reconstruction we need.

Output:
    --out_npz contains:
        pred_env       : (N, 3, 64, 128) float32  HDR linear, scaled by intensity
        target_indices : (N,)            int64     metadata.json indices, in eval order
        meta : dict with model='emlight', notes
"""

import argparse
import os
import sys
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

# Local project imports (run from repo root via `python -m code.external.emlight_infer`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from config import config
from dataset.data import SphereDataset

# EMLight DenseNet
EMLIGHT_DIR = os.path.join(os.path.dirname(__file__),
                           os.pardir, os.pardir, "EMLight", "RegressionNetwork")
sys.path.insert(0, EMLIGHT_DIR)
import DenseNet  # noqa: E402


# -- panorama reconstruction (re-implemented from EMLight/util.py to avoid the
#    merge-conflict markers in that file). 96 anchor directions on the sphere
#    via a Fibonacci lattice; for each, broadcast a Gaussian-on-sphere kernel
#    onto a 128x256 equirect grid, weighted by the predicted color.

def fibonacci_sphere_points(n: int) -> np.ndarray:
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1.0 - 1.0 / n, 1.0 / n - 1.0, n)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    pts = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    return pts.astype(np.float32)  # (n, 3)


def reconstruct_panorama(distribution, intensity, rgb_ratio,
                         n_anchors=96, H=128, W=256, kernel_size=0.0025,
                         device="cpu"):
    """Inputs:
        distribution: (n_anchors,) softmaxed weights
        intensity:    scalar (already scaled by the *500 factor used in test.py)
        rgb_ratio:    (3,)
    Returns: (3, H, W) float32 HDR linear panorama.
    """
    dirs = torch.from_numpy(fibonacci_sphere_points(n_anchors)).to(device)  # (A,3)

    # equirect grid: latitude in [0, pi], longitude in [0, 2pi]
    lat = (torch.arange(H, dtype=torch.float32, device=device) + 0.5) * (np.pi / H)
    lon = (torch.arange(W, dtype=torch.float32, device=device) + 0.5) * (2 * np.pi / W)
    lat_g, lon_g = torch.meshgrid(lat, lon, indexing="ij")
    x = torch.sin(lat_g) * torch.cos(lon_g)
    y = torch.sin(lat_g) * torch.sin(lon_g)
    z = torch.cos(lat_g)
    xyz = torch.stack([x, y, z], dim=0).reshape(3, -1)            # (3, H*W)

    # Gaussian-on-sphere kernel: exp((dot - 1)/sigma)
    dots = dirs @ xyz                                              # (A, H*W)
    kern = torch.exp((dots - 1.0) / float(kernel_size))            # (A, H*W)

    weight = (torch.from_numpy(distribution).to(device).float()
              * float(intensity))                                  # (A,)
    color = torch.from_numpy(np.asarray(rgb_ratio, dtype=np.float32)).to(device)  # (3,)

    # (A, H*W) -> (H*W,)  then * (3,) broadcast
    radiance = (weight[:, None] * kern).sum(dim=0)                 # (H*W,)
    pano = radiance[None, :] * color[:, None]                      # (3, H*W)
    pano = pano.reshape(3, H, W).clamp_min(0.0).cpu().numpy()
    return pano.astype(np.float32)


def resize_envmap(env_3hw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize (3, H, W) envmap by area-average to (3, target_h, target_w)."""
    import cv2
    chw = env_3hw
    hwc = np.transpose(chw, (1, 2, 0))
    res = cv2.resize(hwc, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return np.transpose(res, (2, 0, 1)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="path to EMLight latest_net.pth")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--input_size", type=int, default=192,
                    help="DenseNet was trained on Laval crops. The released "
                         "checkpoint's FC head expects 8208 features. The "
                         "default DenseNet config in EMLight's repo does NOT "
                         "produce 8208 with any common input size — when you "
                         "download the ckpt, inspect its state_dict to figure "
                         "out the right (growth_rate, num_init_features, "
                         "input_size). Use the flags below to adjust.")
    ap.add_argument("--growth_rate", type=int, default=12)
    ap.add_argument("--num_init_features", type=int, default=24)
    ap.add_argument("--block_config", type=str, default="16,16,16",
                    help="comma-separated layer counts per dense block")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available()
                                         else "cpu"))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--intensity_scale", type=float, default=500.0,
                    help="EMLight test.py multiplies intensity head by 500; "
                         "predictions on out-of-domain LDR inputs are likely "
                         "miscalibrated, so this acts as a knob.")
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) Build the same test split your evaluator uses. We override image_size
    #    to feed EMLight at its expected input resolution.
    tfm = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),  # already in [0,1]; EMLight's tonemap pipeline normally
                       # produces a similar range from HDR — feeding LDR PNGs
                       # directly is the OOD compromise documented in PLAN.md
    ])
    ds = SphereDataset("test", config, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    # 2) Load EMLight DenseNet. Architecture is tweakable from CLI because the
    #    repo's default __init__ args do NOT match the released checkpoint's
    #    8208-feature linear head — peek at the .pth keys and adjust if needed.
    block_config = tuple(int(x) for x in args.block_config.split(","))
    model = DenseNet.DenseNet(growth_rate=args.growth_rate,
                              block_config=block_config,
                              num_init_features=args.num_init_features
                              ).to(device).eval()
    state = torch.load(args.weights, map_location=device)
    # Some checkpoints are wrapped in DataParallel; strip the prefix if so.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[emlight] load_state_dict: missing={len(missing)} "
              f"unexpected={len(unexpected)} (this is fine if minor)")

    # 3) Run inference. Reconstruct pano per-sample (96 anchors → 128x256),
    #    then resize to your target shape (64x128 by default).
    target_h, target_w = config.envmap_height, config.envmap_width
    pred_env_all = []
    target_indices = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            out = model(x)
            dist = torch.softmax(out["distribution"], dim=1).cpu().numpy()  # (B, 96)
            inten = out["intensity"].cpu().numpy().reshape(-1)              # (B,)
            rgb = out["rgb_ratio"].cpu().numpy()                            # (B, 3)
            idxs = batch["index"].numpy()

            for b in range(x.shape[0]):
                pano128 = reconstruct_panorama(
                    distribution=dist[b],
                    intensity=float(inten[b]) * args.intensity_scale,
                    rgb_ratio=rgb[b],
                    n_anchors=96,
                    H=128, W=256, kernel_size=0.0025,
                    device=device,
                )
                env_small = resize_envmap(pano128, target_h, target_w)
                pred_env_all.append(env_small)
                target_indices.append(int(idxs[b]))

            print(f"[emlight] {len(pred_env_all)}/{len(ds)}", flush=True)

    pred_env = np.stack(pred_env_all, axis=0).astype(np.float32)
    target_indices = np.asarray(target_indices, dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        pred_env=pred_env,
        target_indices=target_indices,
        model="emlight",
        notes=("EMLight DenseNet pretrained on Laval Indoor; "
               "OOD on synthetic spheres; camera-frame panorama (no z-roll)."),
    )
    print(f"[emlight] wrote {args.out_npz}: pred_env={pred_env.shape}")


if __name__ == "__main__":
    main()
