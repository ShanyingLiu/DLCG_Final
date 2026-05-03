# CNN Envmap Decoder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the 27-dim SH lighting target with a directly-regressed 64×128 HDR equirectangular envmap, predicted by a CNN decoder operating on the ResNet50 spatial feature map. Both `BaselineLightingNet` and `MaterialAwareLightingNet` are upgraded; the loss switches to log-space MSE; eval/visualization keys are renamed to envmap-based equivalents.

**Architecture:**
- New preprocessing pass downsamples HDRI files to (64, 128, 3) float32 `.npy` envmaps, cached on disk under `dataset/envmaps/`.
- Dataset loads cached envmaps and applies per-render `world_strength` (scalar) and `rotation_z_rad` (azimuth roll along width axis) so the loaded envmap matches what Blender used when rendering.
- A shared `EnvmapDecoder` upsamples the (B, C, 4, 4) ResNet50 feature map to (B, 3, 64, 128) HDR through a sequence of ConvTranspose2d blocks, ending with Softplus.
- Loss is `log_hdr_mse(pred, target, eps=1.0)`; effectively MSE on `log1p(x)`.
- Evaluator/metrics/visualizer drop SH-specific functions (`angular_error`, `sh_mse`, `dominant_light_direction`, sphere render, SH bar charts) and gain `log_mse`, `linear_mse`, `psnr_log`, plus `plot_envmap_comparison`.
- `data_gen.py` and `sh_utils.py` are NOT modified — extra `sh_coefficients` field in metadata is harmless and silently ignored.

**Tech Stack:** PyTorch, torchvision, NumPy, imageio (HDR read), opencv-python or scikit-image (resize), matplotlib.

---

## Pre-flight verification

Before any code changes, confirm the following are true. Each is a single shell call.

- `python -c "import imageio.v3 as iio"` succeeds (install `imageio` + `imageio[freeimage]` if not).
- `python -c "import cv2"` succeeds, OR `python -c "import skimage.transform"` succeeds. Prefer cv2; fall back to skimage.
- `dataset/hdris/puresky` and `dataset/hdris/scene` exist and contain `.hdr`/`.exr` files (you've confirmed both).
- A sample render's metadata has `hdri_file`, `world_strength`, `rotation_z_rad`. (Check `dataset/renders/metadata.json` first entry.)

---

## Task 1: Preprocessing script (`code/preprocess_hdris.py`)

**Files:**
- Create: `code/preprocess_hdris.py`
- Output: `dataset/envmaps/<stem>.npy` for each HDR file

**Step 1.1: Verify metadata format**

Run: `python -c "import json; d=json.load(open('dataset/renders/metadata.json')); print(d[0])"`
Expected: keys include `hdri_file`, `world_strength`, `rotation_z_rad`.

**Step 1.2: Write the preprocessing script**

Create `code/preprocess_hdris.py`:

```python
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
```

**Step 1.3: Run preprocessing**

Run: `python code/preprocess_hdris.py`
Expected: prints progress, terminates with "done." Files appear in `dataset/envmaps/*.npy`.

**Step 1.4: Spot-check one envmap**

Run: `python -c "import numpy as np, os; f=sorted(os.listdir('dataset/envmaps'))[0]; a=np.load('dataset/envmaps/'+f); print(f, a.shape, a.dtype, a.min(), a.max(), a.mean())"`
Expected: shape (64, 128, 3), dtype float32, min ≥ 0, max in HDR range (likely > 1.0 for outdoor sky).

**Step 1.5: Commit**

```bash
git add code/preprocess_hdris.py
git commit -m "feat(envmap): add HDRI preprocessing to 64x128 .npy cache"
```

---

## Task 2: Config changes (`code/config.py`)

**Files:**
- Modify: `code/config.py`

**Step 2.1: Edit config**

Remove `sh_dim = 27`. Add envmap fields. Final block under `# Data`:

```python
    # Envmap (replaces SH lighting target)
    envmap_height = 64
    envmap_width = 128
    envmaps_root = "./dataset/envmaps"
    log_eps = 1.0  # log(eps + x) — log1p style for HDR loss
```

**Step 2.2: Sanity check imports**

Run: `python -c "from code.config import config; print(config.envmap_height, config.envmap_width, config.envmaps_root, config.log_eps); assert not hasattr(config, 'sh_dim')"`
Expected: prints `64 128 ./dataset/envmaps 1.0` and assertion holds.

(If `python -c "from code.config import ..."` fails because of how the project runs scripts, run from `code/` dir: `cd code && python -c "from config import config; ..."`.)

**Step 2.3: Commit**

```bash
git add code/config.py
git commit -m "feat(config): replace sh_dim with envmap dims and log_eps"
```

---

## Task 3: Dataset (`code/dataset/data.py`)

**Files:**
- Modify: `code/dataset/data.py`

**Step 3.1: Rewrite `SphereDataset`**

Replace the file with:

```python
# load data and do augmentation

import os
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SphereDataset(Dataset):
    def __init__(self, split, config, transform=None):
        self.config = config
        self.transform = transform

        metadata_path = os.path.join(config.metadata_root, "metadata.json")
        with open(metadata_path, 'r') as f:
            all_entries = json.load(f)

        self.image_paths = [
            os.path.join(config.images_root, entry["filename"])
            for entry in all_entries
        ]

        # Envmap path = envmaps_root / <stem>.npy where stem comes from hdri_file.
        self.envmap_paths = [
            os.path.join(config.envmaps_root,
                         os.path.splitext(entry["hdri_file"])[0] + ".npy")
            for entry in all_entries
        ]
        self.world_strengths = np.array(
            [entry["world_strength"] for entry in all_entries], dtype=np.float32
        )
        self.rotation_z_rads = np.array(
            [entry["rotation_z_rad"] for entry in all_entries], dtype=np.float32
        )

        self.material_labels = np.array(
            [entry["material_label"] for entry in all_entries], dtype=np.int64
        )

        ior_range = config.ior_max - config.ior_min
        self.material_params = np.array([
            [
                entry["metallic"],
                entry["roughness"],
                entry["specular"],
                entry["transmission"],
                (entry["ior"] - config.ior_min) / ior_range,
            ]
            for entry in all_entries
        ], dtype=np.float32)

        n = len(all_entries)
        rng = np.random.RandomState(config.seed)
        indices = rng.permutation(n)

        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        if split == "train":
            self.indices = indices[:n_train]
        elif split == "val":
            self.indices = indices[n_train:n_train + n_val]
        elif split == "test":
            self.indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split '{split}', expected train/val/test")

    def __len__(self):
        return len(self.indices)

    def _load_envmap(self, real_idx):
        env = np.load(self.envmap_paths[real_idx])  # (H, W, 3) float32
        env = env * float(self.world_strengths[real_idx])

        # Z-rotation: shift columns along width. Equirect azimuth maps to a
        # horizontal pixel roll. Sign chosen to match Blender's mapping-node
        # Z rotation; verify empirically (see plan task "rotation sanity").
        H, W, _ = env.shape
        rot = float(self.rotation_z_rads[real_idx])
        shift = int(round(rot / (2.0 * np.pi) * W))
        if shift != 0:
            env = np.roll(env, shift, axis=1)
        return env  # (H, W, 3)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])

        image = Image.open(self.image_paths[real_idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        env = self._load_envmap(real_idx)                        # (H, W, 3)
        env_t = torch.from_numpy(env).permute(2, 0, 1).contiguous()  # (3, H, W)

        return {
            'image': image,
            'lighting': env_t,
            'material_params': torch.FloatTensor(self.material_params[real_idx]),
            'material_label': torch.LongTensor([self.material_labels[real_idx]])[0],
            'index': real_idx,
        }
```

**Step 3.2: Smoke-test the dataset**

Run from `code/` dir:
```bash
cd code && python -c "
from config import config
from dataset.data import SphereDataset
ds = SphereDataset('train', config, transform=None)
sample = ds[0]
print('image:', type(sample['image']))
print('lighting:', sample['lighting'].shape, sample['lighting'].dtype,
      sample['lighting'].min().item(), sample['lighting'].max().item())
print('material_params:', sample['material_params'].shape)
"
```
Expected: lighting shape `torch.Size([3, 64, 128])` dtype `torch.float32`, min ≥ 0.

**Step 3.3: Rotation sanity check (manual visual)**

Save 3 envmaps from the dataset (entries with distinct `rotation_z_rad`) as tonemapped PNGs and compare against the corresponding rendered images — features in the envmap (sun, bright wall) should appear at the angle the render shows. If sign is wrong, flip `shift = -shift`. This is a one-shot visual confirmation, not an automated test.

```bash
cd code && python -c "
import numpy as np, matplotlib.pyplot as plt
from config import config
from dataset.data import SphereDataset
ds = SphereDataset('train', config)
for i in [0, 50, 100]:
    s = ds[i]
    env = s['lighting'].permute(1,2,0).numpy()
    tone = env / (1.0 + env)
    plt.imsave(f'/tmp/envmap_check_{i}.png', np.clip(tone, 0, 1))
    print(i, 'rot_rad=', float(ds.rotation_z_rads[ds.indices[i]]))
"
```
Inspect the renders for those same `index` values and confirm rotation matches.

**Step 3.4: Commit**

```bash
git add code/dataset/data.py
git commit -m "feat(data): load cached envmaps with strength + Z-rotation"
```

---

## Task 4: Loss (`code/training/losses.py`)

**Files:**
- Modify: `code/training/losses.py`

**Step 4.1: Add `log_hdr_mse` and update `MultiTaskLoss`**

Replace the file with:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def log_hdr_mse(pred, target, eps: float = 1.0):
    """MSE in log-HDR space: ((log(pred+eps) - log(target+eps))^2).mean().

    eps=1.0 yields effectively log1p, which is well-behaved across HDR
    range [0, ~10k] and at exactly zero. pred is assumed non-negative
    (e.g. Softplus output); target is non-negative HDR irradiance.
    """
    return F.mse_loss(torch.log(pred + eps), torch.log(target + eps))


class MultiTaskLoss(nn.Module):
    """Combined lighting (log-HDR MSE) + material parameter regression (MSE)."""

    def __init__(self, config):
        super().__init__()
        self.lighting_weight = config.lighting_loss_weight
        self.material_weight = config.material_loss_weight
        self.log_eps = float(getattr(config, "log_eps", 1.0))
        self.material_loss = nn.MSELoss()

    def forward(self, pred_lighting, pred_material,
                target_lighting, target_material):
        loss_lighting = log_hdr_mse(pred_lighting, target_lighting,
                                    eps=self.log_eps)
        loss_material = self.material_loss(pred_material, target_material)

        total = (self.lighting_weight * loss_lighting +
                 self.material_weight * loss_material)
        return total, loss_lighting, loss_material
```

**Step 4.2: Update Trainer baseline path**

In `code/training/train.py`, replace the two occurrences of:

```python
loss_light = torch.nn.functional.mse_loss(pred_light, target_sh)
```

with:

```python
loss_light = log_hdr_mse(pred_light, target_sh, eps=self.config.log_eps)
```

And add at the top: `from training.losses import log_hdr_mse`.

(The rest of train.py is structurally unchanged — `target_sh` is still the dict key locally; rename to `target_env` only if you also rename the batch key, which we are NOT doing. The dataset keeps `'lighting'` as the key and the variable name `target_sh` becomes a misnomer — rename to `target_light` for clarity in the same edit.)

Concretely, in both `train_epoch` and `validate`, rename local `target_sh` → `target_light` and update both branches accordingly.

**Step 4.3: Sanity test the loss**

```bash
cd code && python -c "
import torch
from training.losses import log_hdr_mse
a = torch.zeros(2, 3, 4, 4)
b = torch.full_like(a, 100.0)
print('log_hdr_mse(zero, zero):', log_hdr_mse(a, a).item())
print('log_hdr_mse(zero, 100):', log_hdr_mse(a, b).item())
"
```
Expected: first prints `0.0`; second prints a positive value (`~ (log(101))^2 = ~21.4`).

**Step 4.4: Commit**

```bash
git add code/training/losses.py code/training/train.py
git commit -m "feat(loss): swap SH MSE for log-HDR MSE in baseline + multitask"
```

---

## Task 5: Models — `EnvmapDecoder` + both networks

**Files:**
- Create: `code/models/envmap_decoder.py`
- Modify: `code/models/baseline_model.py`
- Modify: `code/models/multitask_model.py`

### Step 5.1: Write the decoder

Create `code/models/envmap_decoder.py`:

```python
"""Shared CNN decoder: ResNet50 spatial feats (B, C, 4, 4) -> HDR envmap (B, 3, 64, 128)."""

import torch
import torch.nn as nn


class EnvmapDecoder(nn.Module):
    """Upsample (B, in_channels, 4, 4) -> (B, 3, 64, 128) HDR envmap.

    Channel ladder:  in_channels -> 256 (1x1) -> 128 -> 64 -> 32 -> 16 -> 3.
    Spatial ladder:  4 -> 8 -> 16 -> 32 -> 64 -> (64, 128) via final stride (1,2).
    Output activation: Softplus, ensures non-negative HDR with smooth gradient.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 256, kernel_size=1)

        self.up1 = self._block(256, 128)   #  4 -> 8
        self.up2 = self._block(128, 64)    #  8 -> 16
        self.up3 = self._block(64, 32)     # 16 -> 32
        self.up4 = self._block(32, 16)     # 32 -> 64

        # Asymmetric upsample: H stays 64, W goes 64 -> 128.
        self.up_w = nn.ConvTranspose2d(
            16, 16, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
        )
        self.bn_w = nn.BatchNorm2d(16)
        self.act_w = nn.ReLU(inplace=True)

        self.head = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.out_act = nn.Softplus()

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.proj(x)        # (B, 256, 4, 4)
        x = self.up1(x)         # (B, 128, 8, 8)
        x = self.up2(x)         # (B, 64, 16, 16)
        x = self.up3(x)         # (B, 32, 32, 32)
        x = self.up4(x)         # (B, 16, 64, 64)
        x = self.act_w(self.bn_w(self.up_w(x)))  # (B, 16, 64, 128)
        x = self.head(x)        # (B, 3, 64, 128)
        return self.out_act(x)
```

**Step 5.1b: Decoder shape test**

```bash
cd code && python -c "
import torch
from models.envmap_decoder import EnvmapDecoder
dec = EnvmapDecoder(2048)
x = torch.zeros(2, 2048, 4, 4)
y = dec(x)
print(y.shape, y.min().item(), y.max().item())
assert tuple(y.shape) == (2, 3, 64, 128), y.shape
"
```
Expected: shape `(2, 3, 64, 128)`, min ≥ 0.

### Step 5.2: Rewrite `BaselineLightingNet`

Replace `code/models/baseline_model.py`:

```python
# baseline: predict envmap with only visuals (no material head)

import torch.nn as nn
import torchvision.models as models

from models.envmap_decoder import EnvmapDecoder


BACKBONE_FEATURES = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


class BaselineLightingNet(nn.Module):
    """Single-task model: ResNet backbone -> EnvmapDecoder -> HDR envmap."""

    def __init__(self, config):
        super().__init__()
        backbone_name = config.backbone
        feat_dim = BACKBONE_FEATURES[backbone_name]

        backbone_fn = getattr(models, backbone_name)
        weights = "IMAGENET1K_V1" if config.pretrained else None
        backbone = backbone_fn(weights=weights)
        # Strip avgpool + fc: keep spatial feats from layer4 (4x4 @ image=128).
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        if config.pretrained:
            for name, param in self.features.named_parameters():
                if not name.startswith('7'):  # layer4 is child index 7
                    param.requires_grad = False

        self.decoder = EnvmapDecoder(feat_dim)

    def forward(self, x):
        feat = self.features(x)        # (B, feat_dim, 4, 4)
        return self.decoder(feat)      # (B, 3, 64, 128)
```

### Step 5.3: Rewrite `MaterialAwareLightingNet`

Replace `code/models/multitask_model.py`:

```python
# 2-head model predicting material parameters and envmap, with material
# features tiled into the lighting decoder's spatial input.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.envmap_decoder import EnvmapDecoder


BACKBONE_FEATURES = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}

MATERIAL_HIDDEN = 128


class MaterialAwareLightingNet(nn.Module):
    """Multi-task model: shared spatial features -> material regression head
    + envmap decoder. Material hidden vector (B, 128) is tiled across the
    4x4 spatial grid and concatenated with backbone feats before decoding."""

    def __init__(self, config):
        super().__init__()
        backbone_name = config.backbone
        feat_dim = BACKBONE_FEATURES[backbone_name]

        backbone_fn = getattr(models, backbone_name)
        weights = "IMAGENET1K_V1" if config.pretrained else None
        backbone = backbone_fn(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        if config.pretrained:
            for name, param in self.features.named_parameters():
                if not name.startswith('7'):
                    param.requires_grad = False

        self.material_encoder = nn.Sequential(
            nn.Linear(feat_dim, MATERIAL_HIDDEN),
            nn.ReLU(inplace=True),
        )
        self.material_dropout = nn.Dropout(0.3)
        self.material_regressor = nn.Linear(MATERIAL_HIDDEN, config.num_material_params)

        self.decoder = EnvmapDecoder(feat_dim + MATERIAL_HIDDEN)

    def forward(self, x):
        feat = self.features(x)                        # (B, C, 4, 4)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, C)

        mat_features = self.material_encoder(pooled)   # (B, 128)
        material = torch.sigmoid(
            self.material_regressor(self.material_dropout(mat_features))
        )

        # Tile material features across the spatial grid and concat.
        B, _, H, W = feat.shape
        mat_tiled = mat_features.view(B, MATERIAL_HIDDEN, 1, 1).expand(B, MATERIAL_HIDDEN, H, W)
        decoder_in = torch.cat([feat, mat_tiled], dim=1)  # (B, C+128, 4, 4)

        envmap = self.decoder(decoder_in)              # (B, 3, 64, 128)
        return envmap, material
```

### Step 5.4: Forward-pass test for both models

```bash
cd code && python -c "
import torch
from config import config
from models.baseline_model import BaselineLightingNet
from models.multitask_model import MaterialAwareLightingNet
config.pretrained = False  # skip downloads
b = BaselineLightingNet(config)
m = MaterialAwareLightingNet(config)
x = torch.zeros(2, 3, config.image_size, config.image_size)
yb = b(x)
ym, mat = m(x)
print('baseline env:', yb.shape, 'mt env:', ym.shape, 'mt mat:', mat.shape)
assert tuple(yb.shape) == (2, 3, 64, 128)
assert tuple(ym.shape) == (2, 3, 64, 128)
assert tuple(mat.shape) == (2, 5)
"
```
Expected: shapes match.

**Step 5.5: Commit**

```bash
git add code/models/envmap_decoder.py code/models/baseline_model.py code/models/multitask_model.py
git commit -m "feat(model): replace SH heads with shared EnvmapDecoder"
```

---

## Task 6: Metrics (`code/evaluation/metrics.py`)

**Files:**
- Modify: `code/evaluation/metrics.py`

**Step 6.1: Replace SH metrics with envmap metrics**

Keep `mean_std_ci`, `paired_ttest`, `material_param_mae`, helper functions. Drop `dominant_light_direction`, `angular_error`, `relative_intensity_error`, `sh_mse`. Replace `lighting_metrics_by_bucket`, `per_material_metrics` to use the new envmap-based per-sample metrics.

Add at the top alongside existing helpers:

```python
def linear_mse(pred_env, target_env):
    """MSE in linear HDR space, per-sample. Inputs (3, H, W) or (H, W, 3)."""
    p = np.asarray(pred_env, dtype=np.float64).ravel()
    t = np.asarray(target_env, dtype=np.float64).ravel()
    return float(np.mean((p - t) ** 2))


def log_mse(pred_env, target_env, eps: float = 1.0):
    """MSE in log-HDR space, per-sample. Mirrors training loss."""
    p = np.log(np.asarray(pred_env, dtype=np.float64) + eps).ravel()
    t = np.log(np.asarray(target_env, dtype=np.float64) + eps).ravel()
    return float(np.mean((p - t) ** 2))


def psnr_log(pred_env, target_env, eps: float = 1.0):
    """PSNR computed in log-HDR space.

    Treats log(env+eps) as the signal. Peak chosen as 1.0 since log values
    are unbounded; this gives a comparable-to-itself number across runs but
    not a calibrated PSNR in the photographic sense.
    """
    mse = log_mse(pred_env, target_env, eps=eps)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))
```

Replace `lighting_metrics_by_bucket` body to compute `log_mse`, `linear_mse`, `psnr_log` per bucket. Replace `per_material_metrics` similarly. Both should:
- iterate over masked indices,
- collect the three per-sample arrays,
- report mean ± CI for `log_mse` (the primary), and means for `linear_mse` and `psnr_log`,
- keep `count`.

Concrete replacement for `per_material_metrics`:

```python
def per_material_metrics(pred_env_all, target_env_all, material_labels, num_classes=5):
    material_names = ["diffuse", "glossy", "metallic", "rough_metallic", "dielectric"]
    results = {}

    for label in range(num_classes):
        mask = np.asarray(material_labels) == label
        count = int(mask.sum())
        if count == 0:
            results[material_names[label]] = {
                "log_mse": None, "log_mse_std": None, "log_mse_ci95_half": None,
                "linear_mse": None, "psnr_log": None, "count": 0,
            }
            continue
        idxs = np.where(mask)[0]
        log_mses = [log_mse(pred_env_all[i], target_env_all[i]) for i in idxs]
        lin_mses = [linear_mse(pred_env_all[i], target_env_all[i]) for i in idxs]
        psnrs    = [psnr_log(pred_env_all[i], target_env_all[i])  for i in idxs]
        s = mean_std_ci(log_mses)
        results[material_names[label]] = {
            "log_mse": s["mean"],
            "log_mse_std": s["std"],
            "log_mse_ci95_half": s["ci95_half"],
            "linear_mse": float(np.mean(lin_mses)),
            "psnr_log": float(np.mean(psnrs)),
            "count": count,
        }
    return results
```

And the matching `lighting_metrics_by_bucket`:

```python
def lighting_metrics_by_bucket(pred_env_all, target_env_all, buckets):
    results = {}
    n_total = len(pred_env_all)
    for name, mask in buckets.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != n_total:
            raise ValueError(f"bucket '{name}' mask length {mask.shape[0]} "
                             f"does not match N={n_total}")
        count = int(mask.sum())
        if count == 0:
            results[name] = {"log_mse": None, "log_mse_std": None,
                             "log_mse_ci95_half": None,
                             "linear_mse": None, "psnr_log": None, "count": 0}
            continue
        idxs = np.where(mask)[0]
        log_mses = [log_mse(pred_env_all[i], target_env_all[i]) for i in idxs]
        lin_mses = [linear_mse(pred_env_all[i], target_env_all[i]) for i in idxs]
        psnrs    = [psnr_log(pred_env_all[i], target_env_all[i])  for i in idxs]
        s = mean_std_ci(log_mses)
        results[name] = {
            "log_mse": s["mean"],
            "log_mse_std": s["std"],
            "log_mse_ci95_half": s["ci95_half"],
            "linear_mse": float(np.mean(lin_mses)),
            "psnr_log": float(np.mean(psnrs)),
            "count": count,
        }
    return results
```

`default_param_buckets` and `material_param_mae` are unchanged.

**Step 6.2: Test the new metrics**

```bash
cd code && python -c "
import numpy as np
from evaluation.metrics import log_mse, linear_mse, psnr_log
a = np.zeros((3,64,128), dtype=np.float32)
b = np.zeros_like(a); b[...] = 1.0
print(linear_mse(a,a), linear_mse(a,b))
print(log_mse(a,a), log_mse(a,b))
print(psnr_log(a,a), psnr_log(a,b))
"
```
Expected: zeros for identical inputs; positive numbers for distinct inputs.

**Step 6.3: Commit**

```bash
git add code/evaluation/metrics.py
git commit -m "feat(metrics): replace SH metrics with log_mse/linear_mse/psnr_log"
```

---

## Task 7: Evaluator (`code/evaluation/evaluator.py`)

**Files:**
- Modify: `code/evaluation/evaluator.py`

**Step 7.1: Rename throughout**

In `evaluate_model`:
- Rename `all_pred_sh` → `all_pred_env`, `all_target_sh` → `all_target_env`.
- `target_sh = batch['lighting'].numpy()` is fine (the key is still `'lighting'`); rename local var to `target_env`.
- Result keys: `pred_sh` → `pred_env`, `target_sh` → `target_env`. Other keys unchanged.

In `compute_all_metrics`:
- Read `pred_env`/`target_env`.
- Compute per-sample arrays via `log_mse`, `linear_mse`, `psnr_log` (instead of `angular_error`, `relative_intensity_error`, `sh_mse`).
- `aggregate` block uses keys: `log_mse_mean/std/ci95_half`, `linear_mse_mean/std/ci95_half`, `psnr_log_mean/std/ci95_half`, `n_samples`.
- `per_sample` keeps the primary per-sample series for paired t-test: `"log_mse": [...]` (replaces `"angular_error"`).
- `per_material` and `per_param_bucket` use the rewritten functions.

In `compare_models`:
- Replace metric labels and keys to read from the new aggregate keys.
- The paired t-test now compares per-sample `log_mse` between baseline and multitask.
- Per-material loop now reads `log_mse` and `log_mse_ci95_half`.

Imports change to:

```python
from evaluation.metrics import (
    log_mse,
    linear_mse,
    psnr_log,
    material_param_mae,
    per_material_metrics,
    lighting_metrics_by_bucket,
    default_param_buckets,
    mean_std_ci,
    paired_ttest,
)
```

**Step 7.2: Smoke test**

```bash
cd code && python -c "
import numpy as np
from evaluation.evaluator import compute_all_metrics
N = 8
res = {
    'pred_env': np.random.rand(N, 3, 64, 128).astype(np.float32),
    'target_env': np.random.rand(N, 3, 64, 128).astype(np.float32) * 5.0,
    'target_material_label': np.random.randint(0, 5, size=N),
    'target_material_params': np.random.rand(N, 5).astype(np.float32),
    'pred_material_params': None,
}
m = compute_all_metrics(res, material_param_names=['metallic','roughness','specular','transmission','ior'])
print(list(m['aggregate'].keys()))
print(list(m['per_material'].keys()))
"
```
Expected: aggregate keys include `log_mse_mean`, `linear_mse_mean`, `psnr_log_mean`, `n_samples`.

**Step 7.3: Commit**

```bash
git add code/evaluation/evaluator.py
git commit -m "feat(eval): switch evaluator to envmap predictions"
```

---

## Task 8: Visualizer (`code/utils/visualizer.py`)

**Files:**
- Modify: `code/utils/visualizer.py`

**Step 8.1: Drop SH functions, add envmap comparison**

Remove: `_eval_sh_basis`, `_sh_to_color`, `_make_sphere_image`, `render_sphere_comparison`, `plot_sh_coefficients`.

Add:

```python
def _tonemap(x):
    """Reinhard-style tonemap for HDR display. x assumed non-negative."""
    return np.clip(x / (1.0 + x), 0.0, 1.0)


def plot_envmap_comparison(pred_env, target_env, save_path, title=None):
    """3-panel figure: GT envmap, predicted envmap, abs-error heatmap.

    Args:
        pred_env, target_env: arrays shape (3, H, W) OR (H, W, 3).
        save_path: output PNG path.
    """
    def _to_hwc(x):
        x = np.asarray(x)
        if x.ndim == 3 and x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))
        return x

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
```

Update `plot_per_material_comparison` to read `log_mse` instead of `angular_error`, and update axis label to "Log-HDR MSE".

`plot_training_curves`'s y-label should read "Lighting Loss (log-HDR MSE)" for the lighting subplot. Otherwise unchanged.

**Step 8.2: Smoke test**

```bash
cd code && python -c "
import numpy as np, tempfile, os
from utils.visualizer import plot_envmap_comparison
p = np.random.rand(3, 64, 128).astype(np.float32) * 2.0
t = np.random.rand(3, 64, 128).astype(np.float32) * 5.0
out = os.path.join(tempfile.gettempdir(), 'envcmp.png')
plot_envmap_comparison(p, t, out, title='test')
print('wrote', out, os.path.exists(out))
"
```
Expected: file written.

**Step 8.3: Commit**

```bash
git add code/utils/visualizer.py
git commit -m "feat(vis): add envmap comparison plot, drop SH sphere/bar charts"
```

---

## Task 9: Main entry point (`code/main.py`)

**Files:**
- Modify: `code/main.py`

**Step 9.1: Update imports**

```python
from utils.visualizer import (
    plot_envmap_comparison,
    plot_per_material_comparison,
    plot_training_curves,
)
```

(Drop `render_sphere_comparison`, `plot_sh_coefficients`.)

**Step 9.2: Update metric prints**

Replace the per-model "Baseline test metrics" / "Multitask test metrics" blocks to print from new aggregate keys:

```python
agg = baseline_metrics["aggregate"]
print(f"  log_mse:    {agg['log_mse_mean']:.4f} ± {agg['log_mse_ci95_half']:.4f}")
print(f"  linear_mse: {agg['linear_mse_mean']:.4f} ± {agg['linear_mse_ci95_half']:.4f}")
print(f"  psnr_log:   {agg['psnr_log_mean']:.2f} ± {agg['psnr_log_ci95_half']:.2f} dB")
```

**Step 9.3: Update the visualization loop**

Replace the sphere + SH bar chart block with an envmap comparison loop:

```python
for tag, raw_results in [("baseline", baseline_results),
                          ("multitask", multitask_results)]:
    if raw_results is None:
        continue
    n_vis = min(5, len(raw_results["pred_env"]))
    for i in range(n_vis):
        pred = raw_results["pred_env"][i]
        target = raw_results["target_env"][i]
        env_path = os.path.join(vis_dir, f"{tag}_envmap_{i}.png")
        plot_envmap_comparison(pred, target, env_path,
                               title=f"{tag.title()} Sample {i}")
```

**Step 9.4: Update paired t-test persistence**

Change `b_ang = baseline_metrics.get("per_sample", {}).get("angular_error")` (and the multitask one) to read `"log_mse"` instead. Saved key becomes `paired_ttest_log_mse`.

**Step 9.5: End-to-end smoke run (1 epoch, tiny batch)**

```bash
cd code && python main.py --mode multitask --epochs 1 --batch-size 4
```
Expected: training runs, validation prints, evaluation runs, results JSON + envmap PNGs written under `result/material_aware_lighting/`. No exceptions. PNG visually shows tonemapped GT vs predicted envmaps and an error heatmap.

If GPU memory is tight, reduce batch size further. The point of this run is wiring, not performance.

**Step 9.6: Commit**

```bash
git add code/main.py
git commit -m "feat(main): wire envmap targets through training, eval, and viz"
```

---

## Task 10: Final verification

**Step 10.1: Full pipeline run for both models**

```bash
cd code && python main.py --mode both --epochs 2 --batch-size 8
```
Expected: completes without errors, saves both checkpoints, writes comparison output.

**Step 10.2: Confirm artifacts exist**

```bash
ls models/material_aware_lighting_baseline_best.pt models/material_aware_lighting_multitask_best.pt models/material_aware_lighting_results.json result/material_aware_lighting/
```
Expected: all files present; `result/.../` contains envmap PNGs and the per-material chart.

**Step 10.3: Final commit**

```bash
git add -u
git commit -m "chore: end-to-end verification of envmap pipeline"
```

---

## Out of scope (explicit non-goals)

- `code/data_gen.py` is unchanged. The `sh_coefficients` field it writes into metadata is silently ignored by the new dataset loader. Avoids a multi-hour re-render.
- `code/utils/sh_utils.py` is unchanged (still imported by `data_gen.py`).
- No changes to image rendering, no new transforms on the input image, no changes to backbone freezing strategy.

## New dependencies

- `imageio` (and `imageio[freeimage]` for HDR/EXR support if not already present)
- `opencv-python` preferred for resize; `scikit-image` is the fallback. Either is fine.

Install if missing:

```bash
pip install "imageio[freeimage]" opencv-python
```
