# External baselines: EMLight + LuxDiT

Goal: compare two off-the-shelf lighting estimators against your trained
baseline / multitask models on the **same test split**, using your existing
metric stack (`log_mse`, envmap LPIPS, **teapot-render LPIPS**).

Both baselines are run *as-is* with their pretrained weights — no retraining.
This is fair given your project trains in-domain (synthetic spheres) while
they are out-of-domain (real photos / real renders); the comparison stresses
how each transfers.

---

## High-level flow

```
test split (your dataset, shuffle=False, deterministic order)
           │
           ├──► your baseline / multitask models   (already runs in main.py)
           │
           ├──► EMLight pretrained DenseNet         (emlight_infer.py)
           │           input:  same sphere PNGs (resized to 192x192, tonemapped path)
           │           output: pred_env_emlight.npz  shape (N, 3, 64, 128)
           │
           └──► LuxDiT image-DiT + HDR merger       (run on NVIDIA GPU)
                       input:  same sphere PNGs (480x720)            ← luxdit_prep.py
                       run:    inference_luxdit.py + hdr_merger.py   ← run_luxdit_remote.sh
                       output: hdr/*.exr                             → luxdit_postprocess.py
                                                                     → pred_env_luxdit.npz

run_external_compare.py
    → loads your models + the two .npz files
    → runs evaluator metrics (log_mse, LPIPS, per-material)
    → runs teapot renders + teapot-LPIPS for ALL four models
    → prints a single comparison table, saves figures
```

The `.npz` files store a `pred_env: (N, 3, 64, 128) float32` aligned to the
test split's `target_indices` (i.e. positionally aligned with the order
`evaluate_model()` returns), so they slot directly into the existing eval.

---

## Step-by-step

### 0. Environment

You need (locally OR on the GPU box; per script):

| Script                       | Where to run        | Python deps |
|------------------------------|---------------------|-------------|
| `emlight_infer.py`           | local (M2 OK) or GPU| torch, numpy, PIL, opencv |
| `luxdit_prep.py`             | local               | numpy, PIL, your repo |
| LuxDiT inference + merger    | **NVIDIA GPU only** | luxdit env (see below) |
| `luxdit_postprocess.py`      | local or GPU        | numpy, opencv, OpenEXR (or imageio) |
| `run_external_compare.py`    | local or GPU        | your project deps + lpips |

LuxDiT environment (on the GPU box):
```bash
conda create -n luxdit python=3.10
conda activate luxdit
# install pytorch 2.4 matching your CUDA, then:
cd LuxDiT && pip install -r requirements.txt
mkdir -p checkpoints && hf download nvidia/LuxDiT --local-dir checkpoints
```

EMLight pretrained weights:
```bash
# from the README — Google Drive link in EMLight/README.md
# place at: EMLight/RegressionNetwork/checkpoints/latest_net.pth
```

> **Note 1:** EMLight's `util.py` has unresolved git merge-conflict markers
> (`<<<<<<<`, `=======`, `>>>>>>>`). `emlight_infer.py` does **not** import
> `util.py` to avoid this; it re-implements the small panorama reconstruction
> we need. If you ever want to use the original `util.py` you'll need to
> resolve those conflicts manually.
>
> **Note 2:** EMLight's `DenseNet.py` declares its FC head as
> `nn.Linear(8208, 1024)` but the default constructor args
> (`growth_rate=12, block_config=(16,16,16), num_init_features=24,
> avgpool_size=4`) do not produce a 8208-feature output for any common input
> size. The released checkpoint must use a different config. **After
> downloading the weights**, inspect the state_dict and the conv channel
> counts to figure out the correct `growth_rate` / `num_init_features` /
> `block_config` / `input_size`, then pass them to `emlight_infer.py`:
> ```bash
> python -c "import torch; sd=torch.load('latest_net.pth', map_location='cpu'); \
>     print({k: tuple(v.shape) for k,v in sd.items() if k.endswith('.weight')})"
> ```
> Look at `features.conv0.weight` (gives `num_init_features`) and the
> conv shapes inside the dense blocks to derive `growth_rate`. The repo
> README also mentions reducing anchors to 96 — `--n_anchors` is hard-coded
> to 96 in `emlight_infer.py` to match the FC head.

---

### 1. EMLight (local laptop OK)

```bash
cd /Users/jasmineliu/Documents/DLCG_Final
python -m code.external.emlight_infer \
    --weights ./EMLight/RegressionNetwork/checkpoints/latest_net.pth \
    --out_npz ./result/material_aware_lighting/pred_env_emlight.npz
```

This (a) builds the same test split your evaluator uses, (b) feeds each PNG
through DenseNet, (c) reconstructs a 128×256 panorama from the 96-anchor
spherical distribution, (d) resizes to 64×128, and (e) saves a packed `.npz`.
Expect ~5–15 min on M2 CPU for ~1500 test samples.

---

### 2. LuxDiT — prep on laptop, run on GPU box

**On laptop:**
```bash
python -m code.external.luxdit_prep \
    --out_dir ./result/material_aware_lighting/luxdit_inputs \
    --copy_metadata
```
This dumps every test-split sphere PNG into a flat directory named
`{eval_pos:05d}.png` so positional alignment is trivial later, and writes
a `eval_index.json` mapping eval-position → metadata index.

**Copy to GPU box:**
```bash
rsync -avz result/material_aware_lighting/luxdit_inputs/ \
    gpu-box:/workspace/luxdit_inputs/
rsync -avz LuxDiT/ gpu-box:/workspace/LuxDiT/
```

**On GPU box:**
```bash
cd /workspace/LuxDiT
conda activate luxdit
bash /workspace/LuxDiT/run_luxdit_remote.sh   # see code/external/run_luxdit_remote.sh
```

That script runs `inference_luxdit.py` over the prepped folder and then
`hdr_merger.py` to produce `.exr` HDR envmaps in `luxdit_outputs/hdr/`.

**Copy results back:**
```bash
rsync -avz gpu-box:/workspace/luxdit_outputs/hdr/ \
    result/material_aware_lighting/luxdit_outputs/hdr/
```

**Postprocess on laptop:**
```bash
python -m code.external.luxdit_postprocess \
    --hdr_dir ./result/material_aware_lighting/luxdit_outputs/hdr \
    --eval_index ./result/material_aware_lighting/luxdit_inputs/eval_index.json \
    --out_npz ./result/material_aware_lighting/pred_env_luxdit.npz
```

---

### 3. Final unified comparison

```bash
python -m code.external.run_external_compare \
    --emlight_npz ./result/material_aware_lighting/pred_env_emlight.npz \
    --luxdit_npz  ./result/material_aware_lighting/pred_env_luxdit.npz \
    --out_dir ./result/material_aware_lighting/external_compare \
    --n_teapot 50
```

This:
1. Loads your `material_aware_lighting_baseline_best.pt` and
   `material_aware_lighting_multitask_best.pt`, runs them over the test set.
2. Loads the two external `.npz` files (positionally aligned).
3. Computes per-model envmap metrics (log_mse, LPIPS).
4. Renders Utah teapots for a subset (`--n_teapot`) under each model's
   predicted envmap + GT, computes **paired LPIPS on the rendered teapots**
   (your downstream metric).
5. Prints a 4-way table and writes a side-by-side figure.

---

## Caveats — flag in writeup

1. **Frame alignment.** Your GT envmap is rolled by `rotation_z_rad` so
   azimuth is in a world-anchored frame. EMLight's panorama is in *camera*
   frame (centered on the photo's view direction). LuxDiT's `.exr` after
   merge is also in *camera* frame (center = forward direction). Your
   teapot render bakes `rotation_z_deg=0, strength=1` for predictions —
   identical treatment for all four models — so the comparison is internally
   consistent, but absolute envmap metrics (log_mse) will be biased *against*
   EMLight/LuxDiT by their azimuth offset. We compensate by reporting
   teapot-render LPIPS as the headline metric (rotation-invariant for diffuse
   shading; less so for sharp speculars — note this).
2. **Tone-mapping mismatch.** EMLight expects an HDR crop tone-mapped via
   its `TonemapHDR(percentile=50, gamma=2.4)`. We feed it your LDR PNG
   directly (treating it as already-tonemapped). This is the only honest
   choice given you don't have HDR crops, but EMLight's intensity head
   will be miscalibrated. Document.
3. **Out-of-distribution.** Both baselines were trained on real photos
   (Laval Indoor / NVIDIA's mix). Your inputs are synthetic Cycles renders
   of a chrome/diffuse sphere on black. Expect both to underperform vs your
   in-domain models. The interesting comparison is *how badly* and *which
   structural cues survive*.
4. **LPIPS direction caveat.** Rendered-teapot LPIPS treats both predicted
   and GT shading as natural images; it rewards correct macro lighting
   direction and color cast more than precise sun-disk position. Pair with
   log_mse and visual inspection.

---

## Files

| File                                          | Purpose |
|-----------------------------------------------|---------|
| `code/external/emlight_infer.py`              | Run pretrained EMLight, save `.npz` |
| `code/external/luxdit_prep.py`                | Stage test inputs for LuxDiT |
| `code/external/luxdit_postprocess.py`         | Read LuxDiT EXRs, save `.npz` |
| `code/external/run_luxdit_remote.sh`          | Run on GPU box |
| `code/external/run_external_compare.py`       | Unified 4-model evaluator |
| `code/external/PLAN.md`                       | This file |
