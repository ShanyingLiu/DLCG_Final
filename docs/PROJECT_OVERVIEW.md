# Project Overview

Material-aware HDR lighting estimation from a single rendered sphere. Two
trained models are compared:

- **Baseline**: ResNet image encoder → CNN decoder → predicted HDR envmap.
- **Multi-task**: Same encoder, plus a material-parameter regression head, with
  the material signal fed into the lighting decoder. Two conditioning variants:
  - `tiled` — material features tiled spatially and concatenated to backbone
    features (material-*aware*).
  - `film`  — material parameters drive FiLM (γ, β) modulation in every decoder
    block (material-*guided*).

Inputs are 128×128 sphere renders. Output envmaps are equirectangular
`(3, 64, 128)` HDR.

---

## Directory layout

```
code/
  main.py                # entry point: train + evaluate
  config.py              # all hyperparameters
  data_gen.py            # Blender script: generate sphere renders + metadata
  preprocess_hdris.py    # cache HDRIs as 64×128 .npy files
  render_teapot.py       # Blender script: render Utah teapot under one HDRI
  render_scene_landscape.py  # one-off scene render (sphere + teapot in HDRI)
  dataset/
    data.py              # SphereDataset: image + envmap + material-params loader
  models/
    baseline_model.py    # BaselineLightingNet
    multitask_model.py   # MaterialAwareLightingNet (tiled), MaterialGuidedLightingNet (FiLM)
    envmap_decoder.py    # EnvmapDecoder + FiLMEnvmapDecoder
  training/
    losses.py            # log-HDR MSE + L1 + SSIM + peak weighting
    train.py             # Trainer: train_epoch / validate / fit / checkpoint
  evaluation/
    metrics.py           # log_mse, paired_ttest, lpips_per_sample, ...
    evaluator.py         # evaluate_model + compute_all_metrics + compare_models
  utils/
    visualizer.py        # envmap GT/pred/error figures + training curves
    teapot_compare.py    # render-LPIPS evaluation on Blender teapots
    sh_utils.py          # (placeholder) spherical-harmonics helpers
docs/
  PROJECT_OVERVIEW.md    # this file
dataset/                 # raw HDRIs, generated renders, cached envmaps (gitignored)
result/                  # eval outputs, figures, teapot renders (gitignored)
models/                  # checkpoints, JSON metrics, eval logs (gitignored)
```

---

## Main pipeline (`main.py`)

A single CLI drives the whole project. The flow is:

1. **Parse args + override config**
   - `--mode {baseline, multitask, both}`
   - `--multitask-arch {tiled, film}` — picks model class + checkpoint tag
   - `--epochs / --batch-size / --backbone` — config overrides
   - `--eval-only` — skip training, load `<exp>_<tag>_best.pt`

2. **Build dataloaders** (`make_dataloaders`)
   - 70/15/15 train/val/test split, seeded by `config.seed` so the test set is
     reproducible across runs.
   - Test loader is `shuffle=False` — `target_indices[i]` is a stable mapping
     to `metadata.json`.

3. **Train (or load)**
   - `train_model(config, is_multitask, ...)` instantiates the right model
     class, optimizer (AdamW), and `CosineAnnealingLR(T_max=num_epochs)`.
   - `Trainer.fit` runs the epochs, checkpointing to
     `<exp>_<tag>_best.pt` when val loss improves.

4. **Run evaluation** (`run_evaluation`, mirrored to
   `<exp>_eval_log.txt` via `_tee_stdout`):
   1. `evaluate_model` → forward each model over the test loader, collect
      arrays: `pred_env`, `target_env`, `target_material_params`,
      `target_material_label`, `target_indices`, `pred_material_params`.
   2. `compute_all_metrics` → log_mse + LPIPS + per-material + per-bucket +
      material MAE.
   3. `compare_models` → side-by-side table + paired t-tests on log_mse and
      LPIPS.
   4. **Shiny-subset eval**: filter to roughness < 0.2 AND metallic > 0.8
      (~455 samples). Re-run `compute_all_metrics` and `compare_models` on
      that mask. This is the slice where material-aware lighting *should*
      matter most.
   5. **Render-LPIPS on shiny subset**: for each of the 455 shiny samples,
      render a Utah teapot in Blender three times (GT envmap, baseline
      prediction, multitask prediction) with transparent background, then
      LPIPS-AlexNet on the alpha-composited renders. Paired t-test on the
      perceptual deltas. Renders are cached on disk.
   6. **Plots**: per-material bar chart, per-attribute bucket chart,
      log_mse distribution, material-param MAE bars, envmap GT/pred/error
      figures, teapot showcase renders.
   7. **Persist**: `<exp>_results.json` with stripped metrics + paired t-test.

---

## The training objective

Lighting loss (in log-HDR space, `eps = 1.0` ≈ `log1p`):

$$
\mathcal{L}_{\text{light}} = w_{\text{mse}}\,\mathcal{L}_{\text{MSE}}^{\text{log}}
                            + w_{\ell_1}\,\mathcal{L}_{\ell_1}^{\text{log}}
                            + w_{\text{ssim}}\,(1 - \text{SSIM}^{\text{log}})
$$

with per-pixel peak weighting on the MSE/L1 terms (sun pixels carry the
structure):

$$
w(p) = 1 + \lambda \cdot \log(1 + y(p)),
\qquad
\mathcal{L}_{\text{MSE}}^{\text{log}}
= \frac{\sum_p w(p)\,\big(\log(\hat{y}(p)+\epsilon)-\log(y(p)+\epsilon)\big)^2}
       {\sum_p w(p)}
$$

Material loss: plain MSE on the 5-vector
$(metallic, roughness, specular, transmission, ior_{\text{norm}})$, where
`ior` is min-max-normalized to `[0, 1]`.

Total:

$$
\mathcal{L} = \alpha_{\text{light}}\,\mathcal{L}_{\text{light}}
            + \alpha_{\text{mat}}\,\mathcal{L}_{\text{mat}}
$$

Defaults: $\alpha_{\text{light}} = 1.0$, $\alpha_{\text{mat}} = 0.1$,
$w_{\text{mse}}=1.0$, $w_{\ell_1}=0.1$, $w_{\text{ssim}}=0.2$,
$\lambda=0.5$ (sun pixels weigh ~5–9× more than dark sky).

---

## Model architectures

Shared backbone for all three models: ImageNet-pretrained ResNet-50, with
everything but `layer4` frozen. Output spatial features are
`(B, 2048, 4, 4)` for a 128×128 input.

### Baseline (`BaselineLightingNet`)

```
image (B,3,128,128) ──ResNet50──▶ feat (B,2048,4,4) ──EnvmapDecoder──▶ envmap (B,3,64,128)
```

### Multi-task tiled (`MaterialAwareLightingNet`)

```
                ┌── avg-pool + Linear(2048,128) ─▶ mat_features (B,128) ─▶ Linear(128,5) ─▶ σ ─▶ material (B,5)
feat (B,2048,4,4)
                └── concat(feat, tile(mat_features, 4×4)) ─▶ EnvmapDecoder ─▶ envmap
```

The decoder receives the same backbone features plus a 128-channel constant
feature map of material context — material-aware but the decoder has to
discover how to use it.

### Multi-task FiLM (`MaterialGuidedLightingNet`)

```
                ┌── ... ─▶ material (B,5) ─▶ FiLM-MLP ─▶ (γ,β) for each decoder block
feat (B,2048,4,4)
                └── FiLMEnvmapDecoder(feat, (γ,β)) ─▶ envmap
```

The decoder is forced to *use* the predicted material because it modulates
every upsample block. FiLM layers apply per-channel feature-wise linear
modulation after BN, before ReLU:

$$
\text{FiLM}(x;\gamma,\beta) \;=\; (1 + \gamma) \odot x + \beta
$$

with γ, β ∈ ℝ^{C_{\text{out}}} broadcast across spatial dims. The
identity-initialization (`(1 + γ)`, last MLP layer zero-init) ensures
training begins from a non-conditioned decoder and learns conditioning
gradually — no destabilization at step 0.

### Decoder ladder (shared shape)

```
proj 1×1   :  (in_ch) → 256          @ 4×4
up1        :  256     → 128          @ 8×8
up2        :  128     →  64          @ 16×16
up3        :   64     →  32          @ 32×32
up4        :   32     →  16          @ 64×64
up_w       :   16     →  16          @ 64×128   (asymmetric stride 1×2)
head + softplus : 16  →   3          @ 64×128   (non-negative HDR)
```

Each `upN` is `ConvTranspose2d(k=4, s=2) → BN → [FiLM] → ReLU`.
FiLM dimension = `2 × (128 + 64 + 32 + 16) = 480`, generated by a
`Linear(5,64) → ReLU → Linear(64,480)` MLP whose last layer is zero-init.

---

## Evaluation in detail

### Headline metrics (full test set)

- **`log_mse`**: per-sample MSE in log-HDR space, identical formula to the
  loss without per-pixel weighting. Mean ± 95% CI; paired t-test
  (baseline − multitask) reported.
- **LPIPS-AlexNet**: per-sample on tonemapped envmaps. Each (pred, target)
  pair is `log1p`-mapped, normalized by the *target's* 99th percentile
  (same scale applied to pred so brightness errors aren't washed out),
  remapped to `[-1, 1]`. Same paired-test treatment.

### Slicings of interest

- **Per-material**: `diffuse / glossy / metallic / rough_metallic / dielectric`
  (5 categorical material types from `data_gen.py`).
- **Per-attribute bucket**: `metallic_yes/no`, `transmissive/opaque`,
  `rough_low/mid/high`. These are continuous — not the discrete material
  type — and reveal whether one model wins specifically for shiny vs rough
  objects.

### Shiny-subset analysis

Hard cutoff: `roughness < 0.2` AND `metallic > 0.8` (≈455 samples). All
metrics are recomputed on this slice. This is the test of the project's
core hypothesis ("material awareness helps the materials it should matter
for"). Per-material/per-bucket tables degenerate on the subset — they print
N/A for all but the metallic+rough_low cells.

### Render-LPIPS (perceptual end-to-end)

For each shiny sample we render a Utah teapot three times in Cycles:

- **GT** lit by the original `.hdr` (with metadata rotation/strength).
- **Baseline pred**, **Multitask pred** lit by the predicted envmap saved
  as `.hdr` (rotation/strength baked into the prediction).

Renders use `film_transparent=True` (envmap shows as α=0) and a fixed
Cycles seed. We alpha-composite each RGBA over neutral gray, compute
LPIPS-AlexNet against the GT render, and run a paired t-test
(baseline − multitask).

This is the *downstream* metric: how perceptually correct does a shiny
object look when re-lit by the predicted envmap. The envmap-pixel metrics
(log_mse, envmap-LPIPS) are proxies for this.

---

## Per-file notes

### Core pipeline

- **`code/main.py`** — argparse, dataloader assembly, model construction
  (chooses `MaterialAwareLightingNet` vs `MaterialGuidedLightingNet` based on
  `--multitask-arch`), checkpoint load/save, calls `run_evaluation`. Owns
  the eval-vs-train branching, the shiny-subset rerun, the render-LPIPS
  block, and all plot-saving.

- **`code/config.py`** — `Config` dataclass (image size, envmap dims, batch
  size, LR, loss weights, backbone, seed, paths). Single source of truth.

- **`code/dataset/data.py`** — `SphereDataset`. Reads `metadata.json`,
  loads each sphere render via PIL, loads the cached envmap from
  `dataset/envmaps/<stem>.npy`, applies the metadata's `world_strength` and
  rotates azimuth via `np.roll` (Z-axis rotation = horizontal pixel shift in
  equirectangular). Returns `image / lighting / material_params /
  material_label / index`. Deterministic 70/15/15 split seeded by `config.seed`.

- **`code/training/train.py`** — `Trainer` class. `train_epoch` and
  `validate` share boilerplate; supports both multi-task (envmap + material)
  and baseline (envmap only). Saves `<exp>_<tag>_best.pt` on best val loss.
  Tag is `multitask_film` or `multitask` depending on
  `config.multitask_arch`.

- **`code/training/losses.py`** — `MultiTaskLoss` plus the standalone
  `lighting_loss` used in baseline training. Implements `log_hdr_mse`,
  `log_hdr_l1`, peak-weighted variants, and a from-scratch SSIM (Gaussian
  window, log-HDR `data_range = log(1e6 + eps) ≈ 13.8`).

- **`code/models/baseline_model.py`** — `BaselineLightingNet`. ResNet
  backbone (head stripped) → `EnvmapDecoder`.

- **`code/models/multitask_model.py`** — both multitask classes:
  - `MaterialAwareLightingNet`: tiled material-feature concat path.
  - `MaterialGuidedLightingNet`: FiLM path (described above).

- **`code/models/envmap_decoder.py`** — `EnvmapDecoder` (used by baseline
  and tiled-multitask), and `FiLMEnvmapDecoder` + `FiLMUpBlock` (used by
  FiLM-multitask).

- **`code/evaluation/evaluator.py`** — `evaluate_model` runs forward over
  the test set; `compute_all_metrics` aggregates log_mse + LPIPS +
  per-material + per-bucket + material MAE; `compare_models` prints the
  side-by-side table and the paired t-tests.

- **`code/evaluation/metrics.py`** — `log_mse`, `mean_std_ci`,
  `paired_ttest`, `material_param_mae`, `per_material_metrics`,
  `lighting_metrics_by_bucket`, `default_param_buckets`, `lpips_per_sample`
  (lazily imports the `lpips` package, tonemaps HDR before scoring).

### Utility / one-shot scripts

- **`code/data_gen.py`** — Blender script. Generates the synthetic dataset:
  for each sample, picks a material category (5), samples Principled BSDF
  parameters (metallic/roughness/specular/transmission/ior), picks an HDRI
  with random Z rotation and strength, renders the sphere at 256×256,
  writes `metadata.json` with all material + lighting parameters.

- **`code/preprocess_hdris.py`** — caches every `.hdr/.exr` under
  `dataset/hdris/{puresky,scene}/` as a 64×128 float32 `.npy` envmap. Uses
  cv2 with `IMREAD_ANYDEPTH | IMREAD_ANYCOLOR` to preserve true HDR float
  values (imageio's default plugin silently quantizes to LDR).

- **`code/render_teapot.py`** — Blender script invoked by
  `teapot_compare.py`. Renders one Utah teapot under one HDRI; flags:
  `--rotation_z_deg`, `--strength`, `--resolution`, `--samples`,
  `--transparent-bg`, `--cycles-seed`.

- **`code/render_scene_landscape.py`** — one-off poster render: sphere +
  teapot floating in a 1920×1080 HDRI scene (`autumn_hill_view`). Not part
  of the training/eval pipeline.

- **`code/utils/visualizer.py`** — matplotlib helpers:
  - `plot_envmap_comparison` — GT / pred / abs-error 3-panel.
  - `plot_per_material_comparison` — bar chart over the 5 material types.
  - `plot_per_param_bucket_comparison` — bar chart over continuous attribute
    buckets.
  - `plot_log_mse_distribution` — paired-test spread histogram.
  - `plot_per_param_mae` — multitask material MAE bars.
  - `plot_training_curves` — train/val loss curves.

- **`code/utils/teapot_compare.py`** — two complementary roles:
  - `run_teapot_comparisons` — render `n_samples` *random* eval positions
    for visual-only side-by-side strips (fresh entropy per run).
  - `render_for_lpips` + `lpips_on_renders` — deterministic eval-mode
    rendering over the full shiny subset, alpha-composite over gray, run
    LPIPS-AlexNet, return paired arrays. Caches PNGs by index — re-runs
    skip rendered samples.

- **`code/utils/sh_utils.py`** — placeholder file (TODOs for future SH
  helpers). Not currently used.

---

## Reproducibility notes

- `config.seed = 42` controls the train/val/test split, weight
  initialization (via `torch.manual_seed`), and the test-loader is
  `shuffle=False`. Two `--eval-only` runs of the same checkpoint are
  bit-identical for all envmap-space metrics.
- The teapot-LPIPS render cache lives at
  `result/<exp>/teapot_lpips/eval_<i>_{gt,baseline,multitask}.png`. Cache
  is keyed on the *eval index*, not the model weights — **delete it
  before re-running after retraining**, or you'll score new predictions
  against stale renders.
- `Cycles` seed is pinned to 0, so a re-render of the same index is
  bit-identical.
- The visual-only teapot showcase renders (`run_teapot_comparisons`) use
  fresh entropy on purpose — they cycle through different test samples
  each run.
