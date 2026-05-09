# System diagram

Three views of the same system, increasing in detail:

1. End-to-end pipeline (one figure for the paper).
2. Multi-task model architecture (tiled vs FiLM side-by-side).
3. Training & evaluation flow (where each component lives).

---

## 1. End-to-end pipeline

```
 ┌──────────┐    ┌────────────────┐    ┌────────────────┐    ┌─────────────┐    ┌────────────────────┐
 │ HDRI     │───▶│ Blender Cycles │───▶│ metadata.json  │───▶│ Sphere      │───▶│ Baseline / Tiled / │
 │ .hdr/exr │    │ • sample BSDF  │    │ + sphere PNGs  │    │ Dataset     │    │ FiLM multi-task    │
 └──────────┘    │ • light sphere │    │ + envmap .npy  │    │ 70/15/15    │    │ networks           │
                 │ • render 256²  │    │ (64×128 HDR)   │    │ split       │    └─────────┬──────────┘
                 └────────────────┘    └────────────────┘    └─────────────┘              ▼
                                                                              predicted envmap (3, 64, 128)
                      ┌──────────────────────────┌────────────────────────────────────────┤
                      │                          │                                        │
                      ▼                          ▼                                        ▼
            ┌─────────────────┐         ┌─────────────────┐                ┌─────────────────────────┐
            │ envmap log_mse  │         │ envmap LPIPS    │                │ render-LPIPS on shiny   │
            │ + paired t-test │         │ + paired t-test │                │ teapots (Blender Cycles)│
            └─────────────────┘         └─────────────────┘                └─────────────────────────┘
```

---

## 2. Three model variants

All three share a frozen ResNet-50 backbone (only `layer4` trains) producing
spatial features `feat ∈ ℝ^{B×2048×4×4}`, and the same `EnvmapDecoder`-style
upsample ladder ending in Softplus. They differ only in **whether material
is predicted** and **how it conditions the decoder**.

### 2a. Baseline — lighting only

```
 image ─► ResNet-50 ─► feat (B, 2048, 4, 4) ─► EnvmapDecoder ─► envmap (B, 3, 64, 128)
                                               proj→up1→up2→up3→up4→up_w→head→Softplus
```

No material head, no conditioning. Decoder sees backbone features and nothing
else.

### 2b. Tiled multi-task — material-aware

```
                                                    ┌─► avgpool ─► Linear(2048→128) ─► ReLU ─► mat_features (B,128) ──┐
                                                    │                          │                                      │
                                                    │                          ▼                                      │ tile 4×4
 image ─► ResNet-50 ─► feat (B, 2048, 4, 4) ────────┤               Dropout ─► Linear(128→5) ─► σ ─► material (B,5)   │
                                                    │                                                                 │
                                                    │                                                                 ▼
                                                    └────────────────────────────────► concat ─► EnvmapDecoder ─► envmap
                                                                                       (2048+128 ch)
```

Material features are tiled across the 4×4 spatial grid and **concatenated**
to backbone features. The decoder receives a 2176-channel input and must
discover how to use the material context.

### 2c. FiLM multi-task — material-guided

```
                                                    ┌─► avgpool ─► Linear(2048→128) ─► ReLU ─► Dropout ─► Linear(128→5) ─► σ ─► material (B,5)
                                                    │                                                                              │
                                                    │                                                                              ▼
 image ─► ResNet-50 ─► feat (B, 2048, 4, 4) ────────┤                                                          ┌─────────────────────────────┐
                                                    │                                                          │ FiLM-MLP                    │
                                                    │                                                          │ Linear(5→64) ─► ReLU ─►     │
                                                    │                                                          │ Linear(64→480) (zero-init)  │
                                                    │                                                          └──────────────┬──────────────┘
                                                    │                                                                         │
                                                    │                                                          (γ, β) per up-block (B, 480)
                                                    │                                                                         │
                                                    ▼                                                                         ▼
                                       ┌──────────────────────────────────────────────────────────────────────────────────────┐
                                       │ FiLM-EnvmapDecoder                                                                   │
                                       │ proj→up1+FiLM₁→up2+FiLM₂→up3+FiLM₃→up4+FiLM₄→up_w→head→Softplus                       │
                                       │ 4×4 ──► 8×8 ──► 16×16 ──► 32×32 ──► 64×64 ──► 64×128 ──► (B, 3, 64, 128)              │
                                       └──────────────────────────────────────────────────────────────────────────────────────┘
                                                                                          │
                                                                                          ▼
                                                                            predicted envmap (B, 3, 64, 128)
```

The 5-d predicted material drives a small MLP that outputs per-block (γ, β)
vectors. These **modulate the post-BN feature map** at every upsample stage,
so the decoder is *forced* to use material — there's no concat path that
the decoder could ignore.

### FiLM modulation inside each `up_k` block

```
   x_in ─► ConvTranspose2d(s=2) ─► BatchNorm2d ─► FiLM: (1+γ)⊙x + β ─► ReLU ─► x_out
                                                        ▲
                                                        │
                                                   γ, β (B, Cₒᵤₜ)
```

$$
\text{FiLM}(x;\gamma,\beta) \;=\; (1 + \gamma) \odot x + \beta
$$

Zero-init of the last MLP layer means γ = β = 0 at step 0, so each block
starts as `ReLU(BN(ConvT(x)))` — a non-conditioned decoder. Conditioning
is *learned* over training.

### How the three compare

|                            | Baseline           | Tiled multi-task                | FiLM multi-task                  |
| -------------------------- | ------------------ | ------------------------------- | -------------------------------- |
| Material head?             | no                 | yes (5-d, supervised)           | yes (5-d, supervised)            |
| Material → decoder         | —                  | concat 128-d tiled to 4×4 grid  | (γ, β) modulation per up-block   |
| Decoder receives           | feat (2048ch)      | feat ⊕ mat_tiled (2176ch)       | feat (2048ch) + (γ, β) injection |
| Material params used       | —                  | 128-d intermediate `mat_features` | 5-d sigmoid predicted vector  |
| Conditioning bottleneck    | —                  | wide (128ch, decoder optional)  | narrow (5-d, every layer forced) |
| Identity at init?          | n/a                | no — material affects from step 1 | yes — γ=β=0 → unconditioned decoder |
| Extra params vs baseline   | 0                  | + Linear(2048→128) + 128→5 + bigger proj | + same material head + ~32K FiLM-MLP |
| Checkpoint tag             | `baseline`         | `multitask`                     | `multitask_film`                 |

```
   Baseline         Tiled (material-aware)          FiLM (material-guided)
   ──────────       ─────────────────────────       ──────────────────────────
   feat ──► Dec     feat ─┬─► concat ─► Dec         feat ──────────────► Dec
                   mat ──┘                               ▲
                                                         │ (γ,β)
                                                    material ─► FiLM-MLP

   no material      decoder *can* use mat            decoder *must* use mat
```

---

## 3. Training & evaluation flow

```
 ┌──────────────┐   ┌────────────────┐   ┌─────────────────┐                        ┌──────────────────────────┐
 │ config + CLI │──▶│ make_data      │──▶│ train_model     │── <exp>_<tag>_best.pt ─▶│ run_evaluation           │
 │ --mode/--arch│   │ loaders        │   │ AdamW + CosineLR│                        │ • evaluate_model         │
 │ --eval-only  │   │ train/val/test │   │ Trainer.fit     │                        │ • compute_all_metrics    │
 └──────────────┘   └────────────────┘   └────────┬────────┘                        │ • compare_models         │
                                                  │                                 │ • shiny-subset rerun     │
                                                  │  (--eval-only branches around)  │ • render-LPIPS (Blender) │
                                                  └─────────────────────────────────▶ • figures + JSON         │
                                                                                    └──────────────────────────┘
```

### The losses driving training

```
                       ┌─► log-HDR weighted MSE ──┐
   predicted envmap ───┼─► log-HDR weighted L1  ──┼──► L_light  ─┐
                       └─► (1 − log-HDR SSIM)   ──┘              │
                                                                 ├──► L_total = 1.0·L_light + 0.1·L_mat
   predicted material ─► MSE on (metallic, roughness, …) ─► L_mat ┘
```

Weights used:
`lighting_loss_weight=1.0`, `material_loss_weight=0.1`,
`mse=1.0`, `l1=0.1`, `ssim=0.2`, `peak_λ=0.5`, `log_eps=1.0`.
