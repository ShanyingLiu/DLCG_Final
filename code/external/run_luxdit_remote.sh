#!/usr/bin/env bash
# Run on the NVIDIA GPU box after rsync'ing LuxDiT/ and luxdit_inputs/ over.
#
# Expected layout:
#   /workspace/LuxDiT/                  ← cloned repo + checkpoints/
#   /workspace/luxdit_inputs/           ← {00000.png, 00001.png, ..., eval_index.json}
#   /workspace/luxdit_outputs/          ← created here (ldr_log/, hdr/)
#
# After this finishes, rsync /workspace/luxdit_outputs/hdr/ back to your laptop.

set -euo pipefail

LUXDIT_DIR=${LUXDIT_DIR:-/workspace/LuxDiT}
INPUT_DIR=${INPUT_DIR:-/workspace/luxdit_inputs}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/luxdit_outputs}

DIT_PATH=${LUXDIT_DIR}/checkpoints/luxdit_image
HDR_MERGE_PATH=${LUXDIT_DIR}/checkpoints/hdr_merge_mlp

cd "$LUXDIT_DIR"

# We use the synthetic-rendering image preset (no LoRA). Your inputs are
# Cycles renders of a sphere, which is closer to LuxDiT's synthetic
# pretraining than to its scene-photo LoRA distribution.
#
# resolution 480 720 is the LuxDiT-recommended inference resolution; output
# envmaps are dual-tone-mapped at the model's native res, then merged to HDR.
python inference_luxdit.py \
    --config configs/luxdit_base.yaml \
    --transformer_path "$DIT_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resolution 480 720 \
    --guidance_scale 2.5 \
    --num_inference_steps 50 \
    --seed 33

python hdr_merger.py \
    --model_path "$HDR_MERGE_PATH" \
    --input_dir "$OUTPUT_DIR/ldr_log" \
    --output_dir "$OUTPUT_DIR/hdr"

echo "[luxdit-remote] done. HDR EXRs in $OUTPUT_DIR/hdr"
