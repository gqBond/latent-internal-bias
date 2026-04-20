#!/usr/bin/env bash
# Round-3 GPU launch plan (after Round-2 reviewer 4/10 verdict).
#
# Reviewer priority ranking:
#   1. Full-answer scoring (000..999 / candidate strings) — fixes first-digit artifact.
#   2. Null-prompt calibration sweep — directly tests whether "argmax=9" is removable.
#   3. MCQ task — sanity check on balanced natural-label task.
#   4. Scale (1.5B, 14B) — ONLY after readout is fixed.
#
# We run goals 1+2 on 7B before spending budget on scale. If the 7B sanity pass
# shows the digit-prior artifact reduced and the conditional-σ signal survives,
# then proceed to 1.5B + 14B. Else, pivot before scaling broken readout.
#
# Budget (A100-40G):
#   - 7B null-prompt calibration re-extract (AIME + MATH500)  : ~1.5 GPU-hr
#   - 7B full-answer re-extract (AIME, n=30, K≈15 candidates) : ~2.0 GPU-hr
#   - 7B full-answer re-extract (MATH500 subset)              : ~2.0 GPU-hr
#   - 1.5B + 14B replication with fixes (conditional)         : ~5.0 GPU-hr
# Total ~10 GPU-hr. Stop at step 3 if step 2 reveals readout still broken.
set -euo pipefail

MATH500_N=100
AIME_SPLIT=aime2024
CFG7=configs/r1_qwen_7b.yaml
OUT7=results/lib/DeepSeek-R1-Distill-Qwen-7B

NULL_PROMPT="Provide a number for this {format} answer."

# ------------------- (A) Null-prompt calibration re-extract on 7B -------------------
for SPLIT in $AIME_SPLIT math500; do
  EXTRA=""
  [ "$SPLIT" = "math500" ] && EXTRA="--limit $MATH500_N --seed 0"
  TAG="${SPLIT}_null"
  [ "$SPLIT" = "math500" ] && TAG="math500_n${MATH500_N}_s0_null"
  python -m scripts.extract_lib --cfg $CFG7 --split $SPLIT $EXTRA \
      --null-prompt "$NULL_PROMPT" \
      --out $OUT7/${TAG}_lib.jsonl
  python -m scripts.analysis_lib --cfg $CFG7 \
      --lib $OUT7/${TAG}_lib.jsonl \
      --out $OUT7/${TAG}_summary.json
done

# ------------------- (B) Full-answer scoring re-extract on 7B -------------------
for SPLIT in $AIME_SPLIT math500; do
  EXTRA=""
  [ "$SPLIT" = "math500" ] && EXTRA="--limit $MATH500_N --seed 0"
  TAG="${SPLIT}_fullans"
  [ "$SPLIT" = "math500" ] && TAG="math500_n${MATH500_N}_s0_fullans"
  python -m scripts.extract_lib --cfg $CFG7 --split $SPLIT $EXTRA \
      --full-answer \
      --null-prompt "$NULL_PROMPT" \
      --out $OUT7/${TAG}_lib.jsonl
  python -m scripts.analysis_lib --cfg $CFG7 \
      --lib $OUT7/${TAG}_lib.jsonl \
      --out $OUT7/${TAG}_summary.json
done

# --- Decision gate: inspect `calibration_sanity.final_layer.dominant_label_frac`.
#     If > 0.35 (was 0.70 without calibration), the readout is still biased.
#     Halt here and pivot. Else continue to scale replication.

# ------------------- (C) Scale replication on 1.5B + 14B (with fixes) -------------------
for CFG_NAME in r1_qwen_1_5b r1_qwen_14b; do
  CFG=configs/${CFG_NAME}.yaml
  OUT_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['model']['name'])")
  OUT=results/lib/${OUT_MODEL}

  for SPLIT in $AIME_SPLIT math500; do
    EXTRA=""
    [ "$SPLIT" = "math500" ] && EXTRA="--limit $MATH500_N --seed 0"
    TAG="${SPLIT}_fullans"
    [ "$SPLIT" = "math500" ] && TAG="math500_n${MATH500_N}_s0_fullans"
    python -m scripts.extract_lib --cfg $CFG --split $SPLIT $EXTRA \
        --full-answer \
        --null-prompt "$NULL_PROMPT" \
        --out $OUT/${TAG}_lib.jsonl
    python -m scripts.analysis_lib --cfg $CFG \
        --lib $OUT/${TAG}_lib.jsonl \
        --out $OUT/${TAG}_summary.json
  done
done

echo "Round-3 experiments complete. Results under results/lib/*/*_fullans_*.json — sync back:"
echo "  rsync -av GPU_SERVER:~/Internal_bias/results/ ./results/"
echo ""
echo "Then check each summary's calibration_sanity.final_layer.dominant_label_frac:"
echo "  python -c \"import json,glob; [print(f, json.load(open(f))['calibration_sanity']['final_layer']['dominant_label_frac']) for f in glob.glob('results/lib/*/*_fullans_summary.json')]\""
