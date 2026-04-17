#!/usr/bin/env bash
# Round-2 GPU launch plan (triggered after Round-1 reviewer reframe).
#
# Three goals:
#   (1) Replicate on 1.5B and 14B to test δ scaling.
#   (2) Train the tuned-lens on 7B and re-extract to kill the raw-logit-lens
#       "argmax = 9 for everything" artifact on AIME24.
#   (3) Add an MCQ benchmark (Knowlogic) so we have a natural-label task.
#
# Budget (A100-40G):
#   - 1.5B pilot  AIME24 + MATH500 subset : ~1.5 GPU-hr
#   - 14B pilot   same                    : ~4.0 GPU-hr
#   - 7B tuned-lens train                 : ~0.3 GPU-hr
#   - 7B re-extract with tuned lens       : ~1.2 GPU-hr
#   - Knowlogic MCQ (all three models)    : ~2.0 GPU-hr
# Total ~9 GPU-hr. Well within the 48 GPU-hr remaining budget.
set -euo pipefail

MATH500_N=100
AIME_SPLIT=aime2024
KNOW_SPLIT=knowlogic-charcount

# ------------------- (1) 1.5B + 14B replication -------------------
for CFG_NAME in r1_qwen_1_5b r1_qwen_14b; do
  CFG=configs/${CFG_NAME}.yaml
  OUT_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['model']['name'])")
  OUT=results/lib/${OUT_MODEL}

  # AIME24
  python -m scripts.extract_lib --cfg $CFG --split $AIME_SPLIT \
      --out $OUT/${AIME_SPLIT}_lib.jsonl
  python -m scripts.analysis_lib --cfg $CFG \
      --lib $OUT/${AIME_SPLIT}_lib.jsonl \
      --out $OUT/${AIME_SPLIT}_summary.json

  # MATH500 subset
  python -m scripts.extract_lib --cfg $CFG --split math500 --limit $MATH500_N --seed 0 \
      --out $OUT/math500_n${MATH500_N}_s0_lib.jsonl
  python -m scripts.analysis_lib --cfg $CFG \
      --lib $OUT/math500_n${MATH500_N}_s0_lib.jsonl \
      --out $OUT/math500_n${MATH500_N}_s0_summary.json
done

# ------------------- (2) 7B tuned-lens -------------------
python -m scripts.train_tuned_lens --cfg configs/r1_qwen_7b.yaml
# Re-run extraction with tuned-lens (cfg has lens.type: tuned)
for SPLIT in $AIME_SPLIT math500; do
  EXTRA=""
  [ "$SPLIT" = "math500" ] && EXTRA="--limit $MATH500_N --seed 0"
  python -m scripts.extract_lib --cfg configs/r1_qwen_7b.yaml --split $SPLIT $EXTRA \
      --out results/lib/DeepSeek-R1-Distill-Qwen-7B/${SPLIT}_tunedlens_lib.jsonl
  python -m scripts.analysis_lib --cfg configs/r1_qwen_7b.yaml \
      --lib results/lib/DeepSeek-R1-Distill-Qwen-7B/${SPLIT}_tunedlens_lib.jsonl \
      --out results/lib/DeepSeek-R1-Distill-Qwen-7B/${SPLIT}_tunedlens_summary.json
done

# ------------------- (3) Knowlogic MCQ (all three models) -------------------
for CFG_NAME in r1_qwen_1_5b r1_qwen_7b r1_qwen_14b; do
  CFG=configs/${CFG_NAME}.yaml
  OUT_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['model']['name'])")
  python -m scripts.extract_lib --cfg $CFG --split $KNOW_SPLIT \
      --out results/lib/${OUT_MODEL}/${KNOW_SPLIT}_lib.jsonl
  python -m scripts.analysis_lib --cfg $CFG \
      --lib results/lib/${OUT_MODEL}/${KNOW_SPLIT}_lib.jsonl \
      --out results/lib/${OUT_MODEL}/${KNOW_SPLIT}_summary.json
done

echo "Round-2 experiments complete. Results under results/lib/*/*.json — sync back with:"
echo "  rsync -av GPU_SERVER:~/Internal_bias/results/ ./results/"
