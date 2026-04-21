#!/usr/bin/env bash
# Round-4 GPU launch (last round in MAX_ROUNDS=4).
#
# Round-3 reviewer verdict: 5/10, "not ready" but endorse-pivot.
# Highest-value remaining run: cross-scale full-answer on 1.5B and 14B to test
# whether the measurement-methodology claim (M1/M2/M3) generalizes, and whether
# the MATH500 P3 inversion (M4-exploratory) replicates across scale.
#
# Budget (A100-40G):
#   - 1.5B pilot AIME24 + MATH500  (null + full-answer)  : ~1.5 GPU-hr
#   - 14B  pilot AIME24 + MATH500  (null + full-answer)  : ~4.0 GPU-hr
# Total ~5.5 GPU-hr. Within the revised 5-GPU-hr budget if 14B uses fp16.
#
# If time left, optional:
#   - MCQ (Knowlogic) full-answer on 1.5B/7B/14B         : ~2.0 GPU-hr
set -euo pipefail

MATH500_N=100
AIME_SPLIT=aime2024
NULL_PROMPT="Provide a number for this {format} answer."

# ------------------- Cross-scale full-answer + null-calib sweep -------------------
for CFG_NAME in r1_qwen_1_5b r1_qwen_14b; do
  CFG=configs/${CFG_NAME}.yaml
  OUT_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['model']['name'])")
  OUT=results/lib/${OUT_MODEL}
  mkdir -p "$OUT"

  for SPLIT in $AIME_SPLIT math500; do
    EXTRA=""
    TAG_BASE="${SPLIT}"
    if [ "$SPLIT" = "math500" ]; then
      EXTRA="--limit $MATH500_N --seed 0"
      TAG_BASE="math500_n${MATH500_N}_s0"
    fi

    # Null-calibrated first-digit — keeps the three-readout triad complete.
    python -m scripts.extract_lib --cfg $CFG --split $SPLIT $EXTRA \
        --null-prompt "$NULL_PROMPT" \
        --out $OUT/${TAG_BASE}_null_lib.jsonl
    python -m scripts.analysis_lib --cfg $CFG \
        --lib $OUT/${TAG_BASE}_null_lib.jsonl \
        --out $OUT/${TAG_BASE}_null_summary.json \
        --min-cell-size 20

    # Full-answer — the paper's primary readout.
    python -m scripts.extract_lib --cfg $CFG --split $SPLIT $EXTRA \
        --full-answer \
        --null-prompt "$NULL_PROMPT" \
        --out $OUT/${TAG_BASE}_fullans_lib.jsonl
    python -m scripts.analysis_lib --cfg $CFG \
        --lib $OUT/${TAG_BASE}_fullans_lib.jsonl \
        --out $OUT/${TAG_BASE}_fullans_summary.json \
        --min-cell-size 20
  done
done

echo "Round-4 cross-scale runs complete."
echo ""
echo "M1 check (dominant_label_frac < 0.35 under full-answer):"
python3 - <<'PY'
import glob, json
for f in sorted(glob.glob("results/lib/*/*_fullans_summary.json")):
    try:
        s = json.load(open(f))
        d = s["calibration_sanity"]["final_layer"]["dominant_label_frac"]
        gt = s["calibration_sanity"]["final_layer"].get("correct_answer_dominant_frac")
        print(f"  {f}: dominant={d:.2f}  gt_dominant={gt if gt is None else f'{gt:.2f}'}  M1_pass={d < 0.35}")
    except Exception as e:
        print(f"  {f}: ERROR {e}")
PY

echo ""
echo "M4-exploratory check (MATH500 P3 inversion — bootstrap CI < 1.0):"
python3 - <<'PY'
import glob, json
for f in sorted(glob.glob("results/lib/*/math500_*_fullans_summary.json")):
    try:
        s = json.load(open(f))
        b = s["decomposition"]["bootstrap_ratio_tau_sweep"]["tau_0.20"]
        print(f"  {f}: mean={b['mean']}  CI=[{b['ci_low']}, {b['ci_high']}]  M4_pass={b['ci_high'] is not None and b['ci_high'] < 1.0}")
    except Exception as e:
        print(f"  {f}: ERROR {e}")
PY

echo ""
echo "Sync back with:"
echo "  rsync -av GPU_SERVER:~/Internal_bias/results/ ./results/"
