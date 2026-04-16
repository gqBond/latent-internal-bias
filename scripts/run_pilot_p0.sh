#!/usr/bin/env bash
# Pilot block P0 — LIB core on R1-Distill-Qwen-7B, AIME24 + MATH500(100).
# Runs the full flow end to end. Stops on first error.
set -euo pipefail

CFG="${CFG:-configs/r1_qwen_7b.yaml}"
MODEL_NAME="$(python -c "from lib.config import load_cfg; print(load_cfg('${CFG}').model.name)")"
LENS="results/lenses/${MODEL_NAME}/tuned_lens.pt"

echo "[P0] tuned-lens training"
python -m scripts.train_tuned_lens --cfg "${CFG}"

for dataset in aime math500; do
    case "$dataset" in
        aime)
            python -m scripts.eval_aime --cfg "${CFG}" --year 2024
            STEM="aime2024"
            PROBLEMS="data/aime/aime2024.jsonl"
            ;;
        math500)
            python -m scripts.eval_math500 --cfg "${CFG}" --n 100 --seed 0
            STEM="math500_n100_s0"
            PROBLEMS="data/math500/math500.jsonl"
            ;;
    esac

    COT="results/cot/${MODEL_NAME}/${STEM}_cot.jsonl"
    DIRECT="results/direct/${MODEL_NAME}/${STEM}_direct.jsonl"
    LIB="results/lib/${MODEL_NAME}/${STEM}_lib.jsonl"
    SUMMARY="results/lib/${MODEL_NAME}/${STEM}_summary.json"

    echo "[P0] extract LIB for ${STEM}"
    python -m scripts.extract_lib \
        --cfg "${CFG}" \
        --problems "${PROBLEMS}" \
        --cot-out "${COT}" \
        --direct-out "${DIRECT}" \
        --lens-path "${LENS}" \
        --out "${LIB}"

    echo "[P0] analyze ${STEM}"
    python -m scripts.analysis_lib \
        --lib "${LIB}" \
        --cfg "${CFG}" \
        --out "${SUMMARY}"
done

echo "[P0] done. See results/lib/${MODEL_NAME}/*_summary.json"
