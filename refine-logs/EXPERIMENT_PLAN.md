# Experiment Plan — LIB (Latent Internal Bias)

## Compute Budget

| Pool | GPU-hours | Purpose |
|------|-----------|---------|
| Pilot (P0-P1) | 4 A100-40G-h | Sanity + pre-registered predictions on one model / two datasets |
| Ablation (A1-A3) | 8 A100-40G-h | Lens variant, layer choice, `τ` sensitivity |
| Scale-up (S1-S3) | 24 A100-40G-h | Multi-model, multi-benchmark |
| Mitigation (M1-M2) | 12 A100-40G-h | Prejudice-conditional truncation vs baselines |
| **Total** | **48 GPU-h** | |

## Claim → Experiment mapping

| Claim | Block | Produces |
|-------|-------|----------|
| C1 (LIB better correlates with length than binary) | P0, S2 | Table 1 |
| C2 (depth `δ` adds orthogonal signal) | P1, A2 | Table 2 |
| C3 (intuition / prejudice populations differ) | P0, S2 | Figure 1 (histograms), Table 3 |
| C4 (prejudice-conditional truncation beats baselines) | M1, M2 | Table 4 (accuracy × tokens) |

## Run Order

### Block P0 — Pilot Core (≤ 1 GPU-h, run first)
- Model: `R1-Distill-Qwen-7B` (bf16, one A100-40G).
- Data: AIME-2024 (30 problems) + MATH500 random 100 (seed 0).
- Pipeline:
  1. Train tuned-lens: `scripts/train_tuned_lens.py --model R1-Distill-Qwen-7B --data openwebmath --steps 200 --bs 4 --lr 1e-3` (≈ 15 min).
  2. Full-CoT generation: reuse `AIME/aime2024.py` and build a `math500_cot.py` mirror.
  3. Direct-answer generation (baseline metric): reuse `AIME/direct2024.py` + mirror.
  4. LIB extraction: `scripts/extract_lib.py --model R1-Distill-Qwen-7B --layers 8,16,20,24,31 --pos last_pre_think`.
  5. Analysis: `scripts/analysis_lib.py --out results/pilot_P0.json`.
- Success gate: predictions P1 and P3 from FINAL_PROPOSAL.md both pass.

### Block P1 — Pilot Extensions (≤ 1 GPU-h)
- Same model; add Knowlogic-EN (50 MC problems) + CharCount-EN (50 problems).
- Verifies metric generalizes across answer-format types (multi-choice, integer, free-form numeric).
- Produces format-stratified correlations.

### Block A1 — Lens variant ablation (≤ 2 GPU-h)
- Raw logit-lens vs tuned-lens vs random-affine-lens (baseline).
- Predict: tuned-lens > logit-lens > random-affine on correlation with length, but logit-lens retains ≥ 80 % of tuned-lens improvement ⇒ LIB is usable without training.

### Block A2 — Layer and position ablation (≤ 3 GPU-h)
- Layers ∈ {4, 8, 12, 16, 20, 24, 28, 31}.
- Positions ∈ {last_of_q, first_after_think_tag, +3 tokens after think_tag}.
- Identify best `(ℓ*, p*)` for reporting; ablation table.

### Block A3 — Threshold / decomposition sensitivity (≤ 3 GPU-h)
- Sweep `τ ∈ [0.3, 0.9]` at step 0.05.
- Cross-validated on held-out 20 %.
- Produce robustness curves.

### Block S1 — Multi-model (≤ 8 GPU-h)
- Add `R1-Distill-Qwen-14B`, `R1-Distill-Qwen-32B`, `QwQ-32B`.
- Re-train tuned-lens per model (≈ 30 min each).
- Re-run P0 protocol.
- Check: is `δ` distribution consistent across scales? Do intuition/prejudice populations look similar?

### Block S2 — Multi-benchmark (≤ 8 GPU-h)
- Add GSM8K-hard (200), OlympiadBench-math (100), Knowlogic-ZH (50).
- Also include the exact subset used in Dang et al. for direct reproducibility comparison.

### Block S3 — DeepSeek-R1 full via API (≤ 8 GPU-h equivalent in API cost)
- Skipped if tuned-lens is unavailable for closed-weight models.
- Alternative: use DeepSeek-R1-Distill open variants only; note scale-limitation.

### Block M1 — Mitigation baseline comparison (≤ 6 GPU-h)
- Baselines: (a) DiffAdapt (arXiv 2510.19669) setup if released, else (b) NYU probe (2504.05419) replication, (c) fixed-length truncation at matched compute, (d) Dang et al. no-mitigation.
- Metric: accuracy vs generated tokens on AIME24 + MATH500.

### Block M2 — Prejudice-conditional mitigation (≤ 6 GPU-h)
- Implement prejudice-conditional truncation (see FINAL_PROPOSAL.md § 6).
- Evaluate P4 prediction (truncation preserves accuracy ≥ 97 % at −30 % tokens on intuition subset).
- Produce Table 4 (accuracy × tokens, stratified by bias type).

## Scripts To Implement

- [ ] `scripts/train_tuned_lens.py` — wrapper around `tuned_lens` package + HF weights.
- [ ] `scripts/extract_lib.py` — single forward pass per problem, caches `h_ℓ(p*)`, computes `π_ℓ, σ, μ, δ`.
- [ ] `scripts/analysis_lib.py` — correlations, population decomposition, plots.
- [ ] `scripts/mitigate_prejudice.py` — bias-conditional early-exit loop around HF `generate`.
- [ ] `scripts/eval_aime.py`, `scripts/eval_math500.py`, `scripts/eval_knowlogic.py` — accuracy + length + bias-type reporting.

## Milestones & Dates

| Milestone | Block | Target date (yyyy-mm-dd) |
|-----------|-------|-------------------------|
| Pilot positive (P1 + P3 pass) | P0 + P1 | 2026-04-22 |
| Ablations complete | A1 + A2 + A3 | 2026-04-29 |
| Multi-model, multi-benchmark | S1 + S2 | 2026-05-06 |
| Mitigation evaluated | M1 + M2 | 2026-05-13 |
| Draft complete | — | 2026-05-25 |
| ICLR 2027 submission ready (stretch) | — | 2026-09-25 |

## Kill Criteria

- **Pilot kill.** If after P0 + P1, Spearman(σ, length) − Spearman(μ, length) < 0, abandon LIB and pivot to Backup Idea (BDE per-step tracking, IDEA_REPORT.md § Idea 2).
- **Decomposition kill.** If prejudice/intuition length ratio < 1.2×, abandon the decomposition claim and reduce paper scope to "continuous metric only".
- **Mitigation kill.** If prejudice-conditional truncation does not beat fixed-length truncation at matched tokens, drop § 6 and keep the paper as an analysis contribution.
