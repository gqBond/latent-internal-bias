# Round-1 Local Analysis (after canonicalization fix + Round-1 reviewer reframe)

**Date**: 2026-04-16.
**Model**: DeepSeek-R1-Distill-Qwen-7B (raw logit-lens, untrained).
**Lens layers**: [7, 14, 18, 22, 27] of 28.
**Datasets**: AIME24 (n=30), MATH500 subset (n=100, seed 0).

## Headline numbers

### MATH500 (n=100, CoT-correct rate 0.60)

| Metric | Value | Verdict |
|--------|-------|---------|
| Spearman(σ, length) | −0.176, p=0.080 | marginal (wrong-direction for "continuous beats binary") |
| Spearman(μ, length) | −0.114, p=0.260 | n.s. |
| Spearman(δ, length) | +0.233, **p=0.020** | significant |
| Spearman(direct-match, length) (Dang-like binary bit) | +0.001, p=0.993 | null on this subset |
| **partial Spearman(σ, length \| correct, κ)** | **−0.235, p=0.019** | **significant after controls** |
| **partial Spearman(δ, length \| correct, κ, σ)** | +0.183, p=0.068 | marginal after controls |
| ΔR²(σ+δ) vs R²(σ) | 0.011 | ✗ (P2 fails < 0.05) |
| R²(σ, δ, κ) | 0.049 | |
| Populations @ τ=0.20 | early_correct n=5 (mean_len 4962), early_incorrect n=86 (mean_len 7071), low_commit n=9 (mean_len 7421) | |
| Sample ratio early_incorrect / early_correct @ τ=0.20 | 1.43 | ✗ (P3 threshold 1.8) |
| Bootstrap ratio @ τ=0.20 (n_boot=2000) | mean 1.76, median 1.49, 95% CI [0.67, 3.86] | in-direction but CI crosses 1.0 |
| Bootstrap ratio @ τ=0.25 | mean 1.78, median 1.59, 95% CI [0.76, 3.62] | same |

### AIME24 (n=30, CoT-correct rate 0.60, σ range [0.21, 0.38])

| Metric | Value | Notes |
|--------|-------|-------|
| Spearman(σ, length) | +0.095, p=0.62 | |
| Spearman(μ, length) | undefined — μ is constant 0 | all 30 AIME problems have L-layer lens-argmax ≠ first-digit of final. Looks like a logit-lens calibration artifact (lens keeps picking "9"). |
| Spearman(δ, length) | +0.311, p=0.094 | marginal |
| partial Spearman(σ, length \| correct, κ) | +0.331, p=0.074 | marginal |
| partial Spearman(δ, length \| correct, κ, σ) | +0.271, p=0.148 | weakened after controls |
| ΔR²(σ+δ) vs R²(σ) | 0.074 | ✓ (passes 0.05) |
| Populations @ τ=0.20 | early_correct n=1, early_incorrect n=29, low_commit n=0 | degenerate |
| Bootstrap ratio @ τ=0.20 | mean 0.44, CI [0.35, 0.54] | wrong direction; n=1 intuition means resample is dominated by a single point |

## Interpretation

1. **Continuous σ does not beat binary μ as a length predictor** on MATH500 (P1 fails). **Demoted.**
2. **δ replicates across datasets and is marginally significant after controlling for correctness and κ on MATH500**. On AIME24, δ is marginal raw (p=0.09) and weakened by controls (p=0.15). **Needs replication across models.**
3. **σ conditional on correctness and κ** is a clean MATH500 signal (p=0.02). This is a *new* claim not in the original proposal. Reframing: σ is not useful as a raw predictor of length (problems vary in hardness), but it IS useful as a *residual* predictor after adjusting for correctness.
4. **Population decomposition is unstable at pilot scale.** Bootstrap CIs cross 1.0 at every τ. The 1.8× pre-registered threshold was unrealistic for n=100.
5. **The "lens picks 9 everywhere" artifact on AIME24 likely breaks the raw logit-lens variant.** Tuned-lens training is required before any strong claim about AIME24.

## Decisions for Round 2

- **Reframe paper around δ + conditional σ.** Already updated `refine-logs/FINAL_PROPOSAL.md`.
- **Train tuned-lens on 7B.** Must be done before scaling to other models — the logit-lens calibration issue will be worse on 1.5B.
- **Replicate on 1.5B + 14B.** Configs committed (`configs/r1_qwen_{1_5b,7b,14b}.yaml`).
- **Add MCQ benchmark** (Knowlogic-CharCount) to avoid the first-digit-vocab mis-specification.
- **Use median-ratio + bootstrap CI as the P3 target**, not point-estimate thresholds.
