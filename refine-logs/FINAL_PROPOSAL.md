# Final Proposal — Latent Internal Bias (LIB)

> A continuous, layer-wise, hidden-state metric replacing the direct-answer internal-bias metric of Dang et al. (ICLR 2026, arXiv 2505.16448).

## Problem Anchor (frozen)

**Anchored problem.** The internal-bias metric of Dang et al. is a binary scalar computed by a second generation (direct-answer prompt). This has three consequences that their own paper flags as open:

1. It is not actually "internal" — it is the model re-generating, so it confounds generation stochasticity with bias.
2. It loses magnitude and distributional information (strength, confidence, direction).
3. It is uniform across problem populations where the optimal intervention should differ — specifically, it does not separate bias that happens to be correct (intuition) from bias that is a common-wrong attractor (prejudice).

As a result, the authors observe that **"the influence of internal bias persisted under all conditions"** of their mitigation experiments. Our anchored claim is that this is not a fundamental result but an artifact of measuring an aggregate of two oppositely-behaving populations.

## Thesis (one sentence)

A **tuned-lens projection of the post-question pre-reasoning hidden state** is a continuous, truly-internal replacement for the direct-answer internal-bias metric that (i) correlates more strongly with overthinking length, (ii) admits a **bias-emergence depth** signal orthogonal to bias strength, and (iii) cleanly separates *intuition-bias* from *prejudice-bias* populations that the original metric conflates.

## Dominant Contribution

A **single unified metric** (LIB) with three derived scalars — strength `σ`, alignment `μ`, depth `δ` — that simultaneously (a) reproduces the Dang et al. correlation at lower variance, (b) isolates the overthinking-causing subpopulation, and (c) enables a decomposition-conditional mitigation that the original paper could not derive.

## Method

### 1. Tuned-Lens Preparation
- Adapt the Belrose-et-al. tuned-lens (arXiv 2303.08112) per target model (R1-Distill-Qwen-7B/14B/32B, QwQ-32B).
- Training data: 50 MB of OpenWebMath + FineMath → ≤ 200 gradient steps on A100-40G (≈ 15 min per model).
- Fallback: raw logit-lens (no training) as an ablation.

### 2. Bias Extraction Pipeline
- Construct prompt `q` with the model's reasoning preamble through the `<think>` tag — **do not** include a direct-answer suffix.
- Run one forward pass; cache hidden states `h_ℓ(p*)` at layers `ℓ ∈ {⌊L/4⌋, ⌊L/2⌋, ⌊3L/4⌋, L-1, L}` and position `p*` = last token of preamble.
- Apply tuned-lens: `π_ℓ(a | q) ∝ softmax( Lens_ℓ h_ℓ(p*) ) | A(q)` where `A(q)` is the answer-vocabulary (§ 3).

### 3. Answer Vocabulary `A(q)`
Three cases:
- **Multiple-choice (Knowlogic)**: `A(q) = {A, B, C, D}` at letter-token level.
- **Integer open-answer (AIME / MATH500)**: Sample K = 16 direct-answer generations at T = 0.7; take the union of their first-digit tokens as candidates (max 10 integer + "9 of" hedge tokens).
- **Free-form numeric (CharCount)**: first two digit tokens after the ``\boxed{`` tag from a sampled direct-answer.

This canonicalization addresses reviewer concern #3 (multi-digit answers).

### 4. Derived Scalars
- **Strength** `σ(q) = max_a π_{L}(a | q) ∈ [0,1]`.
- **Alignment** `μ(q) = 1[argmax_a π_{L}(a | q) = final_CoT_answer(q)]`.
- **Depth** `δ(q) = min { ℓ / L : argmax π_ℓ = argmax π_L }`.
- Extra: **KL-to-uniform** `κ(q) = KL( π_L || Uniform(|A(q)|) )` as redundancy check.

### 5. Population Decomposition
With threshold `τ` tuned on a held-out 20 % split:
- Intuition-biased = `σ ≥ τ ∧ argmax π_L = correct_answer`
- Prejudice-biased = `σ ≥ τ ∧ argmax π_L ≠ correct_answer`
- Unbiased = `σ < τ`

### 6. Mitigation (Bonus)
**Prejudice-conditional truncation.** At reasoning-step boundary tokens (those preceded by `"Wait"`, `"Alternatively"`, `"Hmm"`), if the running logit-lens argmax has matched `argmax π_L` (the initial bias) for N consecutive boundaries AND the problem is *intuition-biased*, emit `</think>` and the initial-bias answer. For *prejudice-biased* problems, do NOT truncate — those are the cases that need the overthinking.

This inverts the usual "confidence → truncate" heuristic by conditioning on *bias type*.

## Pre-registered Predictions

| Prediction | Metric | Threshold for "pilot positive" |
|------------|--------|-------------------------------|
| P1 | `Spearman(σ, length) − Spearman(μ, length)` | ≥ +0.08 |
| P2 | `ΔR²(σ+δ) − R²(σ alone)` | ≥ +0.05 |
| P3 | `mean_length(prejudice) / mean_length(intuition)` | ≥ 1.8× |
| P4 | Truncation-preservation accuracy on intuition-biased set | ≥ 97 % of full-CoT accuracy at − 30 % tokens |

If P1 or P3 fails, fall back to Backup Idea (BDE per-step tracking, see IDEA_REPORT.md) before implementing the mitigation.

## Contribution Table

| Contribution | Replaces/extends | Status |
|--------------|------------------|--------|
| Continuous internal-bias metric via tuned-lens | Dang et al. binary direct-answer metric | Novel (gap check clean) |
| Bias-emergence depth `δ` | — | Novel |
| Intuition-vs-prejudice decomposition | — | Novel |
| Prejudice-conditional truncation | DiffAdapt, NYU probe (both un-decomposed) | Novel application of decomposition |

## Risks

- **R1 — Tuned-lens variance across answer types.** Mitigation: per-format `A(q)` canonicalization in § 3; ablate raw logit-lens.
- **R2 — Correlation improvement is driven by more-informative targets, not better-internal probe.** Mitigation: include "K-sample direct-answer peakedness" as a competitive baseline; the LIB metric must still win.
- **R3 — Decomposition thresholds leak dataset info.** Mitigation: all thresholds cross-validated; report both held-in and held-out numbers.
- **R4 — ICLR 2026 scoop risk.** The Dang paper is fresh; a follow-up on its metric is timely but competitive. Mitigation: prioritize the decomposition claim (which is harder to scoop than a lens-replacement).

## Deliverables

- `refine-logs/EXPERIMENT_PLAN.md` — runs, budget, order.
- `refine-logs/EXPERIMENT_TRACKER.md` — live run status.
- Pilot P0 artifact: `results/pilot_P0.json` with per-problem `{σ, μ, δ, length, correct, type}` and dataset-level correlations.

## Venue Target

- **Minimum viable**: ACL 2026 short paper or ICLR 2026 workshop on reasoning models.
- **Stretch**: NeurIPS 2026 main or ICLR 2027 full paper — requires multi-model + mitigation head-to-head.
