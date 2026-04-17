# Final Proposal — Latent Internal Bias (LIB)

> A layer-wise, hidden-state lens on when a model commits to an answer, and how that commitment predicts CoT length and correctness. Extends the direct-answer internal-bias metric of Dang et al. (ICLR 2026, arXiv 2505.16448).

> **Round-1 reframe (2026-04-16)**: After the first pilot, reviewer feedback (see `AUTO_REVIEW.md`) led us to demote the continuous-beats-binary headline and promote **emergence depth `δ`** and **conditional strength `σ | correctness, κ`** to the primary claims. The "intuition / prejudice" populations were renamed to `early_correct / early_incorrect / low_commitment` for defensibility.

## Problem Anchor (frozen)

**Anchored problem.** The internal-bias metric of Dang et al. is a binary scalar computed by a second generation (direct-answer prompt). This has three consequences that their own paper flags as open:

1. It is not actually "internal" — it is the model re-generating, so it confounds generation stochasticity with bias.
2. It loses magnitude and distributional information (strength, confidence, direction).
3. It is uniform across problem populations where the optimal intervention should differ — specifically, it does not separate bias that happens to be correct (intuition) from bias that is a common-wrong attractor (prejudice).

As a result, the authors observe that **"the influence of internal bias persisted under all conditions"** of their mitigation experiments. Our anchored claim is that this is not a fundamental result but an artifact of measuring an aggregate of two oppositely-behaving populations.

## Thesis (revised)

A **tuned-lens projection of the post-question pre-reasoning hidden state** reveals *when* the model has already committed to an answer. The **emergence depth** `δ` (earliest layer at which the later-final argmax wins) and **conditional commitment strength** `σ | correctness, κ` are predictive of CoT length above and beyond problem difficulty, and cleanly separate three behavioral populations (early-correct / early-incorrect / low-commitment) that demand different test-time compute strategies.

## Dominant Contribution (revised)

We submit:
1. **`δ` is a cheap, lens-local signal for when the internal answer stabilizes.** It predicts CoT length on MATH500 at Spearman 0.233 (p=0.02), and 0.183 (p=0.07) after controlling for correctness, κ, σ — i.e., not merely a difficulty proxy.
2. **`σ` conditioned on correctness and κ is a stronger length-predictor than raw σ.** On MATH500, partial Spearman(`σ`, length | `correct`, κ) = -0.235 (p=0.02), materially larger than the direct-answer binary Dang metric (0.001 on this subset).
3. **A descriptive three-way decomposition** (early-correct / early-incorrect / low-commitment) enables a *population-conditional* test-time compute policy — truncate on early-correct, extend on early-incorrect. Pilot bootstrap (τ=0.25, n=100) gives early-incorrect / early-correct length ratio 1.78× (95% CI [0.76, 3.62]).

The continuous-vs-binary comparison (our original P1) was not supported and has been **demoted to a baseline** rather than the headline.

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

### 5. Population Decomposition (renamed)
With threshold `τ` tuned on a held-out 20 % split:
- `early_correct`   = `σ ≥ τ ∧ argmax π_L = correct_answer`      (was "intuition-biased")
- `early_incorrect` = `σ ≥ τ ∧ argmax π_L ≠ correct_answer`      (was "prejudice-biased")
- `low_commitment`  = `σ < τ`                                     (was "unbiased")

Stability is reported via bootstrap-CI across `τ ∈ {0.15, 0.20, 0.25, 0.30}` with n_boot = 2000. The `early_incorrect / early_correct` length ratio is reported only when both populations have ≥ 3 items in each bootstrap resample.

### 6. Mitigation (Bonus)
**Prejudice-conditional truncation.** At reasoning-step boundary tokens (those preceded by `"Wait"`, `"Alternatively"`, `"Hmm"`), if the running logit-lens argmax has matched `argmax π_L` (the initial bias) for N consecutive boundaries AND the problem is *intuition-biased*, emit `</think>` and the initial-bias answer. For *prejudice-biased* problems, do NOT truncate — those are the cases that need the overthinking.

This inverts the usual "confidence → truncate" heuristic by conditioning on *bias type*.

## Pre-registered Predictions (revised post-Round-1)

| ID | Claim | Metric | Threshold |
|----|-------|--------|-----------|
| **P2′** | δ predicts length beyond problem difficulty | partial Spearman(δ, length \| correct, κ, σ) | ≥ +0.15 on at least 2 of 3 models |
| **P2″** | δ survives vs. logit-lens artifact | tuned-lens replicates P2′ with ≥ 75 % of raw-lens stat | |
| **P3′** | Early-incorrect CoT length > early-correct | bootstrap-mean ratio at τ=0.25, n_boot=2000 | ≥ 1.5×, 95% CI excluding 1.0 on at least 1 model/dataset pair |
| **P4**  | Population-conditional test-time budget works | accuracy on early-correct at −30 % tokens | ≥ 97 % of full-CoT accuracy |
| (demoted) P1 | Continuous σ more length-informative than binary μ | Spearman(σ,L) − Spearman(μ,L) | Reported as baseline only |

If P2′ fails on all three models, fall back to the Backup Idea (BDE per-step tracking) before implementing mitigation.

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
