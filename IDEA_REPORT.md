# Idea Discovery Report

**Direction**: Improve the "internal bias" metric of Dang et al. 2025 ("The First Impression Problem", arXiv 2505.16448; repo: NJUNLP/LongCoT-Internal-Bias)
**Date**: 2026-04-16
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline
**Pilot status**: no GPU server configured in this environment → pilots are **planned, not executed**. Hand-off to `/run-experiment` in Stage 3 of `/research-pipeline`.

## Executive Summary
Dang et al. operationalize "internal bias" via a **direct-answer prompt** (no-CoT generation) and compare the result to the final CoT answer — a binary, re-generation-based metric that is not actually "internal" and conflates several phenomena. We recommend replacing it with **Latent Internal Bias (LIB)**: a layer-wise, continuous, hidden-state metric computed via logit/tuned-lens at the post-question pre-reasoning position. LIB unifies three improvements — continuous strength, emergence depth, and intuition-vs-prejudice decomposition — and is expected to give a stronger correlation with overthinking length while also supporting an online early-exit rule that the current metric cannot.

## Literature Landscape

### Neighborhood 1: Hidden-state probes for early exit / correctness
- **"Reasoning Models Know When They're Right"** (arXiv 2504.05419, NYU) — two-layer probe on chunk-last-token hidden state predicts correctness; −24 % tokens at 0.85 confidence threshold on R1-Distill-Qwen-32B.
- **"No Answer Needed"** (arXiv 2509.10625) — question-only linear probes on activations predict answer accuracy before any token is generated.
- **FlashThink** (arXiv 2505.13949) — early-exit based on hidden-state readout.

### Neighborhood 2: Overthinking mitigation by monitoring signals
- **DiffAdapt** (arXiv 2510.19669) — lightweight classifier on internal states gates difficulty-adaptive thinking.
- **Evolution of Thought** (arXiv 2508.17627) — overthinking tracked via reasoning dynamics.
- **ROM: Real-time Overthinking Mitigation** (arXiv 2603.22016) — detection head on late-layer hidden states + attention-pooled summary.
- **Cumulative Entropy Regulation** (arXiv 2510.02249) — token-entropy cumulative average guides "explore-briefly-then-decide".
- **Reasoning Completion Point Detector** — monitors rank dynamics of termination tokens, −44 % tokens.
- **Thought Calibration** (arXiv 2505.18404) — up to −60 % thinking tokens.

### Neighborhood 3: Confidence, calibration, and reasoning dynamics
- **"Reasoning Models Better Express Their Confidence"** (arXiv 2505.14489) — reasoning models strictly better calibrated in 33/36 settings.
- **Deep Think with Confidence** (DeepConf) — confidence-driven test-time scaling.
- **Trace-length as uncertainty signal** — reasoning-length itself correlates with uncertainty.

### Structural gaps around Dang et al.'s specific metric
- No work replaces direct-answer prompting with a **hidden-state-based, continuous** internal-bias metric using logit/tuned-lens at the post-question position.
- No work **decomposes bias** into the *intuition* case (bias == correct) vs. the *prejudice* case (bias == a plausible-but-wrong answer); the current paper treats both uniformly and calls the effect "persistent under all conditions" — which is what you'd expect if you are averaging two populations with opposite optimal interventions.
- No work frames the metric in terms of **bias-emergence depth** (earliest layer at which the answer token is top-1 in the logit lens).
- The existing probes (NYU, DiffAdapt, ROM) predict **correctness** or **difficulty** — none predicts **the identity of the model's first-impression answer token itself**, which is the quantity Dang et al. actually define.

## Candidate Ideas (12 generated, 3 surviving)

Ideas were filtered on: (a) feasibility ≤ 2 GPU-hours pilot; (b) novelty vs. above neighborhood; (c) directly attacks a limitation of the existing metric.

| # | Idea | Feasible | Novel | Kept |
|---|------|----------|-------|------|
| 1 | **LIB: layer-wise continuous metric via logit-lens** (★) | ✅ | ✅ | ★ Top |
| 2 | **Bias Decomposition: intuition vs prejudice** | ✅ | ✅ | Bundled into 1 |
| 3 | **Bias Emergence Depth** | ✅ | ✅ | Bundled into 1 |
| 4 | **Bias-Disagreement Event tracking (per-step)** | ✅ | ✅ | ★ Backup |
| 5 | **LIB-Guided adaptive early exit** (prescriptive application) | ✅ | Partially (overlaps NYU 2504.05419) | Follow-up paper |
| 6 | Ensemble direct-answer (K-sample) bias | ✅ | ❌ incremental | Dropped |
| 7 | Attention-entropy bias proxy | ✅ | ❌ overlaps ROM + head-entropy work | Dropped |
| 8 | Cross-lingual bias transfer | ✅ | ❌ incremental | Dropped |
| 9 | Bias-calibration fine-tuning | ❌ training heavy | ✅ | Future work |
| 10 | Bias-steering via activation patching | ⚠️ infra-heavy | ✅ | Future work |
| 11 | Semantic (tolerance) matching | ✅ | ❌ engineering | Absorbed as a preprocessing step |
| 12 | Cross-model bias conformity | ✅ | ❌ incremental | Dropped |

## Ranked Ideas (post-filter)

### 🏆 Idea 1: Latent Internal Bias (LIB) — Layer-wise Continuous Metric with Intuition/Prejudice Decomposition — RECOMMENDED

**Hypothesis.** Applying tuned-lens at the post-question, pre-reasoning hidden-state position yields a continuous internal-bias score that (a) correlates with overthinking length more strongly than the binary direct-answer metric, (b) separates an *intuition* population (bias == correct; short CoT; no overthink) from a *prejudice* population (bias == common-wrong; long CoT; overthink) that respond to opposite mitigations, and (c) admits a per-problem **bias-emergence depth** that predicts overthinking severity orthogonally to bias strength.

**Formal definition.**

Let `q` be the question prompt and `p*` the first post-question token position (e.g., just after the `<think>` tag). For a model with `L` layers and hidden states `h_ℓ(p*)`, let `Lens_ℓ : ℝ^d → ℝ^V` be the tuned-lens (or logit-lens baseline) decoder.

Define the per-layer answer distribution:
```
π_ℓ(a | q) ∝ softmax( Lens_ℓ( h_ℓ(p*) ) ) restricted to the answer-vocabulary A(q)
```
where `A(q)` is either the multiple-choice set or, for open answers, the top-K first-answer-tokens across K greedy direct-answer generations.

Three derived scalar metrics per problem:
1. **Bias strength** `σ(q) = max_a π_L(a | q)` — continuous, in `[0,1]`.
2. **Bias-final alignment** `μ(q) = 1[argmax π_L == final_answer]` (matches Dang et al.'s binary, kept as a baseline to compare against).
3. **Emergence depth** `δ(q) = min { ℓ / L : argmax π_ℓ == argmax π_L }` — in `[0,1]`.

Decomposition at the dataset level: split the problem population into
- **Intuition-biased**: `σ ≥ τ` and `argmax π_L == correct_answer`.
- **Prejudice-biased**: `σ ≥ τ` and `argmax π_L ≠ correct_answer`.
- **Unbiased**: `σ < τ`.

**Claim.** The *prejudice-biased* subset carries almost all the overthinking signal; the *intuition-biased* subset is a short-CoT population that existing mitigations should not touch.

**Pilot (planned, feasible in < 1 GPU-hour)**
- Model: R1-Distill-Qwen-7B (16-bit, one A100-40G).
- Data: AIME24 (30 problems) + MATH500 random 100.
- Implementation:
  - Train tuned-lens for R1-Distill-Qwen-7B on a small corpus of OpenWebMath (~200 iters, ≈ 15 min once per model).
  - Run full-CoT generation (reuse `AIME/aime2024.py` from the repo).
  - Run direct-answer generation (reuse `AIME/direct2024.py`) — **baseline metric**.
  - Extract hidden states at `p*` from one extra forward pass through `q` only.
  - Compute `π_ℓ`, `σ`, `μ`, `δ` at ℓ ∈ {8, 16, 20, 24, L}.
- Analysis: Pearson + Spearman correlation of `σ`, `μ`, `δ`, `(σ × δ)` with reasoning length (and with length-after-first-answer). Regression R² when combining multiple metrics. Stratified length distribution across intuition / prejudice / unbiased buckets.

**Pilot prediction (pre-registered)**
- `Spearman(σ, length) > Spearman(μ, length)` by ≥ 0.08 absolute.
- `R²(σ + δ, length) > R²(σ alone, length)` by ≥ 0.05.
- `mean_length(prejudice) / mean_length(intuition) ≥ 1.8×`.

**Novelty differentiation.**
- vs. NYU probe (2504.05419): that probe predicts **correctness** from chunk-end hidden state. LIB predicts the **answer token itself** from post-question hidden state and ties it to the specific *internal-bias* construct.
- vs. "No Answer Needed" (2509.10625): that paper predicts accuracy **yes/no** from question-only activations. LIB predicts the **answer distribution**, decomposes bias type, and explains overthinking.
- vs. ROM (2603.22016) / DiffAdapt (2510.19669): those predict *overthinking score* or *difficulty*. LIB predicts the first-impression answer itself and derives overthinking downstream.

**Reviewer-anticipated weaknesses (see § Critical Review).**

---

### Idea 2: Bias-Disagreement Event (BDE) Tracking — BACKUP

**Hypothesis.** Overthinking is not uniform; it is concentrated in **disagreement events** where the model's ongoing intermediate answer (logit-lens on the latest generated token) disagrees with the initial bias (`π_0 = LIB at p*`). The count and timing of these events predicts overthinking better than total length.

**Metric.**
```
BDE(trace) = | { t : argmax π_0 ≠ argmax logit_lens_L( h_L(t) ) } over reasoning tokens t |
```

**Why it matters.** Gives a per-trace (not per-problem) decomposition of overthinking that can be used to localize and surgically truncate reasoning.

**Pilot.** Same infra as Idea 1; adds streaming hidden-state extraction during generation; ≈ 2 GPU-hours.

---

### Idea 3 (follow-up paper): LIB-Guided Adaptive Early Exit

**Hypothesis.** Using LIB online at each reasoning step and truncating when `σ` is high AND the current intermediate answer matches `argmax π_0` achieves stronger compute savings than chunk-level correctness probes (NYU 2504.05419), because it directly targets the *bias-reasoning agreement* state that characterizes low-value reasoning.

**Status.** Parked as Idea 1's natural follow-up; to be pursued only after Idea 1 shows positive signal on the metric.

## Eliminated Ideas

| Idea | Reason |
|------|--------|
| K-sample direct-answer ensemble | Incremental; still uses generation, doesn't address the "not truly internal" critique. |
| Attention-entropy bias proxy | Overlaps ROM (attention-pooled state) and Head-Entropy prior work. |
| Cross-lingual bias transfer | Incremental; unclear what claim it supports. |
| Bias-calibration fine-tuning | Training-heavy; orthogonal to metric improvement. |
| Activation-patching bias steering | Infra-heavy (nnsight / TransformerLens); better pursued as a follow-up. |
| Cross-model bias conformity | Descriptive; no actionable claim. |

## Deep Novelty Check (Phase 3, completed on desk)

Queries across arXiv + Semantic Scholar + Google Scholar (2024-06 → 2026-04):
- `"internal bias" "overthinking"` → only Dang et al. (2505.16448) and Internal-Bias Moonlight review.
- `logit lens "internal bias"` → zero hits with both terms.
- `"first impression" reasoning LLM` → Dang et al. only.
- `logit lens "answer token" probe "before reasoning"` → zero specific hits; closest is "No Answer Needed" (accuracy, not answer identity).
- `"bias emergence" layer reasoning` — zero hits.
- `intuition vs prejudice reasoning LLM` — zero hits in this context.

**Verdict for Idea 1:** **NOVEL**. Closest existing work (NYU 2504.05419 and 2509.10625) uses hidden-state probes but for *correctness*, not for the *answer-identity bias* construct; and does not decompose intuition vs prejudice. The ICLR 2026 Dang paper is the exact grounding, and no follow-up addressing its metric has yet appeared (per arXiv listing as of 2026-04-16).

## Critical Review (Phase 4, self-simulated senior-reviewer)

**Score: 6.5 / 10.** Credible paper-sized contribution; directly attacks the acknowledged gap of a recent ICLR paper; pilot is feasible in < 1 GPU-hour.

**Strengths.**
- Clear, direct gap: the original authors call their own metric limiting ("all mitigations failed").
- Pilot is low-risk, pre-registered predictions are falsifiable.
- Bundles three sub-contributions (continuous / decomposition / depth) into one coherent story.

**Weaknesses & required fixes (for the claim score to reach ≥ 8).**
1. *Small benchmark*. Extend beyond AIME/MATH500 to GSM8K-hard, Knowlogic (their own benchmark), OlympiadBench and HumanEval-Reasoning. Else the correlation improvement may be noise.
2. *Tuned-lens training overhead*. Need to show logit-lens (no training) gives most of the signal so the metric stays cheap. Ablation required.
3. *Define answer-vocabulary A(q) carefully for open answers*. Using "first answer token" is brittle for multi-digit integers like 143. Need a tokenizer-aware canonicalization (e.g., project to the first digit, or use the final-answer-tag region).
4. *Clear comparison table with the three closest works*. Must show a head-to-head column for correlation-with-length, not just prose claims.
5. *No mitigation evaluation*. The paper will be stronger if LIB is shown to enable a simple mitigation (decomposition-conditional truncation) that beats non-decomposed baselines on prejudice-biased problems.
6. *Same-family generalization*. Show the metric transfers across R1-Distill-Qwen-{7B,14B,32B} and QwQ-32B with the lens re-trained.

**Minimum viable paper** (4-page ACL short / ICLR workshop): pilot on AIME + MATH500, R1-Distill-Qwen-7B only, metrics `σ, μ, δ`, one mitigation (truncate on prejudice-match events).

**Full paper** (ICLR/NeurIPS): multi-benchmark, multi-model, tuned-lens vs logit-lens ablation, mitigation head-to-head vs DiffAdapt / NYU probe / direct-answer baseline.

## Refined Proposal & Experiment Plan
- `refine-logs/FINAL_PROPOSAL.md` — thesis, method, contribution table, risks.
- `refine-logs/EXPERIMENT_PLAN.md` — pilot and full-paper experiment roadmap with GPU/time budgets.
- `refine-logs/EXPERIMENT_TRACKER.md` — run-by-run tracker (empty until Stage 3).

## Next Steps
- [ ] `/run-experiment` — execute Pilot Block P0 (LIB core metric on AIME24 + R1-Distill-Qwen-7B).
- [ ] `/auto-review-loop` — iterate until reviewer score ≥ 8.
- [ ] Promote LIB-Guided Adaptive Early Exit (Idea 3) to its own follow-up paper only after Idea 1 pilot is positive.
