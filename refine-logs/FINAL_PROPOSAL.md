# Final Proposal — LIB Measurement Methodology

> A measurement-methodology paper on latent-internal-bias probes in reasoning models. Shows that single-token lens readouts create answer-prior artifacts, calibration alone does not remove them, and target-matched full-answer readouts change the claimed link between latent answers and reasoning length.

> **Round-3 reframe (2026-04-21)**: After Round-3 reviewer (score 5/10, "not ready" but pivot-endorsed), the core contribution is the full-answer lens readout + calibration sanity protocol. The pre-registered σ/δ/P1/P2/P3 claims are de-promoted to *diagnostic* signals showing how much of prior work's conclusions depends on the readout choice.
>
> **Round-1 reframe** and **Round-2 reframe** history is preserved in `AUTO_REVIEW.md` for provenance.

## Problem Anchor (revised)

Dang et al. (ICLR 2026, arXiv 2505.16448) measure *internal bias* via a second direct-answer generation. Concurrent follow-ups, including an initial version of this proposal, attempted to replace that binary with a continuous tuned-lens scalar σ computed at a pre-reasoning position. We find that every variant of *single-token* lens readout on open-numeric math reasoning produces a severe answer-prior artifact: a single digit dominates the argmax across 67–97% of problems regardless of content. Null-prompt calibration (a cheap per-layer baseline subtraction) shifts which digit dominates but does not remove the dominance. Correlations with CoT length that survive first-digit scoring — including our Round-2 conditional-σ signal — do not survive full-answer scoring. The scientific claim is therefore entangled with the measurement choice.

## Thesis (new)

1. **Lens readouts on reasoning models must be target-matched.** A first-token lens over a digit vocabulary is not a measurement of latent answer; it is a measurement of the lens's digit prior convolved with the task. The dominant-label fraction is a sufficient sanity test: any single label > 0.35 on a benchmark whose ground-truth distribution is not that peaked is a red flag.
2. **Full-answer teacher-forced scoring reduces the artifact.** Scoring candidate strings via summed per-position lens log-softmax restores an entropy and a ground-truth-dominance profile that matches the task, not the lens.
3. **Post-hoc corollary (MATH500, 7B only, exploratory):** under full-answer scoring, problems on which the model has early latent commitment to the *correct* answer have ~1.7× longer CoT than problems with early commitment to a *wrong* answer. This inverts the naive "commitment → long CoT" reading of Dang-style metrics and is consistent with a verification-over-commitment account of chain-of-thought.

## Core contribution

1. **Calibration sanity protocol** for lens-based internal-bias probes:
   - report argmax-frequency distribution and entropy per layer;
   - report dominant-label fraction at the reporting layer;
   - compare against the benchmark's ground-truth label dominance;
   - fail-closed on implausible peaks (e.g., `dominant_frac > 0.35` when GT is flat).
2. **Full-answer lens scoring** (`--full-answer` in `scripts/extract_lib.py`): for each candidate answer string c ∈ 𝒞(q), teacher-force `prompt ⊕ c` through the model and compute `log P_lens(c | prompt) = Σ_i log softmax(lens_ℓ(h_{p*+i}))[t_i]`, then softmax across candidates → π_per_layer. Candidate set 𝒞(q) = direct-answer samples ∪ {CoT answer, ground truth}.
3. **Three-readout ablation** on the same model/data: first-digit vs null-calibrated first-digit vs full-answer. Quantifies how much of each downstream metric is an artifact of the readout. Table-ready in the paper.
4. **Exploratory secondary finding** (kept as exploratory, not a headline): the MATH500-7B full-answer P3 inversion, framed as a verification-consistent alternative interpretation.

## Method

### 1. Lens preparation
Belrose-et-al. tuned-lens (arXiv 2303.08112), per target model. 200–400 steps on OpenWebMath+FineMath ≈ 15–25 min per model on A100-40G. `lib/lens.py` hard-fails on load if `min_ℓ ||W_ℓ − I||_F < 0.1` to prevent a silent fallback to logit-lens.

### 2. Readouts compared
For the same hidden states and same tuned lens, compute three readouts per problem:
- **First-digit**: restrict lens logits to digit tokens {0,…,9}, softmax, π_per_layer.
- **Null-calibrated first-digit**: subtract per-layer lens logits of a neutral prompt ("Provide a number for this {format} answer.") from each problem's lens logits before restricting.
- **Full-answer**: teacher-force each candidate c ∈ 𝒞(q) as the suffix, sum per-position log-softmax of lens outputs at target tokens, softmax across candidates.

### 3. Calibration sanity output
For each layer ℓ and readout, record: top-3 argmax labels with frequencies, argmax entropy (nats), dominant-label fraction. For the reporting layer, also record ground-truth answer dominance and σ mean/std. Emitted in every summary JSON under `calibration_sanity`.

### 4. Scalars (unchanged surface, now diagnostic)
- σ(q) = max probability at last lens layer.
- μ(q), μ_correct(q) = label-level alignment.
- δ(q) = earliest layer where argmax stabilizes, normalized.
- κ(q) = KL(π_L || Uniform).

These are not the headline claims. They are reported as *probes* whose behavior across readouts documents the artifact.

### 5. P3 population decomposition (with guards)
`early_correct = σ ≥ τ ∧ argmax matches correct`, `early_incorrect = σ ≥ τ ∧ ¬match`, `low_commitment = σ < τ`. Cell-size guard: ratio reported only when both `early_correct` and `early_incorrect` have n ≥ 20 (smaller cells return null with `reason=insufficient_cell`). Bootstrap CI (n_boot = 2000) across τ ∈ {0.15, 0.20, 0.25, 0.30}.

### 6. Cross-readout comparison (the paper's main table)
For every (model, benchmark) the paper reports a three-column table of σ/δ/P1/P2/P3 under each readout plus the calibration sanity row. The table is the contribution: it quantifies how much of the literature's continuous-internal-bias claims survive a target-matched readout.

## De-promoted (Round-1/2 headline, now diagnostic)

- P1 (continuous σ beats binary μ) is not a stable phenomenon across readouts and is reported only as a row in the three-readout table.
- P2 (ΔR² for δ) passes on n=30 AIME only, fails MATH500 n=100. Kept as a diagnostic row.
- Pre-reg P3 in the Round-2 form (early-incorrect > early-correct length) is violated in the wrong direction on MATH500 full-answer with bootstrap CI [0.38, 0.98] excluding 1. The inversion is the exploratory finding, not P3.

## Pre-registered generalization checks (Round 4)

| ID | Claim | Metric | Threshold |
|----|-------|--------|-----------|
| **M1** | Full-answer reduces dominant-label fraction | `calibration_sanity.final_layer.dominant_label_frac` under full-answer < 0.35 | on at least 2 of 3 model sizes, on both AIME and MATH500 |
| **M2** | Full-answer raises GT-answer dominance | `correct_answer_dominant_frac` under full-answer ≥ 0.45 (AIME) / ≥ 0.25 (MATH500) | on at least 2 of 3 model sizes |
| **M3** | Readout choice flips ≥ one pre-reg outcome | At least one of P1/P2/P3 changes pass/fail state between first-digit and full-answer | on both AIME and MATH500 (already satisfied at 7B) |
| **M4-explo** | MATH500 full-answer P3 inversion replicates across scale | bootstrap-mean ratio < 1.0 with CI excluding 1.0 | on at least 2 of 3 model sizes, MATH500 only |

M1–M3 establish the methodology claim. M4 is exploratory and labeled as such; a single positive replication is enough to keep the inversion as a claim, but it is not required for paper acceptance.

## Contribution Table

| Contribution | Replaces/extends | Status |
|--------------|------------------|--------|
| Full-answer teacher-forced lens readout | first-token lens-vocab restriction | Implemented (round 2/3) |
| Calibration sanity protocol | implicit assumption of lens fidelity | Implemented, table-ready |
| Three-readout ablation table | single-readout internal-bias claims | 7B done; scale replication in Round 4 |
| P3 inversion on MATH500 full-answer | direction in Dang-style commitment story | Exploratory, needs scale replication |

## Risks (revised)

- **R1 — Cross-scale replication may fail M1/M2.** Mitigation: if full-answer does not reduce dominant-label frac on 1.5B or 14B, that is itself a finding ("the artifact is model-specific"); the paper still ships with honest reporting.
- **R2 — Full-answer candidate set biases inference.** Dedup + cap at 64 candidates; candidate-set size reported alongside every result; ablate by restricting to {CoT answer, ground truth}.
- **R3 — Calibration sanity threshold (0.35) is author-chosen.** Reported alongside the actual number and the benchmark's ground-truth label distribution for reader judgment.
- **R4 — Verification-vs-commitment (M4) is a single-dataset story.** We flag it as exploratory throughout. No title/abstract claim. One line in the discussion with a pointer to intervention as follow-up.

## Deliverables (Round 4)

- `scripts/round4_launch.sh`: cross-scale full-answer runs on 1.5B + 14B with sanity-gate.
- Results JSONs: `results/lib/{R1-Distill-Qwen-1.5B,R1-Distill-Qwen-14B}/{aime2024,math500}_fullans_summary.json`.
- `scripts/readout_comparison.py`: emits the three-readout table from existing summaries for the paper.
- Updated `AUTO_REVIEW.md` round 4 entry (target: score ≥ 6).

## Venue Target (revised)

- **Primary**: NeurIPS 2026 workshop on mechanistic interpretability, or ICLR 2027 blog post (both receptive to measurement-correction papers).
- **Stretch**: ACL 2026 short paper framed as "How to evaluate internal-bias probes in reasoning models", if M1–M3 replicate at scale.
- **Aspiration**: full NeurIPS 2026 main paper — requires M4-exploratory to replicate on at least one additional model and an intervention confirming the verification account.
