# Auto Review — Latent Internal Bias (LIB)

Autonomous review loop. External reviewer: Codex (`gpt-5.4`, reasoning `high`).
Thread: `019d9ca9-b384-7501-88c9-dc254b3c17eb`.

## Round 1 (2026-04-16)

### Context sent to reviewer

- Pilot done on DeepSeek-R1-Distill-Qwen-7B (28 layers, YaRN RoPE → 64k), raw logit-lens (no tuned-lens yet).
- AIME24 (n=30) and MATH500 subset (n=100, seed 0).
- Per-problem LIB scalars (σ, μ, δ, κ) computed at layers [7,14,18,22,27].
- Pre-bug: μ / μ_correct were constant 0 because `bias_argmax` was a single digit but `cot_answer` was a multi-digit string — fixed with `canonicalize_to_vocab` (commit `b188cee`).

### Assessment (Summary)

- **Score: 3/10**
- **Verdict: not ready**
- Key criticisms:
  - The headline claim ("continuous σ beats binary μ") FAILS its own pre-registered tests. P1 fails on both sets; P2 passes only on AIME24 (n=30); P3 misses threshold on MATH500 and is wrong-direction on AIME24.
  - Target is mis-specified for open-numeric: a 10-way *first-digit* vocab measures "latent belief over first digits", not over answers.
  - Raw logit-lens looks broken (AIME: L-layer argmax = "9" for many unrelated problems) → likely calibration artifact, not real signal.
  - Scope too small: one distilled 7B model, small n, intuition-population n=1 in one case.
  - The strongest signal (δ, emergence depth) is confounded with problem difficulty / correctness / uncertainty.
  - Framing "prejudice / intuition" oversells the evidence.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

- Score: 3/10
- Verdict: not ready

- Weaknesses (ranked)
1. The main paper claim fails on its own preregistered tests. P1 fails on both datasets, P2 passes only on a tiny AIME slice, and P3 is either below threshold or uninterpretable. A top-tier reviewer will read this as: the headline hypothesis did not survive contact with data.
2. The measurement target is mis-specified for open numeric tasks. With a 10-way digit vocabulary and first-character canonicalization, you are not measuring latent belief over answers; you are measuring latent belief over first digits. That is a much weaker and noisier object. Reviewers will call this out immediately.
3. The raw logit-lens looks broken enough to invalidate conclusions. The AIME "many unrelated problems map to 9" pathology and `mu = 0` everywhere strongly suggest calibration failure, not a property of the model's latent state.
4. The empirical scope is too small and too narrow. One distilled model, `n=30` AIME, `n=100` MATH subset, and one result with `intuition n=1` is not serious evidence for a new metric paper at NeurIPS/ICML/ICLR.
5. The strongest surviving signal, `delta`, is confounded with difficulty/correctness/uncertainty. Harder problems naturally produce longer CoT and later answer concentration. Right now "emergence depth predicts length" is plausible, but "internal bias explains reasoning" is not established.
6. The thresholded population story is unstable. With `tau=0.2`, your key decomposition produces tiny or degenerate groups. P3 is therefore not a real result.
7. The novelty claim is weak in its current form. "Read pre-reasoning hidden state with a lens" is an instrumentation tweak, not yet a paper. The only potentially publishable idea is the emergence-depth finding, but that needs a tighter conceptual story and stronger replication.
8. The framing is over-ambitious and somewhat anthropomorphic. "Intuition" and especially "prejudice" are much stronger words than your evidence supports. Reviewers will see this as overselling.

- Minimum fixes per weakness
1. For the failed headline claim: stop claiming "continuous beats binary" unless new experiments actually show it. Minimum acceptable fix is an apples-to-apples comparison against the Dang regeneration metric on the same items and at least 2 models, with one robust win for your method beyond noise.
2. For the target mismatch: move at least part of the evaluation to tasks with a natural discrete answer space, such as MCQ. If you stay on AIME/MATH, use full normalized answer candidates rather than first digits. If that is infeasible, the open-answer version should be framed as exploratory only.
3. For the lens artifact: train a tuned lens or at minimum a calibrated affine lens, then show the dominant-digit collapse disappears. If the artifact remains, the current metric is dead.
4. For scope: spend the remaining budget on replication, not more clever metrics. Minimum: 2 more models from your ready configs, and at least one additional task family with natural labels. I would prioritize `R1-Distill-Qwen-1.5B`, `7B`, `14B`, plus one small MCQ benchmark.
5. For confounding: run partial correlations or regressions controlling for correctness, dataset, and uncertainty (`kappa` or entropy). Also stratify by correct vs incorrect CoT. If `delta` only tracks hardness, reviewers will notice.
6. For the unstable decomposition: drop P3 unless you can show threshold sweeps with bootstrap CIs and nontrivial group sizes. Continuous analyses are safer.
7. For novelty: add one stronger result tied to `delta`, not `sigma`. Minimum viable version: show `delta` replicates across models/tasks and is the best predictor of CoT length after controls. Better: show it predicts recoverability from an initially wrong latent answer.
8. For framing: rename the populations to something descriptive and defensible, e.g. `early-correct commitment`, `early-incorrect commitment`, `low commitment`. This is a small edit but an important one.

- Reframing suggestion (if warranted)
Yes. Reframe away from "continuous strength beats binary." That claim is not supported. The salvageable paper is about answer-emergence depth.

The paper I would write is: models often form a provisional answer before overt reasoning, and the layer/depth at which that answer stabilizes is more informative than raw confidence. Then show:
- `delta` replicates as a predictor of CoT length across models and datasets.
- `delta` survives controls for correctness and uncertainty.
- Early incorrect commitment and late commitment have distinct behavioral profiles.
- `sigma` and the Dang-style binary metric are secondary baselines, not the headline.

If you can add one mechanistic or practical angle, the paper gets much better:
- `delta` predicts when longer reasoning will help.
- `delta` predicts whether the model will revise an early wrong answer.
- `delta` can gate adaptive test-time compute.

Right now, this is best described as a negative result on the original thesis plus one promising subplot. That is not submission-ready. It becomes "almost" if you do three things: fix/calibrate the lens, replicate `delta` across multiple models/tasks, and rewrite the paper around emergence depth rather than strength.

</details>

### Actions Taken (planned for Round 2)

Local-only (can run now):
- [ ] Partial-correlation analysis: Spearman(δ, length | correctness, κ) — controls for difficulty & uncertainty.
- [ ] Bootstrap CIs for P3 ratio across τ ∈ {0.15, 0.20, 0.25, 0.30}.
- [ ] Rename populations to `early-correct / early-incorrect / low-commit` in analysis output (keeps code backward-compatible via aliases).
- [ ] Drop P1 as primary; demote σ to baseline. Reframe the proposal doc around δ.

GPU runs (launch on server, ~10 GPU-hr total):
- [ ] Re-run 7B pilot with tuned-lens (instead of raw logit-lens) — fixes "all argmax → 9" artifact.
- [ ] Replicate on R1-Distill-Qwen-1.5B and 14B (configs already committed).
- [ ] Add MCQ benchmark (Knowlogic or CommonsenseQA subset) so we have a natural-label task.

Deprioritized (save for Round 3 if budget allows):
- Mitigation experiment (P4) — only after δ story is solidified.

### Results

- Pending Round 2 experiments. Nothing new to report yet beyond the corrected summaries under `results/lib/DeepSeek-R1-Distill-Qwen-7B/`.

### Status

- Continuing to Round 2 after local reframing + GPU replication runs.

---

## Round 2 (2026-04-20)

Thread: `019dac49-405f-7de0-bffd-6ed175dbcc4c` (original Round 1 thread `019d9ca9-...` expired; started fresh with full Round 1 recap in the prompt).

### Context sent to reviewer

- Tuned-lens actually loaded this round (Round 1 silently fell back to logit-lens on corrupt `.pt`). `lib/lens.py` now raises `RuntimeError` if `min ||W - I||_F < 0.1` across configured layers — no silent fallback.
- `scripts/extract_lib.py` gained `--null-prompt` flag that caches per-layer lens logits on a neutral prompt and subtracts them before restricting to `A(q)`. Code path committed but NOT YET run on the 7B re-extraction — round 2 numbers are without null-prompt calibration.
- `scripts/analysis_lib.py`: partial-Spearman (rank-residual), bootstrap-ratio τ-sweep (n_boot=2000), population rename (early_correct / early_incorrect / low_commitment), Spearman(δ, length) reported alongside P1/P2/P3.
- Re-ran 7B pipeline on AIME2024 (n=30) and MATH500 subset (n=100, seed 0). New results at `results/lib/DeepSeek-R1-Distill-Qwen-7B/{aime2024,math500_n100_s0,aime2024_tunedlens,math500_tunedlens}_*.json`.

### Assessment (Summary)

- **Score: 4/10** (up from 3/10)
- **Verdict: not ready**
- Key criticisms (ranked):
  1. Lens readout still dominated by digit prior. `bias_argmax` = "9" on ≈70% of MATH500 and 29/30 AIME — means μ / populations / δ are all measured on a distorted readout.
  2. First-digit target is too weak for open-numeric math. Full-answer scoring required.
  3. Pre-registered P1 still fails on both sets, P2 passes only on n=30 AIME, P3 bootstrap CI includes 1 on MATH500 and is degenerate on AIME (`early_correct n=1`).
  4. Best result — **partial Spearman(σ, length | correct, κ) = -0.235, p=0.019 on MATH500** — is post-hoc, not yet replicated.
  5. δ layer coverage too coarse (only 5 layers out of 28).
  6. P3 populations unstable; need minimum cell sizes or quantile groups.
  7. Only one model. Reviewer: *do not* spend budget scaling a broken readout.

### Reviewer Raw Response

<details>
<summary>Click to expand full Round 2 reviewer response</summary>

**Score: 4/10. Verdict: not ready.**

This is better than Round 1 because the tuned-lens loading is now real, the analysis is more serious, and the writeup is more honest. But the central measurement problem is still unresolved. The current data do not yet support an ICLR/NeurIPS-level metric paper. Most pre-registered tests fail, the strongest result is post-hoc, and the answer-space artifact remains large enough that the main measurements are not clean.

The short version: the project is not dead, but the current results are not usable as main evidence.

**Is `argmax = 9` fatal?**

For the present first-digit pipeline, yes, mostly. It is fatal for any strong claim about `μ`, early-correct versus early-incorrect populations, and `δ` as "emergence of the model answer," because the argmax is still dominated by a label prior rather than clear problem-specific answer evidence. If `bias_argmax` is `"9"` on 70% of MATH500 and 29/30 AIME, then early commitment is not yet measuring commitment to the problem answer. It is measuring commitment inside a distorted digit-only readout.

For `σ`, the flaw is serious but not fully fatal. Since `σ` is max probability, it can still carry information about confidence or sharpness even if the winning label is often wrong. But without showing that `σ` is problem-conditioned after removing the digit prior, the MATH500 conditional result could be "confidence in the digit prior predicts shorter length," not "latent internal bias predicts shorter length."

So the conditional-`σ` finding is not publishable on its own yet. With scale replication alone on the current pipeline, I would still not trust it. With null-prompt calibration, full-answer scoring or MCQ validation, and held-out replication, it could become a real secondary finding.

**Ranked Weaknesses And Minimum Fixes**

1. The answer readout is still broken by the digit prior.
Minimum fix: show that after calibration, no single digit dominates the argmax distribution across problems. Report argmax entropy, top-label frequencies, first-digit accuracy, and problem-vs-null logit lift. A rough sanity target: `"9"` should not be near 70% unless the dataset answer distribution itself supports that.

2. The first-digit target is too weak for math answers.
Minimum fix: implement full-answer scoring for AIME-style answers, for example `"000"` through `"999"` sequence probabilities. For MATH500, either restrict to answers that can be cleanly normalized or use a task with closed answer choices. First-digit scoring should not be the main evidence.

3. The pre-registered tests mostly fail.
Minimum fix: do not frame the current results as confirming the original proposal. Either revise the paper around a new confirmed result after a held-out test, or treat this as a failed prereg plus a new hypothesis.

4. The best result is post-hoc and does not replicate yet.
Minimum fix: freeze the conditional-`σ` analysis now, including controls and exclusion rules, then test it on held-out data: more MATH500 items, AIME years, another math set, and at least one other model size.

5. `δ` is promising but not yet strong.
Minimum fix: measure it on denser layers or all layers if possible. With only `[7, 14, 18, 22, 27]`, "earliest stable layer" is coarse. Also show that the `δ` signal survives null-prompt calibration and full-answer scoring.

6. P3 populations are unstable.
Minimum fix: require minimum cell sizes before reporting ratios, for example `n >= 20` per group, or use quantile-based groups instead of fixed `τ`. Current AIME P3 is not meaningful with `early_correct n=1`.

7. One model is not enough.
Minimum fix: after the readout is fixed, run 1.5B, 7B, and 14B. But do not spend the next 10 GPU-hours scaling the current broken readout.

8. Length has many confounds.
Minimum fix: add controls for correctness, problem difficulty, answer type, censoring at max tokens, and maybe problem source. Also report whether long generations are truncated. If many outputs hit a token limit, length correlations are hard to read.

9. There is still no intervention.
Minimum fix: not required for a first metric paper, but needed for a stronger claim. A small intervention could test whether examples with earlier, sharper internal answer states really need less reasoning, or whether the metric is only correlated with surface generation behavior.

**What Would Unstick The Project**

The minimum evidence is not more scale. The minimum evidence is a measurement sanity pass.

You need one clean table showing that the lens readout is no longer mostly a global digit prior. I would want to see, for each benchmark and model: null-calibrated top-label distribution, uncalibrated top-label distribution, first/full-answer accuracy of the lens readout, entropy or effective number of labels, and the main correlations before and after calibration.

Then I would want one target-clean validation: either full-answer AIME scoring or an MCQ benchmark where labels are balanced and the answer set is natural. If the conditional-`σ` or `δ` result survives there, the project becomes salvageable.

**Next Move Ranking For 10 GPU-Hours**

1. **(b) Implement full-answer scoring.** This fixes the core target mismatch. For AIME, scoring `"000"` through `"999"` is the cleanest next step. Without this, `μ`, `δ`, and P3 remain hard to trust.
2. **(c) Run null-prompt calibration sweep.** This is already implemented and directly tests whether the `"9"` artifact is removable. Run it, but do not stop there if the target is still first digit.
3. **(d) Add an MCQ task.** This is a good sanity check because it avoids open-answer normalization and can expose whether the metric works when the label set is balanced.
4. **(a) Finish 1.5B + 14B on the current pipeline.** Lowest value right now. Scaling a biased readout mainly tells you whether the artifact scales.

My recommended small experiment set is: null-prompt calibration on current 7B outputs, full-answer AIME scoring on 7B, then one clean MCQ task if time remains. Only after the `"9"` dominance is reduced and the same signal survives should you run 1.5B and 14B.

</details>

### Actions Taken (in this round, completed locally)

- [x] `lib/lens.py`: tuned-lens load now asserts `min ||W - I||_F > 0.1`, raises `RuntimeError` on identity.
- [x] `lib/lens.py`: `lens_distribution(calibration_logits=...)` param; `lens_logits()` helper.
- [x] `scripts/extract_lib.py`: `--null-prompt` flag, caches per-layer null logits, subtracts before `A(q)`.
- [x] `scripts/analysis_lib.py`: partial-Spearman (rank-residual Pearson on rankdata), bootstrap-ratio τ-sweep (n_boot=2000), population rename.
- [x] `configs/r1_qwen_{1_5b,7b,14b}.yaml`: `tau: 0.20` to match observed σ range (was 0.6 → degenerate).
- [x] Re-ran 7B on AIME24 + MATH500 with real tuned-lens (results in `results/lib/DeepSeek-R1-Distill-Qwen-7B/*tunedlens*.json`).

### Actions planned for Round 3 (this queue)

Local-code work, priority 1 (blocking Round 3 review):
- [ ] **Full-answer scoring**: add `full_answer_vocab(tok, candidates)` that returns per-candidate sequence log-prob from a lens-induced next-token distribution chain. For AIME: candidates = `["000", "001", ..., "999"]`. Replaces first-digit `integer_vocab`.
- [ ] **Calibration sanity report** in `scripts/analysis_lib.py`: emit (a) argmax label frequency table, (b) argmax entropy across problems, (c) first/full-answer accuracy of lens readout, (d) mean log-odds lift over null prompt — per layer. Purpose: one-glance table that shows whether "9 everywhere" is gone after calibration.
- [ ] **Minimum cell size guard in P3**: require `n >= 20` per group to report the ratio; else mark `null` with reason `insufficient_cell`.
- [ ] **Denser δ layers**: switch to `lens_layers = list(range(num_layers))` for the final analysis pass (keeps the ckpt intact — affines we don't have get logit-lens fallback? No — per your RoundT assert, must train denser tuned-lens or revert to logit-lens with clear warning).

GPU runs (queued, user executes):
- [ ] Null-prompt calibration sweep on 7B → compare uncalibrated vs calibrated artifact table.
- [ ] Full-answer scoring on 7B AIME (n=30, candidates `000`..`999`).
- [ ] MCQ benchmark (Knowlogic or CommonsenseQA subset).
- [ ] Only then: 1.5B + 14B replication.

### Results

- 7B on AIME24 + MATH500 with tuned-lens, no null calibration, first-digit vocab:

| Metric | AIME (n=30) | MATH500 (n=100) |
|---|---|---|
| CoT accuracy | 0.60 | 0.60 |
| Spearman(σ, length) | 0.095 (p=0.62) | −0.176 (p=0.08) |
| Spearman(δ, length) | 0.311 (p=0.09) | **0.233 (p=0.02)** |
| R² σ only | 0.009 | 0.037 |
| R² σ+δ | 0.083 | 0.048 |
| R² σ+δ+κ | 0.135 | 0.049 |
| Partial ρ(δ, len \| correct, κ, σ) | 0.271 (p=0.15) | 0.183 (p=0.07) |
| Partial ρ(σ, len \| correct, κ) | 0.331 (p=0.07) | **−0.235 (p=0.019)** |
| P1 pass | no | no |
| P2 pass | yes | no |
| P3 pass | no (degen n=1) | no (CI inc. 1) |

- Novel post-hoc signal: **MATH500 partial Spearman(σ, length | correct, κ) = −0.235, p=0.019**. Interpretation is confounded while readout is digit-prior-dominated.

### Status

- Continuing to Round 3 after implementing full-answer scoring + calibration sanity report locally, then user re-runs 7B extraction on GPU.
- Not-ready on all 9 weakness items; 4 have local-code fixes landing this round; 5 require GPU or new data.

