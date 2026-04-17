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
