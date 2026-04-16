# Reference Paper Summary

**Title**: The First Impression Problem: Internal Bias Triggers Overthinking in Reasoning Models
**Authors**: Renfei Dang, Zhening Li, Shujian Huang, Jiajun Chen (NJUNLP)
**Venue**: ICLR 2026 (poster); OpenReview id `2PP70tFY0S`
**arXiv**: 2505.16448
**Code**: https://github.com/NJUNLP/LongCoT-Internal-Bias

## What They Did
Introduce **"internal bias"** — a preliminary answer guess formed by a reasoning model immediately upon reading a problem, *before* any systematic reasoning. They operationalize it by **direct-answer prompting**: ask the model to answer the question with no CoT and compare that answer to the final CoT answer. They show internal bias is causally tied to overthinking via two counterfactual interventions:
1. **MASK intervention** — remove / mask the input question after reasoning starts → reasoning shortens.
2. **Bias-injection** — manually prepend a biased first-impression → overthinking is modulated.

Interpretability analysis identifies **excessive attention to input-question tokens** as the causal pathway.

## Key Results
- Statistically significant positive correlation between **bias-deviation degree** (direct-answer ≠ final-answer) and **reasoning length** (proxy for overthinking).
- Relative length increases of **+17.2 % to +42.1 %** in high-bias-deviation cases.
- Effect persists across tested models (R1-Distill-Qwen-7B/14B/32B, QwQ, DeepSeek-R1-API) and benchmarks (AIME24/25, Knowlogic, CharCount in EN and ZH).
- Several tested mitigation strategies **fail** to remove the bias influence.

## Limitations & Open Questions
1. **Metric is not truly "internal"** — bias is measured by another explicit generation (direct-answer prompt), not from internal states.
2. **Binary metric** — bias-deviation is essentially a 0/1 match between direct-answer and final-answer, losing magnitude/confidence information.
3. **No decomposition** — bias that aligns with the *correct* answer (useful intuition) is lumped together with bias that aligns with a *wrong* answer (harmful prejudice).
4. **Diagnostic only** — authors acknowledge no proposed mitigation eliminates the effect.
5. **Narrow benchmark coverage** — only AIME, Knowlogic, CharCount; no broader math/logic/code evaluation.
6. **Sampling noise** — direct-answer metric is a single greedy generation; no confidence/distribution over candidates.
7. **No per-step or per-layer structure** — treats bias as a single scalar per problem.

## Potential Improvement Directions
- Replace direct-answer prompting with **hidden-state-based logit-lens / tuned-lens** projections at the post-question pre-reasoning position → continuous, no extra forward pass, truly internal.
- **Decompose bias** into *intuition* (bias == correct) vs *prejudice* (bias == wrong common answer) — they have opposite effects on overthinking.
- **Layer-wise emergence depth**: at what depth does the final answer become the argmax in the logit-lens? Shallow lock-in = confident, deep lock-in / oscillation = overthinking risk.
- Use the improved metric as an **online signal** for adaptive early-exit or bias-injection-correction.

## Codebase
- `AIME/{aime2024,aime2025,direct*,dpsk*,qwq*,mask_aime*}.py` — full-CoT, direct-answer, API, and MASK-intervention generation.
- `CharCount/` and `Knowlogic/` — same pattern for the other two benchmarks.
- `acc_and_length.py` — Table 2 (accuracy + length).
- `length_trend.py` — Table 1 (length vs bias deviation bucket).
- `hiddenStates/` — Figure 4 (attention bars) and Figure 5 (mask-or-not reasoning).
- Runs on R1-Distill-Qwen (7B/14B/32B), QwQ-32B, DeepSeek-R1 (via API).
