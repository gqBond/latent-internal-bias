"""Extract LIB (Latent Internal Bias) scalars for each problem.

Pipeline per problem:
    1. Build pre-think prompt (ends just after `<think>` tag).
    2. One forward pass; cache hidden states at configured layers, at position p*
       = last token of the prompt.
    3. Build answer vocabulary A(q):
         - mcq   -> {A,B,C,D,...}
         - integer -> union of first-digit tokens of K direct-answer samples
    4. Apply lens (tuned or logit) at each layer, restrict logits to A(q),
       softmax -> pi[ℓ].
    5. Pair with full-CoT and direct-answer outputs (from eval_*.py) to compute
       σ, μ, μ_correct, δ, κ and save.

Usage:
    python -m scripts.extract_lib \
        --cfg configs/r1_qwen_7b.yaml \
        --problems data/aime/aime2024.jsonl \
        --cot-out results/cot/R1-Distill-Qwen-7B/aime2024_cot.jsonl \
        --direct-out results/direct/R1-Distill-Qwen-7B/aime2024_direct.jsonl \
        --lens-path results/lenses/R1-Distill-Qwen-7B/tuned_lens.pt \
        --out results/lib/R1-Distill-Qwen-7B/aime2024_lib.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from lib.answer_vocab import (
    AnswerVocab,
    FullAnswerVocab,
    full_answer_vocab,
    integer_vocab,
    mcq_vocab,
)
from lib.config import load_cfg
from lib.datasets import normalize_row
from lib.io_utils import dump_json, read_jsonl, set_seeds, write_jsonl
from lib.lens import lens_distribution, lens_logits, make_lens, score_full_answers
from lib.metrics import compute_lib
from lib.model_load import load_model
from lib.prompting import build_pre_think_prompt


def _build_vocab(tok, row, direct_row) -> AnswerVocab:
    fmt = row["format"]
    if fmt == "mcq":
        n = len(row.get("choices", ["A", "B", "C", "D"]))
        return mcq_vocab(tok, num_choices=n)
    das = [a for a in direct_row.get("direct_answers", []) if a is not None]
    return integer_vocab(tok, das)


def _build_full_vocab(tok, row, direct_row, cot_row) -> FullAnswerVocab:
    """Round-2 reviewer ask: score full answer strings instead of first digit.

    Candidates = dedup(direct_answers ∪ {cot_answer, correct_answer}). Each is
    tokenized as a sequence and scored by teacher-forced lens log-prob."""
    das = [a for a in direct_row.get("direct_answers", []) if a]
    cot = cot_row.get("cot_answer") or ""
    correct = row.get("answer") or ""
    cands = [c for c in (das + [cot, correct]) if c]
    return full_answer_vocab(tok, cands)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--problems", required=True, help="JSONL with raw problems (id, question, answer, format).")
    ap.add_argument("--cot-out", required=True, help="Output of eval_*.py CoT file.")
    ap.add_argument("--direct-out", required=True, help="Output of eval_*.py direct-answer file.")
    ap.add_argument("--lens-path", default=None, help="Tuned-lens checkpoint (required if lens.type=tuned).")
    ap.add_argument("--null-prompt", default=None,
                    help="Format string with '{format}' placeholder used to build a neutral prompt whose "
                         "lens logits are subtracted from each problem's lens logits — a cheap global-prior "
                         "calibration for the 'argmax=9 everywhere' pathology. If omitted, no calibration.")
    ap.add_argument("--full-answer", action="store_true",
                    help="Round-2 reviewer ask: score full candidate answer strings (teacher-forced "
                         "lens sequence log-prob) instead of first-digit vocab. Candidates = dedup "
                         "(direct_answers ∪ {cot_answer, correct_answer}). Cost: one extra forward "
                         "pass per candidate per problem.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seeds(cfg.generation.seed)

    tok, mdl, device = load_model(cfg)

    ln_f = getattr(mdl.model, "norm", None)
    lm_head = mdl.get_output_embeddings()
    lens = make_lens(
        cfg.lens.type,
        lm_head=lm_head,
        ln_f=ln_f,
        lens_path=Path(args.lens_path) if args.lens_path else None,
        layers=cfg.model.lens_layers,
    )

    problems = [normalize_row(r, i) for i, r in enumerate(read_jsonl(args.problems))]
    cot_rows = {r["id"]: r for r in read_jsonl(args.cot_out)}
    direct_rows = {r["id"]: r for r in read_jsonl(args.direct_out)}

    # Optional null-prompt baseline: one forward pass on a neutral question,
    # cached per layer. Subtracted from each problem's lens logits before
    # restricting to A(q) — removes the lens's global token prior.
    calib_per_layer: dict[int, torch.Tensor] = {}
    if args.null_prompt:
        null_text = args.null_prompt.format(format="integer")
        null_prompt = build_pre_think_prompt(tok, null_text, "integer")
        null_enc = tok(null_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            null_out = mdl(**null_enc, output_hidden_states=True, use_cache=False)
        for ℓ in cfg.model.lens_layers:
            h_null = null_out.hidden_states[ℓ + 1][0, -1]
            calib_per_layer[ℓ] = lens_logits(lens, h_null, layer=ℓ).detach()
        print(f"null-prompt calibration built for layers {list(calib_per_layer)}")

    out_rows = []
    for ex in tqdm(problems, desc="extract-LIB"):
        if ex["id"] not in cot_rows or ex["id"] not in direct_rows:
            print(f"  skipping {ex['id']}: missing cot/direct row")
            continue
        cot = cot_rows[ex["id"]]
        drow = direct_rows[ex["id"]]

        prompt = build_pre_think_prompt(tok, ex["question"], ex["format"])
        enc = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = mdl(
                **enc,
                output_hidden_states=True,
                use_cache=False,
            )
        h_layers = out.hidden_states  # tuple len = num_layers + 1; [0] is embeddings

        pi_per_layer = {}
        if args.full_answer:
            full_vocab = _build_full_vocab(tok, ex, drow, cot)
            if not full_vocab.labels:
                print(f"  skipping {ex['id']}: no full-answer candidates")
                continue
            prompt_ids = enc.input_ids[0]
            p_star = prompt_ids.shape[0] - 1
            # Collect hidden states at positions p*..p*+M-1 for each candidate.
            cand_h_per_layer: dict[int, list[torch.Tensor]] = {
                ℓ: [] for ℓ in cfg.model.lens_layers
            }
            for toks in full_vocab.token_id_lists:
                full_ids = torch.cat([
                    prompt_ids,
                    torch.tensor(toks, dtype=prompt_ids.dtype, device=device),
                ], dim=0).unsqueeze(0)
                with torch.no_grad():
                    c_out = mdl(input_ids=full_ids, output_hidden_states=True, use_cache=False)
                M = len(toks)
                for ℓ in cfg.model.lens_layers:
                    cand_h_per_layer[ℓ].append(
                        c_out.hidden_states[ℓ + 1][0, p_star:p_star + M]
                    )
            # Score candidates per layer → softmax across candidates.
            for ℓ in cfg.model.lens_layers:
                lps = []
                for k, toks in enumerate(full_vocab.token_id_lists):
                    h_k = cand_h_per_layer[ℓ][k]
                    lp = 0.0
                    for i, tid in enumerate(toks):
                        logits = lens_logits(lens, h_k[i], layer=ℓ)
                        if ℓ in calib_per_layer:
                            logits = logits - calib_per_layer[ℓ].to(
                                logits.device, dtype=logits.dtype
                            )
                        lp += float(torch.log_softmax(logits.float(), dim=-1)[tid].item())
                    lps.append(lp)
                pi = torch.softmax(torch.tensor(lps, dtype=torch.float32), dim=-1)
                pi_per_layer[ℓ] = pi
            labels = full_vocab.labels
        else:
            vocab = _build_vocab(tok, ex, drow)
            answer_ids = vocab.token_ids.to(device)
            for ℓ in cfg.model.lens_layers:
                h_last = h_layers[ℓ + 1][0, -1]            # (d,)
                pi = lens_distribution(
                    lens, h_last, layer=ℓ, answer_token_ids=answer_ids,
                    calibration_logits=calib_per_layer.get(ℓ),
                )
                pi_per_layer[ℓ] = pi.detach().cpu()
            labels = vocab.labels

        lib = compute_lib(
            pi_per_layer=pi_per_layer,
            labels=labels,
            final_answer=cot["cot_answer"] or "",
            correct_answer=ex["answer"],
            num_model_layers=cfg.model.num_layers,
        )

        out_rows.append({
            "id": ex["id"],
            "format": ex["format"],
            "labels": labels,
            "scoring_mode": "full_answer" if args.full_answer else "first_digit",
            "pi_per_layer": {str(k): v.tolist() for k, v in pi_per_layer.items()},
            "sigma": lib.sigma,
            "mu": lib.mu,
            "mu_correct": lib.mu_correct,
            "delta": lib.delta,
            "kappa": lib.kappa,
            "bias_argmax": lib.bias_argmax_label,
            "cot_answer": cot["cot_answer"],
            "cot_correct": cot["cot_correct"],
            "cot_length": cot["cot_length"],
            "direct_argmax": drow.get("direct_answer_argmax"),
            "direct_matches_cot": int(str(drow.get("direct_answer_argmax")) == str(cot["cot_answer"])),
            "correct_answer": ex["answer"],
        })

    write_jsonl(args.out, out_rows)
    dump_json(Path(args.out).with_suffix(".meta.json"), {
        "model": cfg.model.name,
        "lens_type": cfg.lens.type,
        "layers": cfg.model.lens_layers,
        "n_problems": len(out_rows),
        "scoring_mode": "full_answer" if args.full_answer else "first_digit",
        "null_prompt_calibration": args.null_prompt is not None,
    })
    print(f"wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
