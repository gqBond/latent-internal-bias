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
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.answer_vocab import AnswerVocab, integer_vocab, mcq_vocab
from lib.config import load_cfg
from lib.io_utils import dump_json, read_jsonl, set_seeds, write_jsonl
from lib.lens import lens_distribution, make_lens
from lib.metrics import compute_lib
from lib.prompting import build_pre_think_prompt


def _build_vocab(tok, row, direct_row) -> AnswerVocab:
    fmt = row["format"]
    if fmt == "mcq":
        n = len(row.get("choices", ["A", "B", "C", "D"]))
        return mcq_vocab(tok, num_choices=n)
    das = [a for a in direct_row.get("direct_answers", []) if a is not None]
    return integer_vocab(tok, das)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--problems", required=True, help="JSONL with raw problems (id, question, answer, format).")
    ap.add_argument("--cot-out", required=True, help="Output of eval_*.py CoT file.")
    ap.add_argument("--direct-out", required=True, help="Output of eval_*.py direct-answer file.")
    ap.add_argument("--lens-path", default=None, help="Tuned-lens checkpoint (required if lens.type=tuned).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seeds(cfg.generation.seed)

    dtype = getattr(torch, cfg.model.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_id, torch_dtype=dtype, device_map=device
    )
    mdl.eval()

    ln_f = getattr(mdl.model, "norm", None)
    lm_head = mdl.get_output_embeddings()
    lens = make_lens(
        cfg.lens.type,
        lm_head=lm_head,
        ln_f=ln_f,
        lens_path=Path(args.lens_path) if args.lens_path else None,
        layers=cfg.model.lens_layers,
    )

    problems = read_jsonl(args.problems)
    cot_rows = {r["id"]: r for r in read_jsonl(args.cot_out)}
    direct_rows = {r["id"]: r for r in read_jsonl(args.direct_out)}

    out_rows = []
    for ex in tqdm(problems, desc="extract-LIB"):
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
        vocab = _build_vocab(tok, ex, drow)
        answer_ids = vocab.token_ids.to(device)

        for ℓ in cfg.model.lens_layers:
            h_last = h_layers[ℓ + 1][0, -1]            # (d,)
            pi = lens_distribution(lens, h_last, layer=ℓ, answer_token_ids=answer_ids)
            pi_per_layer[ℓ] = pi.detach().cpu()

        lib = compute_lib(
            pi_per_layer=pi_per_layer,
            labels=vocab.labels,
            final_answer=cot["cot_answer"] or "",
            correct_answer=ex["answer"],
            num_model_layers=cfg.model.num_layers,
        )

        out_rows.append({
            "id": ex["id"],
            "format": ex["format"],
            "labels": vocab.labels,
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
    })
    print(f"wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
