"""Prejudice-conditional adaptive early exit.

At each reasoning-boundary token (e.g., the next token starts with `"Wait"` /
`"Alternatively"` / `"Hmm"`), recompute a running logit-lens argmax from the hidden
state at the boundary position. If the bias population for this problem is
`intuition` and the running argmax has matched the initial bias for N consecutive
boundaries, close `</think>` and emit the bias answer.

Baseline comparisons (fixed-length truncation, Dang-style no-mitigation) are run via
`--baseline` flags.

Usage:
    python -m scripts.mitigate_prejudice \
        --cfg configs/r1_qwen_7b.yaml \
        --lib results/lib/R1-Distill-Qwen-7B/aime2024_lib.jsonl \
        --problems data/aime/aime2024.jsonl \
        --lens-path results/lenses/R1-Distill-Qwen-7B/tuned_lens.pt \
        --mode lib_prejudice \
        --out results/mitigation/R1-Distill-Qwen-7B/aime2024_lib_prejudice.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessor

from lib.answer_vocab import canonicalize_integer, canonicalize_mcq
from lib.config import load_cfg
from lib.datasets import normalize_row
from lib.io_utils import dump_json, read_jsonl, set_seeds, write_jsonl
from lib.lens import lens_distribution, make_lens
from lib.model_load import load_model
from lib.prompting import build_cot_prompt


class _BoundaryTracker(LogitsProcessor):
    """Fires a callback whenever the *next* token is about to be generated after a
    token whose decoded text matches one of the boundary strings."""

    def __init__(self, tok, boundary_tokens: List[str], on_boundary):
        self.tok = tok
        self.boundary_prefixes = boundary_tokens
        self.on_boundary = on_boundary

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            return scores
        last_id = int(input_ids[0, -1])
        last_text = self.tok.decode([last_id], skip_special_tokens=True)
        if any(last_text.strip().startswith(b) for b in self.boundary_prefixes):
            self.on_boundary(input_ids)
        return scores


def _classify(row: Dict, tau: float) -> str:
    if row["sigma"] < tau:
        return "unbiased"
    return "intuition" if row["mu_correct"] == 1 else "prejudice"


def _greedy_extend(mdl, tok, ids: torch.Tensor, final_text: str, max_new: int = 32) -> str:
    """Force an early `</think>` then let the model emit its answer."""
    tail = "\n</think>\n\nThe answer is \\boxed{" + final_text + "}"
    tail_ids = tok(tail, return_tensors="pt", add_special_tokens=False).input_ids.to(ids.device)
    return tok.decode(torch.cat([ids[0], tail_ids[0]]), skip_special_tokens=True)


def run_mitigation(cfg, mode: str, lens_path: Optional[Path], problems: List[Dict],
                   lib_rows: Dict[str, Dict], out_path: Path,
                   n_boundary_hits: int = 2) -> None:
    set_seeds(cfg.generation.seed)
    tok, mdl, device = load_model(cfg)

    lens = None
    if mode == "lib_prejudice":
        ln_f = getattr(mdl.model, "norm", None)
        lens = make_lens(cfg.lens.type, lm_head=mdl.get_output_embeddings(),
                         ln_f=ln_f, lens_path=lens_path, layers=cfg.model.lens_layers)

    rows_out = []
    for ex in tqdm(problems, desc=f"mitigate[{mode}]"):
        lib = lib_rows.get(ex["id"])
        if lib is None:
            continue
        pop = _classify(lib, cfg.decomposition.tau)
        prompt = build_cot_prompt(tok, ex["question"], ex["format"])
        enc = tok(prompt, return_tensors="pt").to(device)

        if mode == "none":
            ids = mdl.generate(
                **enc, max_new_tokens=cfg.generation.max_new_tokens,
                do_sample=cfg.generation.temperature > 0,
                temperature=max(cfg.generation.temperature, 1e-5),
                top_p=cfg.generation.top_p, pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

        elif mode == "fixed":
            ids = mdl.generate(
                **enc,
                max_new_tokens=min(cfg.generation.max_new_tokens, 2048),
                do_sample=cfg.generation.temperature > 0,
                temperature=max(cfg.generation.temperature, 1e-5),
                top_p=cfg.generation.top_p, pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

        elif mode == "lib_prejudice":
            if pop != "intuition":
                ids = mdl.generate(
                    **enc, max_new_tokens=cfg.generation.max_new_tokens,
                    do_sample=cfg.generation.temperature > 0,
                    temperature=max(cfg.generation.temperature, 1e-5),
                    top_p=cfg.generation.top_p, pad_token_id=tok.eos_token_id,
                )
                text = tok.decode(ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            else:
                # streaming boundary-aware loop
                hits = {"count": 0, "stop": False}
                initial_bias = lib["bias_argmax"]
                answer_ids = tok(initial_bias, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

                def _on_boundary(cur_ids):
                    with torch.no_grad():
                        out = mdl(cur_ids, output_hidden_states=True, use_cache=False)
                    h = out.hidden_states[cfg.model.lens_layers[-1] + 1][0, -1]
                    pi = lens_distribution(lens, h, layer=cfg.model.lens_layers[-1], answer_token_ids=answer_ids)
                    if int(torch.argmax(pi).item()) == 0:
                        hits["count"] += 1
                    else:
                        hits["count"] = 0
                    if hits["count"] >= n_boundary_hits:
                        hits["stop"] = True

                proc = _BoundaryTracker(tok, cfg.decomposition.boundary_tokens, _on_boundary)
                generated = enc.input_ids
                max_steps = cfg.generation.max_new_tokens
                for _ in range(max_steps):
                    with torch.no_grad():
                        out = mdl(generated)
                    next_logits = out.logits[:, -1, :]
                    proc(generated, next_logits)
                    if hits["stop"]:
                        break
                    if cfg.generation.temperature > 0:
                        probs = torch.softmax(next_logits / cfg.generation.temperature, dim=-1)
                        next_id = torch.multinomial(probs, 1)
                    else:
                        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_id], dim=1)
                    if int(next_id) == tok.eos_token_id:
                        break

                text = _greedy_extend(mdl, tok, generated, initial_bias) if hits["stop"] else \
                    tok.decode(generated[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            raise ValueError(f"unknown mode {mode}")

        fmt = ex["format"]
        canon = canonicalize_mcq(text) if fmt == "mcq" else canonicalize_integer(text)
        rows_out.append({
            "id": ex["id"],
            "population": pop,
            "mode": mode,
            "answer": canon,
            "correct_answer": ex["answer"],
            "correct": int(str(canon) == str(ex["answer"])),
            "tokens": len(tok.encode(text, add_special_tokens=False)),
            "text": text,
        })

    write_jsonl(out_path, rows_out)
    dump_json(Path(out_path).with_suffix(".meta.json"), {
        "mode": mode,
        "n": len(rows_out),
        "tau": cfg.decomposition.tau,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--lens-path", default=None)
    ap.add_argument("--mode", choices=["none", "fixed", "lib_prejudice"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boundary-hits", type=int, default=2)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    problems = [normalize_row(r, i) for i, r in enumerate(read_jsonl(args.problems))]
    lib_rows = {r["id"]: r for r in read_jsonl(args.lib)}

    run_mitigation(
        cfg, args.mode,
        Path(args.lens_path) if args.lens_path else None,
        problems, lib_rows, Path(args.out),
        n_boundary_hits=args.n_boundary_hits,
    )


if __name__ == "__main__":
    main()
