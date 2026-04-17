"""Shared CoT and direct-answer evaluation loop used by eval_{aime,math500,knowlogic}.py.

Writes JSONL with one row per problem:
    {"id", "question", "answer", "format", "cot_text", "cot_answer", "cot_correct",
     "cot_length", "direct_samples": [...], "direct_answer_argmax"}
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List

import torch
from tqdm import tqdm

from lib.answer_vocab import canonicalize_integer, canonicalize_mcq
from lib.config import Cfg
from lib.io_utils import set_seeds, write_jsonl
from lib.model_load import load_model
from lib.prompting import build_cot_prompt, build_direct_prompt


def _generate(tok, mdl, device, prompt: str, max_new: int, temperature: float, top_p: float,
              num_return: int = 1) -> List[str]:
    enc = tok(prompt, return_tensors="pt").to(device)
    gen = mdl.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        num_return_sequences=num_return,
        pad_token_id=tok.eos_token_id,
    )
    out_ids = gen[:, enc.input_ids.shape[1] :]
    return [tok.decode(o, skip_special_tokens=True) for o in out_ids]


def _canon(text: str, fmt: str) -> str | None:
    return canonicalize_mcq(text) if fmt == "mcq" else canonicalize_integer(text)


def run_eval(
    cfg: Cfg,
    problems: List[Dict],
    out_cot: Path,
    out_direct: Path,
) -> None:
    set_seeds(cfg.generation.seed)
    tok, mdl, device = load_model(cfg)

    rows_cot, rows_direct = [], []
    for ex in tqdm(problems, desc="eval"):
        fmt = ex["format"]

        cot_prompt = build_cot_prompt(tok, ex["question"], fmt)
        cot_text = _generate(
            tok, mdl, device, cot_prompt,
            max_new=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            top_p=cfg.generation.top_p,
        )[0]
        cot_answer = _canon(cot_text, fmt)
        cot_correct = int(str(cot_answer) == str(ex["answer"]))
        cot_length = len(tok.encode(cot_text, add_special_tokens=False))
        rows_cot.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "format": fmt,
            "cot_text": cot_text,
            "cot_answer": cot_answer,
            "cot_correct": cot_correct,
            "cot_length": cot_length,
        })

        direct_prompt = build_direct_prompt(tok, ex["question"], fmt)
        direct_samples = _generate(
            tok, mdl, device, direct_prompt,
            max_new=cfg.direct_answer.max_new_tokens,
            temperature=cfg.direct_answer.temperature,
            top_p=cfg.direct_answer.top_p,
            num_return=cfg.direct_answer.num_samples,
        )
        direct_answers = [_canon(s, fmt) for s in direct_samples]
        valid = [a for a in direct_answers if a is not None]
        argmax_answer = Counter(valid).most_common(1)[0][0] if valid else None
        rows_direct.append({
            "id": ex["id"],
            "direct_samples": direct_samples,
            "direct_answers": direct_answers,
            "direct_answer_argmax": argmax_answer,
        })

    write_jsonl(out_cot, rows_cot)
    write_jsonl(out_direct, rows_direct)


def cli_main(loader: Callable[[], List[Dict]], out_stem: str) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out-cot", default=None)
    ap.add_argument("--out-direct", default=None)
    args = ap.parse_args()

    from lib.config import load_cfg
    cfg = load_cfg(args.cfg)

    out_cot = Path(args.out_cot or f"{cfg.paths.cot_dir}/{cfg.model.name}/{out_stem}_cot.jsonl")
    out_direct = Path(args.out_direct or f"{cfg.paths.direct_dir}/{cfg.model.name}/{out_stem}_direct.jsonl")

    problems = loader()
    run_eval(cfg, problems, out_cot, out_direct)
