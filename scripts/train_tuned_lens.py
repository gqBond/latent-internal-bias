"""Train per-layer affine tuned-lens probes for a HF decoder model.

Minimal, single-GPU, uses Belrose-et-al. loss: cross-entropy against the LM's final
next-token distribution, so each affine learns to project layer-ℓ hidden states into
the model's output-vocabulary space.

Usage:
    python -m scripts.train_tuned_lens --cfg configs/r1_qwen_7b.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm

from lib.config import load_cfg
from lib.io_utils import set_seeds
from lib.model_load import load_model


def _hidden_states(model, input_ids, attention_mask):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states, out.logits


def _iter_batches(tok, ds_stream, max_len: int, batch_size: int):
    buf_text: list[str] = []
    for ex in ds_stream:
        t = ex.get("text") or ex.get("content") or ""
        if not t:
            continue
        buf_text.append(t)
        if len(buf_text) < batch_size:
            continue
        enc = tok(
            buf_text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        yield enc
        buf_text = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seeds(cfg.generation.seed)

    tok, model, device = load_model(cfg)
    dtype = getattr(torch, cfg.model.dtype)
    for p in model.parameters():
        p.requires_grad_(False)

    d = model.config.hidden_size
    affines = {
        ℓ: nn.Linear(d, d, bias=True).to(device=device, dtype=dtype)
        for ℓ in cfg.model.lens_layers
    }
    for ℓ, lin in affines.items():
        nn.init.eye_(lin.weight)
        nn.init.zeros_(lin.bias)

    opt = AdamW(
        [p for lin in affines.values() for p in lin.parameters()],
        lr=cfg.lens.train_lr,
    )

    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    batch_iter = _iter_batches(tok, ds, args.max_len, cfg.lens.train_batch_size)

    pbar = tqdm(range(cfg.lens.train_steps), desc="tuned-lens")
    for step in pbar:
        enc = next(batch_iter)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        with torch.no_grad():
            h_layers, logits = _hidden_states(model, input_ids, attn)
        target = torch.log_softmax(logits.float(), dim=-1)  # (B, T, V)

        loss = 0.0
        for ℓ in cfg.model.lens_layers:
            h = h_layers[ℓ + 1]  # hidden_states[0] is the embedding layer
            projected = affines[ℓ](h)
            if hasattr(model.model, "norm"):
                projected = model.model.norm(projected)
            lm_head = model.get_output_embeddings()
            lens_logits = lm_head(projected.to(lm_head.weight.dtype))
            lens_logp = torch.log_softmax(lens_logits.float(), dim=-1)
            kl = torch.nn.functional.kl_div(
                lens_logp, target, log_target=True, reduction="batchmean"
            )
            loss = loss + kl

        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=float(loss))

    out_dir = Path(args.out or f"{cfg.paths.lens_dir}/{cfg.model.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        f"layer_{ℓ}": {
            "W": lin.weight.detach().float().cpu(),
            "b": lin.bias.detach().float().cpu(),
        }
        for ℓ, lin in affines.items()
    }
    torch.save(state, out_dir / "tuned_lens.pt")
    print(f"saved tuned-lens to {out_dir/'tuned_lens.pt'}")


if __name__ == "__main__":
    main()
