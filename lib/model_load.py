"""Unified model + tokenizer loader that applies optional RoPE scaling / context
extension from the YAML config.

Usage:
    tok, mdl, device = load_model(cfg)
"""
from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from lib.config import Cfg


def load_model(cfg: Cfg) -> Tuple[PreTrainedTokenizerBase, "AutoModelForCausalLM", str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg.model.dtype)

    tok = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    hf_cfg = AutoConfig.from_pretrained(cfg.model.hf_id)
    rope = getattr(cfg.model, "rope_scaling", None)
    max_pos = getattr(cfg.model, "max_position_embeddings", None)

    if rope:
        hf_cfg.rope_scaling = dict(rope)
    if max_pos is not None:
        hf_cfg.max_position_embeddings = int(max_pos)
        if hasattr(tok, "model_max_length"):
            tok.model_max_length = int(max_pos)

    mdl = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_id,
        config=hf_cfg,
        torch_dtype=dtype,
        device_map=device,
    )
    mdl.eval()
    return tok, mdl, device
