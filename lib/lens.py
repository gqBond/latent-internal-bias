"""Unified wrapper over raw logit-lens and Belrose-et-al. tuned-lens.

All functions expect a hidden-state tensor of shape (d_model,) or (B, d_model) and
return logits over the full vocabulary of shape (..., V). Restriction to an answer
vocabulary is applied downstream.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn


LensType = Literal["logit", "tuned"]


class LogitLens:
    """Plain logit-lens: final-layer unembedding applied to intermediate hidden state."""

    def __init__(self, lm_head: nn.Linear, ln_f: nn.Module | None):
        self.lm_head = lm_head
        self.ln_f = ln_f

    def __call__(self, h: torch.Tensor, layer: int | None = None) -> torch.Tensor:
        if self.ln_f is not None:
            h = self.ln_f(h)
        return self.lm_head(h)


class TunedLens:
    """Per-layer affine probe W_ℓ h + b_ℓ → then final lm_head."""

    def __init__(self, affines: dict[int, nn.Linear], lm_head: nn.Linear, ln_f: nn.Module | None):
        self.affines = affines
        self.lm_head = lm_head
        self.ln_f = ln_f

    def __call__(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        if layer not in self.affines:
            raise KeyError(f"no tuned-lens affine for layer {layer} (available: {list(self.affines)})")
        h = self.affines[layer](h)
        if self.ln_f is not None:
            h = self.ln_f(h)
        return self.lm_head(h)


def make_lens(
    kind: LensType,
    lm_head: nn.Linear,
    ln_f: nn.Module | None,
    lens_path: Path | None = None,
    layers: list[int] | None = None,
) -> LogitLens | TunedLens:
    if kind == "logit":
        return LogitLens(lm_head, ln_f)
    if kind == "tuned":
        if lens_path is None or layers is None:
            raise ValueError("tuned lens requires lens_path and layers")
        affines = {}
        state = torch.load(lens_path, map_location="cpu")
        d = lm_head.in_features
        for ℓ in layers:
            key = f"layer_{ℓ}"
            if key not in state:
                raise KeyError(f"tuned-lens checkpoint missing {key}")
            linear = nn.Linear(d, d, bias=True)
            linear.weight.data = state[key]["W"].to(lm_head.weight.dtype)
            linear.bias.data = state[key]["b"].to(lm_head.weight.dtype)
            linear.to(lm_head.weight.device)
            affines[ℓ] = linear
        return TunedLens(affines, lm_head, ln_f)
    raise ValueError(f"unknown lens kind {kind!r}")


def lens_distribution(
    lens: LogitLens | TunedLens,
    h: torch.Tensor,
    layer: int,
    answer_token_ids: torch.LongTensor,
) -> torch.Tensor:
    """h: (d,) or (B, d). Returns probability over `answer_token_ids` via softmax of
    the full-vocab logits restricted to those indices."""
    logits = lens(h, layer=layer) if isinstance(lens, TunedLens) else lens(h, layer=layer)
    sub = logits.index_select(-1, answer_token_ids.to(logits.device))
    return torch.softmax(sub.float(), dim=-1)
