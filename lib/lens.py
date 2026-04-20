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
    identity_eps: float = 0.1,
) -> LogitLens | TunedLens:
    if kind == "logit":
        return LogitLens(lm_head, ln_f)
    if kind == "tuned":
        if lens_path is None or layers is None:
            raise ValueError("tuned lens requires lens_path and layers")
        # Explicit failure, no silent fallback — Round-2 reviewer asked for this:
        # truncated/corrupt ckpts previously made tuned lens indistinguishable
        # from logit lens.
        state = torch.load(lens_path, map_location="cpu", weights_only=True)
        d = lm_head.in_features
        affines = {}
        min_deviation = float("inf")
        for ℓ in layers:
            key = f"layer_{ℓ}"
            if key not in state:
                raise KeyError(f"tuned-lens checkpoint missing {key}")
            W = state[key]["W"]
            b = state[key]["b"]
            dev = float((W.float() - torch.eye(d)).norm())
            min_deviation = min(min_deviation, dev)
            linear = nn.Linear(d, d, bias=True)
            linear.weight.data = W.to(lm_head.weight.dtype)
            linear.bias.data = b.to(lm_head.weight.dtype)
            linear.to(lm_head.weight.device)
            affines[ℓ] = linear
        if min_deviation < identity_eps:
            raise RuntimeError(
                f"tuned-lens affines look like identity (min ||W-I||_F = {min_deviation:.4f} < {identity_eps}). "
                "Training likely did not run — refuse to fall back to logit lens silently."
            )
        return TunedLens(affines, lm_head, ln_f)
    raise ValueError(f"unknown lens kind {kind!r}")


def lens_distribution(
    lens: LogitLens | TunedLens,
    h: torch.Tensor,
    layer: int,
    answer_token_ids: torch.LongTensor,
    calibration_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """h: (d,) or (B, d). Returns probability over `answer_token_ids` via softmax of
    the full-vocab logits restricted to those indices.

    If `calibration_logits` (shape (V,)) is provided, subtract it from the lens
    output before restricting — this implements the null-prompt baseline calibration
    that the Round-2 reviewer requested to counteract the lens's global prior over
    digit tokens (the "argmax = 9 for 29/30 AIME" pathology).
    """
    logits = lens(h, layer=layer) if isinstance(lens, TunedLens) else lens(h, layer=layer)
    if calibration_logits is not None:
        logits = logits - calibration_logits.to(logits.device, dtype=logits.dtype)
    sub = logits.index_select(-1, answer_token_ids.to(logits.device))
    return torch.softmax(sub.float(), dim=-1)


def lens_logits(
    lens: LogitLens | TunedLens,
    h: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Raw lens logits at one position — used for computing the null-prompt
    calibration vector."""
    return lens(h, layer=layer)


def score_full_answers(
    lens: LogitLens | TunedLens,
    hidden_states_at_positions: torch.Tensor,   # (M, d) where M = max candidate length
    layer: int,
    candidate_token_id_lists: list[list[int]],
    calibration_logits_per_position: torch.Tensor | None = None,  # (M, V)
) -> torch.Tensor:
    """Score each candidate as a teacher-forced sequence log-prob under the lens.

    For candidate c = (t_0, t_1, ..., t_{k-1}):
        log P_lens(c) = sum_{i=0..k-1} log softmax(lens(h_i))[t_i]
    where h_i is the hidden state at position p* + i (prompt's last token is p*,
    then we append c_0..c_{k-1} and read hidden states at positions that predict
    each t_i given the prefix).

    Returns: tensor of shape (K,) with log-probs per candidate, normalized across
    candidates via log_softmax. Callers typically exp() to get pi over candidates.

    Added in Round 2 to replace first-digit scoring ("argmax=9" pathology)."""
    K = len(candidate_token_id_lists)
    device = hidden_states_at_positions.device
    log_probs = torch.zeros(K, dtype=torch.float32, device=device)

    # Cache per-position log-softmax over vocab (compute once).
    per_pos_logprobs: dict[int, torch.Tensor] = {}
    for i in range(hidden_states_at_positions.shape[0]):
        logits = lens(hidden_states_at_positions[i], layer=layer)
        if calibration_logits_per_position is not None:
            logits = logits - calibration_logits_per_position[i].to(
                logits.device, dtype=logits.dtype
            )
        per_pos_logprobs[i] = torch.log_softmax(logits.float(), dim=-1)

    for k, toks in enumerate(candidate_token_id_lists):
        lp = 0.0
        for i, tok_id in enumerate(toks):
            if i not in per_pos_logprobs:
                # Candidate longer than available positions — penalize to -inf.
                lp = float("-inf")
                break
            lp = lp + float(per_pos_logprobs[i][tok_id].item())
        log_probs[k] = lp

    return torch.log_softmax(log_probs, dim=-1)
