"""Core LIB scalars.

Given, per problem:
    - pi[ℓ] : probability over |A(q)| tokens at each layer ℓ in `layers`
    - labels: list[str] of the answer tokens, aligned to pi's last dim
    - final_answer: str (from full CoT)
    - correct_answer: str

compute:
    σ  = max_a pi[L](a)                              strength at the reporting layer
    μ  = 1[argmax pi[L] == final_answer]             alignment (binary — baseline)
    μc = 1[argmax pi[L] == correct_answer]           alignment to ground truth
    δ  = min { ℓ / L : argmax pi[ℓ] == argmax pi[L] } emergence depth (in [0, 1])
    κ  = KL(pi[L] || Uniform)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class LIBScalars:
    sigma: float
    mu: int               # 0/1
    mu_correct: int       # 0/1
    delta: float
    kappa: float
    bias_argmax_label: str


def compute_lib(
    pi_per_layer: Dict[int, torch.Tensor],     # layer -> (|A|,)
    labels: List[str],
    final_answer: str,
    correct_answer: str,
    num_model_layers: int,
) -> LIBScalars:
    layers_sorted = sorted(pi_per_layer.keys())
    L_layer = layers_sorted[-1]
    piL = pi_per_layer[L_layer]
    topL = int(torch.argmax(piL).item())
    label_top = labels[topL]

    sigma = float(piL[topL].item())
    mu = int(label_top == str(final_answer))
    mu_correct = int(label_top == str(correct_answer))

    delta_layer: int | None = None
    for ℓ in layers_sorted:
        if int(torch.argmax(pi_per_layer[ℓ]).item()) == topL:
            delta_layer = ℓ
            break
    if delta_layer is None:
        delta_layer = L_layer
    delta = delta_layer / max(1, num_model_layers - 1)

    K = len(labels)
    uni = 1.0 / K
    kappa = sum(
        float(p.item()) * math.log(max(float(p.item()), 1e-12) / uni)
        for p in piL
    )

    return LIBScalars(
        sigma=sigma,
        mu=mu,
        mu_correct=mu_correct,
        delta=delta,
        kappa=kappa,
        bias_argmax_label=label_top,
    )
