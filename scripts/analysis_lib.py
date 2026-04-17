"""Analyze LIB extraction output.

Computes the pre-registered metrics (Spearman / Pearson correlations, ΔR², population
decomposition, stratified length), writes a summary JSON, and optionally saves plots.

Usage:
    python -m scripts.analysis_lib \
        --lib results/lib/R1-Distill-Qwen-7B/aime2024_lib.jsonl \
        --cfg configs/r1_qwen_7b.yaml \
        --out results/lib/R1-Distill-Qwen-7B/aime2024_summary.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.linear_model import LinearRegression

from lib.answer_vocab import canonicalize_to_vocab
from lib.config import load_cfg
from lib.io_utils import dump_json, read_jsonl


# Reframed population names (descriptive, per Round-1 reviewer feedback).
# Old name → new name: intuition → early_correct, prejudice → early_incorrect, unbiased → low_commitment.
POP_RENAME = {
    "intuition": "early_correct",
    "prejudice": "early_incorrect",
    "unbiased": "low_commitment",
}


def _pairs(rows: List[Dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def _recompute_match(rows: List[Dict]) -> int:
    """Recompute mu, mu_correct in-place by canonicalizing the multi-char
    final_answer/correct_answer onto the single-token labels vocab.

    Returns the number of rows whose mu or mu_correct changed — sanity telemetry
    for the P3 None issue seen before the canonicalization fix."""
    changed = 0
    for r in rows:
        labels = r.get("labels")
        if not labels:
            continue
        bias = r.get("bias_argmax") or ""
        f_can = canonicalize_to_vocab(r.get("cot_answer") or "", labels)
        c_can = canonicalize_to_vocab(r.get("correct_answer") or "", labels)
        new_mu = int(bias == f_can) if f_can else 0
        new_mc = int(bias == c_can) if c_can else 0
        if new_mu != r.get("mu") or new_mc != r.get("mu_correct"):
            changed += 1
        r["mu"] = new_mu
        r["mu_correct"] = new_mc
        r["final_canon"] = f_can
        r["correct_canon"] = c_can
    return changed


def _r2(X: np.ndarray, y: np.ndarray) -> float:
    reg = LinearRegression().fit(X, y)
    return float(reg.score(X, y))


def _partial_spearman(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """Partial Spearman correlation between y and x controlling for columns of z.

    Implemented as Pearson correlation of rank-residuals after regressing rank(y) and
    rank(x) on rank(z). Returns (stat, p) or (NaN, NaN) if undefined.
    """
    try:
        y_r = rankdata(y)
        x_r = rankdata(x)
        z_r = np.stack([rankdata(z[:, i]) for i in range(z.shape[1])], axis=1)
        y_res = y_r - LinearRegression().fit(z_r, y_r).predict(z_r)
        x_res = x_r - LinearRegression().fit(z_r, x_r).predict(z_r)
        stat, p = pearsonr(y_res, x_res)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def _bootstrap_ratio(rows: List[Dict], tau: float, n_boot: int = 2000,
                     rng: np.random.Generator | None = None) -> Dict:
    """Bootstrap CI for prejudice/intuition length ratio at threshold tau.

    Samples rows with replacement n_boot times, recomputes decomposition + ratio per
    bootstrap. Returns central estimate, 95% CI, and fraction of bootstrap samples
    where both populations were non-empty (otherwise ratio undefined).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = len(rows)
    ratios: List[float] = []
    n_defined = 0
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        sample = [rows[i] for i in idx]
        pops = _decomp(sample, tau)
        if pops["intuition"] and pops["prejudice"]:
            mi = float(np.mean([r["cot_length"] for r in pops["intuition"]]))
            mp = float(np.mean([r["cot_length"] for r in pops["prejudice"]]))
            if mi > 0:
                ratios.append(mp / mi)
                n_defined += 1
    if not ratios:
        return {"tau": tau, "n_boot": n_boot, "defined_frac": 0.0,
                "mean": None, "ci_low": None, "ci_high": None}
    arr = np.asarray(ratios)
    return {
        "tau": tau,
        "n_boot": n_boot,
        "defined_frac": n_defined / n_boot,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def _decomp(rows: List[Dict], tau: float) -> Dict[str, List[Dict]]:
    out = {"intuition": [], "prejudice": [], "unbiased": []}
    for r in rows:
        if r["sigma"] < tau:
            out["unbiased"].append(r)
        elif r["mu_correct"] == 1:
            out["intuition"].append(r)
        else:
            out["prejudice"].append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-recompute-match", action="store_true",
                    help="skip canonicalization-based recomputation of mu/mu_correct")
    ap.add_argument("--tau", type=float, default=None,
                    help="override decomposition.tau (default: cfg value)")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    rows = read_jsonl(args.lib)
    if not rows:
        raise RuntimeError(f"empty LIB file: {args.lib}")

    if not args.no_recompute_match:
        changed = _recompute_match(rows)
        if changed:
            print(f"recompute_match: updated mu/mu_correct on {changed}/{len(rows)} rows")

    tau = args.tau if args.tau is not None else cfg.decomposition.tau

    length = _pairs(rows, "cot_length")
    sigma = _pairs(rows, "sigma")
    mu = _pairs(rows, "mu")
    delta = _pairs(rows, "delta")
    kappa = _pairs(rows, "kappa")
    direct_match = np.array([r["direct_matches_cot"] for r in rows], dtype=float)

    def _safe(fn, a, b):
        try:
            return fn(a, b)
        except Exception as e:
            return (float("nan"), float("nan"))

    sp_sigma = _safe(spearmanr, sigma, length)
    sp_mu = _safe(spearmanr, mu, length)
    sp_delta = _safe(spearmanr, delta, length)
    sp_direct = _safe(spearmanr, direct_match, length)
    pr_sigma = _safe(pearsonr, sigma, length)
    pr_mu = _safe(pearsonr, mu, length)

    r2_sigma = _r2(sigma.reshape(-1, 1), length)
    r2_sigma_delta = _r2(np.stack([sigma, delta], axis=1), length)
    r2_sigma_delta_kappa = _r2(np.stack([sigma, delta, kappa], axis=1), length)

    pops = _decomp(rows, tau)
    pop_stats = {
        POP_RENAME[name]: {
            "n": len(rs),
            "mean_length": float(np.mean([r["cot_length"] for r in rs])) if rs else None,
            "median_length": float(np.median([r["cot_length"] for r in rs])) if rs else None,
            "mean_correct": float(np.mean([r["cot_correct"] for r in rs])) if rs else None,
        }
        for name, rs in pops.items()
    }
    ratio = None
    if pops["intuition"] and pops["prejudice"]:
        ratio = (pop_stats["early_incorrect"]["mean_length"]
                 / pop_stats["early_correct"]["mean_length"])

    # Round-1 reviewer asked for stability: bootstrap CI + τ sweep.
    tau_sweep = {}
    for t in [0.15, 0.20, 0.25, 0.30]:
        tau_sweep[f"tau_{t:.2f}"] = _bootstrap_ratio(rows, t)

    # Partial correlations: does δ predict length after controlling for correctness,
    # κ (uncertainty), and σ (commitment strength)?
    correct = np.array([r["cot_correct"] for r in rows], dtype=float)
    z_cols = np.stack([correct, kappa, sigma], axis=1)
    pc_delta = _partial_spearman(length, delta, z_cols)
    pc_sigma = _partial_spearman(length, sigma, np.stack([correct, kappa], axis=1))

    def _pack(x):
        s, p = x
        return {"stat": float(s), "p": float(p)}

    preregistered = {
        "P1_delta_spearman_sigma_minus_mu":
            (sp_sigma[0] if not np.isnan(sp_sigma[0]) else 0.0)
            - (sp_mu[0] if not np.isnan(sp_mu[0]) else 0.0),
        "P1_pass": (sp_sigma[0] - sp_mu[0]) >= 0.08,
        "P2_delta_R2":
            r2_sigma_delta - r2_sigma,
        "P2_pass": (r2_sigma_delta - r2_sigma) >= 0.05,
        "P3_ratio_prejudice_over_intuition": ratio,
        "P3_pass": ratio is not None and ratio >= 1.8,
    }

    summary = {
        "n": len(rows),
        "cot_correct_rate": float(np.mean([r["cot_correct"] for r in rows])),
        "correlations": {
            "spearman_sigma_length": _pack(sp_sigma),
            "spearman_mu_length":    _pack(sp_mu),
            "spearman_delta_length": _pack(sp_delta),
            "spearman_directmatch_length": _pack(sp_direct),
            "pearson_sigma_length":  _pack(pr_sigma),
            "pearson_mu_length":     _pack(pr_mu),
        },
        "regression_r2": {
            "sigma": r2_sigma,
            "sigma_delta": r2_sigma_delta,
            "sigma_delta_kappa": r2_sigma_delta_kappa,
        },
        "decomposition": {
            "tau": tau,
            "populations": pop_stats,
            "early_incorrect_over_early_correct_length_ratio": ratio,
            "bootstrap_ratio_tau_sweep": tau_sweep,
        },
        "partial_correlations": {
            "spearman_delta_length_given_correct_kappa_sigma": {
                "stat": pc_delta[0], "p": pc_delta[1],
            },
            "spearman_sigma_length_given_correct_kappa": {
                "stat": pc_sigma[0], "p": pc_sigma[1],
            },
        },
        "pre_registered": preregistered,
    }

    dump_json(args.out, summary)
    print(f"wrote summary to {args.out}")
    for k, v in preregistered.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
