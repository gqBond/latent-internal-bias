"""Run full-CoT + direct-answer generation on a MATH500 subset."""
from __future__ import annotations

import argparse

from lib.datasets import load_math500


def main() -> None:
    parent_ap = argparse.ArgumentParser(add_help=False)
    parent_ap.add_argument("--n", type=int, default=100)
    parent_ap.add_argument("--seed", type=int, default=0)
    args, _rest = parent_ap.parse_known_args()

    from scripts.eval_common import cli_main
    cli_main(
        loader=lambda: load_math500(n=args.n, seed=args.seed),
        out_stem=f"math500_n{args.n}_s{args.seed}",
    )


if __name__ == "__main__":
    main()
