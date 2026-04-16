"""Run full-CoT + direct-answer generation on AIME.

Usage:
    python -m scripts.eval_aime --cfg configs/r1_qwen_7b.yaml --year 2024
"""
from __future__ import annotations

import argparse

from lib.datasets import load_aime


def main() -> None:
    parent_ap = argparse.ArgumentParser(add_help=False)
    parent_ap.add_argument("--year", type=int, default=2024, choices=[2024, 2025])
    args, _rest = parent_ap.parse_known_args()

    from scripts.eval_common import cli_main
    cli_main(loader=lambda: load_aime(year=args.year), out_stem=f"aime{args.year}")


if __name__ == "__main__":
    main()
