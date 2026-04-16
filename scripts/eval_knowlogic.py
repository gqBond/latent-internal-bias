"""Run full-CoT + direct-answer generation on Knowlogic (multi-choice logic)."""
from __future__ import annotations

import argparse

from lib.datasets import load_knowlogic


def main() -> None:
    parent_ap = argparse.ArgumentParser(add_help=False)
    parent_ap.add_argument("--lang", default="en", choices=["en", "zh"])
    args, _rest = parent_ap.parse_known_args()

    from scripts.eval_common import cli_main
    cli_main(
        loader=lambda: load_knowlogic(lang=args.lang),
        out_stem=f"knowlogic_{args.lang}",
    )


if __name__ == "__main__":
    main()
