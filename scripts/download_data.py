"""Download and normalize AIME24/25, MATH500, Knowlogic, CharCount.

All datasets are written to `data/<name>/<name>.jsonl` in the schema expected by
`lib/datasets.py`:
    - AIME / MATH500: {"id", "problem", "answer"}
    - Knowlogic:      {"id", "question", "answer", "choices"}
    - CharCount:      {"id", "question", "answer"}

Usage:
    python -m scripts.download_data --all
    python -m scripts.download_data --datasets aime24 math500
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset


DATA_ROOT = Path("data")


def _dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>4} rows → {path}")


def _get(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def fetch_aime(year: int) -> None:
    hf_id = {2024: "Maxwell-Jia/AIME_2024", 2025: "opencompass/AIME2025"}[year]
    print(f"[aime{year}] loading {hf_id}")
    ds = load_dataset(hf_id, split="train" if year == 2024 else "test")
    rows = []
    for i, ex in enumerate(ds):
        prob = _get(ex, "Problem", "problem", "question")
        ans = _get(ex, "Answer", "answer")
        if prob is None or ans is None:
            continue
        rows.append({"id": f"aime{year}_{i:03d}", "problem": prob, "answer": str(ans).strip()})
    _dump(DATA_ROOT / "aime" / f"aime{year}.jsonl", rows)


def fetch_math500() -> None:
    print("[math500] loading HuggingFaceH4/MATH-500")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": _get(ex, "unique_id", "id", default=f"math500_{i:03d}"),
            "problem": ex["problem"],
            "answer": str(ex["answer"]).strip(),
            "subject": ex.get("subject"),
            "level": ex.get("level"),
        })
    _dump(DATA_ROOT / "math500" / "math500.jsonl", rows)


def fetch_njunlp_repo() -> Path:
    """Clone the reference paper's repo (for Knowlogic / CharCount raw data)."""
    target = DATA_ROOT / "_upstream" / "LongCoT-Internal-Bias"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[upstream] cloning NJUNLP/LongCoT-Internal-Bias → {target}")
    subprocess.check_call([
        "git", "clone", "--depth=1",
        "https://github.com/NJUNLP/LongCoT-Internal-Bias.git",
        str(target),
    ])
    return target


def fetch_knowlogic(lang: str) -> None:
    upstream = fetch_njunlp_repo()
    candidates = list((upstream / "Knowlogic").rglob(f"*{lang}*.json*")) + \
                 list((upstream / "Knowlogic").rglob("*.json*"))
    candidates = [p for p in candidates if p.is_file() and "result" not in p.name.lower()]
    if not candidates:
        print(f"[knowlogic_{lang}] ⚠ no data file found under {upstream/'Knowlogic'}; skip")
        return
    src = candidates[0]
    print(f"[knowlogic_{lang}] reading {src}")
    with open(src) as f:
        raw = [json.loads(l) for l in f if l.strip()] if src.suffix == ".jsonl" \
              else json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("data") or list(raw.values())[0]
    rows = []
    for i, ex in enumerate(raw):
        q = _get(ex, "question", "problem")
        a = _get(ex, "answer", "label")
        choices = _get(ex, "choices", "options", default=["A", "B", "C", "D"])
        if q is None or a is None:
            continue
        rows.append({
            "id": f"knowlogic_{lang}_{i:03d}",
            "question": q,
            "answer": str(a).strip().upper()[:1],
            "choices": choices,
        })
    _dump(DATA_ROOT / "knowlogic" / f"knowlogic_{lang}.jsonl", rows)


def fetch_charcount(lang: str) -> None:
    upstream = fetch_njunlp_repo()
    candidates = [p for p in (upstream / "CharCount").rglob(f"*{lang}*.json*") if p.is_file()]
    if not candidates:
        print(f"[charcount_{lang}] ⚠ no data file found; skip")
        return
    src = candidates[0]
    print(f"[charcount_{lang}] reading {src}")
    with open(src) as f:
        raw = [json.loads(l) for l in f if l.strip()] if src.suffix == ".jsonl" \
              else json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("data") or list(raw.values())[0]
    rows = []
    for i, ex in enumerate(raw):
        q = _get(ex, "question", "problem", "text")
        a = _get(ex, "answer", "label")
        if q is None or a is None:
            continue
        rows.append({"id": f"charcount_{lang}_{i:03d}", "question": q, "answer": str(a).strip()})
    _dump(DATA_ROOT / "charcount" / f"charcount_{lang}.jsonl", rows)


REGISTRY = {
    "aime24": lambda: fetch_aime(2024),
    "aime25": lambda: fetch_aime(2025),
    "math500": fetch_math500,
    "knowlogic_en": lambda: fetch_knowlogic("en"),
    "knowlogic_zh": lambda: fetch_knowlogic("zh"),
    "charcount_en": lambda: fetch_charcount("en"),
    "charcount_zh": lambda: fetch_charcount("zh"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=list(REGISTRY) + ["all"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all or (args.datasets and "all" in args.datasets):
        targets = list(REGISTRY)
    elif args.datasets:
        targets = args.datasets
    else:
        targets = ["aime24", "math500"]  # minimum for Pilot P0

    for name in targets:
        try:
            REGISTRY[name]()
        except Exception as e:
            print(f"[{name}] ✗ {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
