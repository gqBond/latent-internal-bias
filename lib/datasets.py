"""Dataset loaders for AIME24/25, MATH500, Knowlogic, CharCount.

Each returns a list of dicts: {"id": str, "question": str, "answer": str, "format": str}.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict


def _read_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_aime(year: int = 2024, root: str = "data/aime") -> List[Dict]:
    rows = _read_jsonl(Path(root) / f"aime{year}.jsonl")
    return [
        {
            "id": r.get("id", f"aime{year}_{i}"),
            "question": r["problem"],
            "answer": str(r["answer"]).strip(),
            "format": "integer",
        }
        for i, r in enumerate(rows)
    ]


def load_math500(root: str = "data/math500", n: int | None = 100, seed: int = 0) -> List[Dict]:
    import random

    rows = _read_jsonl(Path(root) / "math500.jsonl")
    if n is not None and n < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, n)
    return [
        {
            "id": r.get("id", f"math500_{i}"),
            "question": r["problem"],
            "answer": str(r["answer"]).strip(),
            "format": "integer" if r.get("answer", "").lstrip("-").isdigit() else "free",
        }
        for i, r in enumerate(rows)
    ]


def load_knowlogic(lang: str = "en", root: str = "data/knowlogic") -> List[Dict]:
    rows = _read_jsonl(Path(root) / f"knowlogic_{lang}.jsonl")
    return [
        {
            "id": r.get("id", f"knowlogic_{lang}_{i}"),
            "question": r["question"],
            "answer": r["answer"].strip().upper(),
            "choices": r["choices"],
            "format": "mcq",
        }
        for i, r in enumerate(rows)
    ]


def load_charcount(lang: str = "en", root: str = "data/charcount") -> List[Dict]:
    rows = _read_jsonl(Path(root) / f"charcount_{lang}.jsonl")
    return [
        {
            "id": r.get("id", f"charcount_{lang}_{i}"),
            "question": r["question"],
            "answer": str(r["answer"]).strip(),
            "format": "integer",
        }
        for i, r in enumerate(rows)
    ]
