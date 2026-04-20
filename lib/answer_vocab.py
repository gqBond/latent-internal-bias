"""Build the per-problem answer vocabulary A(q) used by the LIB metric."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase


MCQ_LETTERS = ["A", "B", "C", "D", "E"]
INT_DIGITS = list("0123456789")


@dataclass
class AnswerVocab:
    """Answer vocabulary for a single problem.

    token_ids: vocabulary indices to restrict the LIB softmax to.
    labels:    human-readable strings aligned to token_ids.
    """

    token_ids: "torch.LongTensor"
    labels: List[str]


@dataclass
class FullAnswerVocab:
    """Multi-token answer vocabulary — scored by teacher-forced sequence log-prob
    under the lens. Added in Round 2 per reviewer ask: first-digit scoring
    ("argmax=9 everywhere" pathology) was conflating label prior with evidence.

    token_id_lists: list of token-id sequences, one per candidate.
    labels:         candidate answer strings aligned to token_id_lists.
    """

    token_id_lists: List[List[int]]
    labels: List[str]


def _first_token_id(tok: "PreTrainedTokenizerBase", s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    assert len(ids) >= 1, f"empty encoding for {s!r}"
    return ids[0]


def mcq_vocab(tok: "PreTrainedTokenizerBase", num_choices: int = 4) -> AnswerVocab:
    import torch
    letters = MCQ_LETTERS[:num_choices]
    ids = [_first_token_id(tok, " " + l) for l in letters]
    return AnswerVocab(torch.tensor(ids, dtype=torch.long), letters)


def integer_vocab(
    tok: "PreTrainedTokenizerBase",
    direct_answers: Sequence[str],
    max_candidates: int = 10,
) -> AnswerVocab:
    """For open integer answers (AIME, MATH500). Candidates = first-digit tokens of the
    sampled direct answers, deduped by first character; fall back to all digits."""
    firsts: List[str] = []
    for da in direct_answers:
        m = re.search(r"-?\d", da)
        if m:
            firsts.append(m.group(0))
    uniq: List[str] = []
    for f in firsts:
        if f not in uniq:
            uniq.append(f)
    if len(uniq) < 3:
        uniq = INT_DIGITS
    uniq = uniq[:max_candidates]
    import torch
    ids = [_first_token_id(tok, u) for u in uniq]
    return AnswerVocab(torch.tensor(ids, dtype=torch.long), uniq)


def full_answer_vocab(
    tok: "PreTrainedTokenizerBase",
    candidates: Sequence[str],
    *,
    dedup: bool = True,
    max_candidates: int = 64,
) -> FullAnswerVocab:
    """Build a full-string candidate vocab. Each candidate is tokenized as a
    sequence (variable length). Callers score each candidate by teacher-forced
    lens sequence log-prob; see `lib.lens.score_full_answers`.

    No-leading-space and leading-space forms are both possible depending on the
    tokenizer — we pick the one whose first token is most common across
    candidates to keep the first-token position consistent."""
    seen: List[str] = []
    for c in candidates:
        if not c:
            continue
        s = str(c).strip()
        if not s:
            continue
        if dedup and s in seen:
            continue
        seen.append(s)
    seen = seen[:max_candidates]

    token_lists: List[List[int]] = []
    labels: List[str] = []
    for s in seen:
        ids = tok.encode(" " + s, add_special_tokens=False)
        if not ids:
            ids = tok.encode(s, add_special_tokens=False)
        if not ids:
            continue
        token_lists.append(list(ids))
        labels.append(s)
    return FullAnswerVocab(token_id_lists=token_lists, labels=labels)


def canonicalize_integer(text: str) -> str | None:
    """Pull a canonical integer answer from either a direct-answer or boxed CoT output."""
    m = re.search(r"\\boxed\{(-?\d+)\}", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|Answer)[^0-9\-]{0,20}(-?\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"-?\d+", text)
    return m.group(0) if m else None


def canonicalize_mcq(text: str) -> str | None:
    m = re.search(r"(?:answer|Answer|choice|Choice)[^A-E]{0,20}([A-E])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-E])\b", text)
    return m.group(1) if m else None


def canonicalize_to_vocab(answer: str | None, labels: list[str]) -> str:
    """Project a full answer string onto the single-token vocabulary `labels`.

    The LIB vocab is at first-token granularity (one digit for integer, one letter
    for MCQ). To check `argmax π_L == answer`, we have to compare at the same
    granularity — otherwise a multi-digit answer like "143" can never equal a
    digit label like "1". Returns "" when no overlap.
    """
    if not answer:
        return ""
    s = str(answer).strip().lstrip("-").lstrip(" ")
    for c in s:
        if c in labels:
            return c
    return ""
