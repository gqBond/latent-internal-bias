"""Prompt builders for reasoning models (R1-Distill style)."""
from __future__ import annotations

from transformers import PreTrainedTokenizerBase


INSTR_MATH = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

INSTR_MCQ = (
    "The following is a multiple-choice question. Think step by step and then "
    "put your final letter choice within \\boxed{}."
)

INSTR_DIRECT_MATH = (
    "Respond with the final numeric answer only, on a single line, no reasoning."
)

INSTR_DIRECT_MCQ = (
    "Respond with only the letter of the correct choice, no reasoning."
)


def _user(q: str, fmt: str, direct: bool) -> str:
    if fmt == "mcq":
        instr = INSTR_DIRECT_MCQ if direct else INSTR_MCQ
    else:
        instr = INSTR_DIRECT_MATH if direct else INSTR_MATH
    return f"{instr}\n\n{q}"


def build_cot_prompt(tokenizer: PreTrainedTokenizerBase, q: str, fmt: str) -> str:
    msgs = [{"role": "user", "content": _user(q, fmt, direct=False)}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def build_pre_think_prompt(tokenizer: PreTrainedTokenizerBase, q: str, fmt: str) -> str:
    """Prompt that ends just after the opening `<think>` tag. The LAST token of this
    prompt is the position p* at which we read the hidden state for LIB."""
    base = build_cot_prompt(tokenizer, q, fmt)
    if "<think>" in base:
        return base
    return base + "<think>\n"


def build_direct_prompt(tokenizer: PreTrainedTokenizerBase, q: str, fmt: str) -> str:
    msgs = [{"role": "user", "content": _user(q, fmt, direct=True)}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
