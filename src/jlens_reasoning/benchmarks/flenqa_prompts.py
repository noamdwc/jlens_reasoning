"""Immutable prompt templates from the FLenQA authors."""

from __future__ import annotations

import hashlib


def build_prompt_text(
    *,
    task: str,
    question: str,
    mixin: str,
    rule: object | None,
) -> str:
    """Build the exact task-specific text presented to the model."""
    if task == "Simplified RuleTaker":
        return f'''Answer whether the statement {question} can be derived from the rule and the facts. Answer with either "True" or "False".
Rule: {rule}
Facts: {mixin}
Answer with either "True or "False".
'''
    if task == "PIR":
        return f'''{mixin}
True/False Question: {question}
Answer only True or False.
'''
    if task == "MonoRel":
        return f'''Here are some facts. Answer the exact following question based on the text: {question} Answer the question as it appears exactly.
{mixin}
{question}
Answer only True or False.
'''
    raise ValueError(f"Unknown FLenQA task: {task!r}")


def compute_prompt_id(text: str) -> str:
    """Return the full SHA-256 hex digest of the final prompt text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
