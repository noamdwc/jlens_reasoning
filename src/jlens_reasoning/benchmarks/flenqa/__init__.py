"""Validated FLenQA dataset preparation and paired-lens execution."""

from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaPrompt,
    FlenqaRow,
    SourceProvenance,
    build_prompt_text,
    compute_prompt_id,
    normalize_rows,
    prepare_prompts,
    verify_count_invariants,
    verify_schema,
)

__all__ = [
    "FlenqaPrompt",
    "FlenqaRow",
    "SourceProvenance",
    "build_prompt_text",
    "compute_prompt_id",
    "normalize_rows",
    "prepare_prompts",
    "verify_count_invariants",
    "verify_schema",
]
