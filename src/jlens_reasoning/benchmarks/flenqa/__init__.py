"""Validated FLenQA dataset preparation and paired-lens execution."""

from jlens_reasoning.benchmarks.flenqa.accuracy import (
    AccuracyRunConfig,
    AccuracyRunManifest,
    load_accuracy_results,
    run_accuracy,
)
from jlens_reasoning.benchmarks.flenqa.accuracy_analysis import (
    FULL_UNIQUE_PROMPT_COUNTS,
    AccuracyPoint,
    TokenLengthPoint,
    VerdictCountPoint,
    summarize_paper_random,
    summarize_token_lengths,
    summarize_unique_prompts,
    summarize_verdicts,
)
from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaPrompt,
    FlenqaRow,
    SourceProvenance,
    build_prompt_text,
    compute_prompt_id,
    deduplicate,
    normalize_rows,
    verify_count_invariants,
    verify_schema,
)

__all__ = [
    "FULL_UNIQUE_PROMPT_COUNTS",
    "AccuracyRunConfig",
    "AccuracyRunManifest",
    "AccuracyPoint",
    "FlenqaPrompt",
    "FlenqaRow",
    "SourceProvenance",
    "TokenLengthPoint",
    "VerdictCountPoint",
    "build_prompt_text",
    "compute_prompt_id",
    "deduplicate",
    "load_accuracy_results",
    "normalize_rows",
    "run_accuracy",
    "summarize_paper_random",
    "summarize_token_lengths",
    "summarize_unique_prompts",
    "summarize_verdicts",
    "verify_count_invariants",
    "verify_schema",
]
