"""Validated FLenQA dataset preparation and paired-lens execution."""

from jlens_reasoning.benchmarks.flenqa.accuracy import (
    AccuracyRunConfig,
    AccuracyRunManifest,
    load_accuracy_results,
    run_accuracy,
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
    "AccuracyRunConfig",
    "AccuracyRunManifest",
    "FlenqaPrompt",
    "FlenqaRow",
    "SourceProvenance",
    "build_prompt_text",
    "compute_prompt_id",
    "deduplicate",
    "load_accuracy_results",
    "normalize_rows",
    "run_accuracy",
    "verify_count_invariants",
    "verify_schema",
]
