"""FLenQA problem split checks and matched experimental pairs."""

from __future__ import annotations

from jlens_reasoning.benchmarks.flenqa.dataset import FlenqaPrompt, FlenqaRow


def validate_split(split: dict, rows: tuple[FlenqaRow, ...]) -> dict[int, str]:
    """Check the existing 180/60/60 split without regenerating its assignments."""
    expected = {
        "format_version": 1,
        "seed": 1729,
        "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "stratified_by": ["task", "label"],
    }
    if any(split.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected split version, seed, fractions, or stratification")
    metadata = {}
    for row in rows:
        value = {"task": row.task, "label": row.label}
        if metadata.setdefault(str(row.problem_id), value) != value:
            raise ValueError("Related variants disagree on task or label")
    if len(metadata) != 300 or split.get("problem_metadata") != metadata:
        raise ValueError("Saved problem metadata does not match the full dataset")
    partitions = split.get("problems", {})
    sizes = {"train": 180, "validation": 60, "test": 60}
    if {name: len(ids) for name, ids in partitions.items()} != sizes:
        raise ValueError("Expected a 180/60/60 train/validation/test split")
    mapping = {}
    for partition, problem_ids in partitions.items():
        for problem_id in problem_ids:
            if problem_id in mapping:
                raise ValueError("Duplicate problem ID or split leakage")
            mapping[problem_id] = partition
    if set(map(str, mapping)) != set(metadata):
        raise ValueError("Split IDs do not cover exactly the current dataset")
    return mapping


def matched_prompt_pairs(
    prompts: tuple[FlenqaPrompt, ...], *, short_ctx: int, long_ctx: int
) -> list[dict]:
    """Match every provenance condition, rejecting ambiguous or changed tasks.

    Exact prompt deduplication can combine several conditions into one prompt.
    Retain all conditions here; downstream summaries must group by problem or
    deduplicate endpoint IDs before treating pairs as independent observations.
    Conditions missing either length are omitted, never cross-matched.
    """
    if short_ctx >= long_ctx:
        raise ValueError("Expected short_ctx < long_ctx")
    grouped: dict[tuple, dict[int, FlenqaPrompt]] = {}
    invariants = {}
    for prompt in prompts:
        # prepare_prompts already guarantees one context length per prompt.
        ctx = prompt.provenance[0].ctx_size
        if ctx not in (short_ctx, long_ctx):
            continue
        invariant = (
            prompt.task,
            prompt.label,
            prompt.question,
            prompt.key_texts,
            prompt.rule,
        )
        if invariants.setdefault(prompt.problem_id, invariant) != invariant:
            raise ValueError(
                "Matched problem variants disagree on task content or label"
            )
        for source in prompt.provenance:
            key = (
                prompt.problem_id,
                prompt.task,
                source.padding_type,
                source.dispersion,
            )
            by_length = grouped.setdefault(key, {})
            previous = by_length.setdefault(ctx, prompt)
            if previous.prompt_id != prompt.prompt_id:
                raise ValueError("Ambiguous prompts for one problem/condition/length")
    return [
        {
            "problem_id": key[0],
            "task": key[1],
            "padding_type": key[2],
            "dispersion": key[3],
            "label": int(pair[short_ctx].label),
            "short_id": pair[short_ctx].prompt_id,
            "long_id": pair[long_ctx].prompt_id,
        }
        for key, pair in sorted(grouped.items())
        if short_ctx in pair and long_ctx in pair
    ]
