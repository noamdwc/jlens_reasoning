"""Testable library run loop for the FLenQA length-drift readout."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import pyarrow as pa
import torch

from experiments.flenqa_length_drift.anchors import (
    Anchor,
    prompt_seed,
    select_anchors,
    select_summary_positions,
)
from experiments.flenqa_length_drift.bridges import (
    bridge_candidate_surfaces,
    extract_bridge,
)
from experiments.flenqa_length_drift.constants import MAX_SEQ_LEN, TOP_K
from experiments.flenqa_length_drift.readout import (
    ReadoutReduction,
    TokenCandidate,
    reduce_readout,
)
from experiments.flenqa_length_drift.scoring import score_binary_answer
from experiments.flenqa_length_drift.tables import (
    REQUIRED_SHARD_TABLES,
    empty_batch,
    record_batch,
)
from jlens_reasoning.benchmarks.flenqa import FlenqaRow
from jlens_reasoning.benchmarks.flenqa_conditions import (
    build_padding_positions,
    derive_conditions,
)
from jlens_reasoning.benchmarks.flenqa_preparation import PreparedPrompt
from jlens_reasoning.benchmarks.flenqa_prompts import compute_prompt_id


@dataclass(frozen=True, slots=True)
class LensPassResult:
    logits_by_layer: Mapping[int, torch.Tensor]
    model_logits: torch.Tensor
    input_ids: Any


class LensRunner(Protocol):
    def run(
        self,
        prompt: str,
        *,
        positions: Sequence[int],
        max_seq_len: int,
    ) -> LensPassResult: ...


def _input_ids(value: Any) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, Sequence)
        and value
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        if len(value) != 1:
            raise RuntimeError("Lens input IDs must contain exactly one batch")
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError("Lens input IDs must be a sequence")
    try:
        return tuple(int(token_id) for token_id in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Lens input IDs must contain integers") from exc


def _bridge(prepared: PreparedPrompt) -> str | None:
    surfaces = {
        diagnostic.surface
        for diagnostic in prepared.diagnostics
        if diagnostic.kind == "bridge"
    }
    if len(surfaces) == 1:
        return next(iter(surfaces))
    return extract_bridge(prepared.prompt)


def _candidates(tokenizer: Any, bridge: str | None) -> tuple[TokenCandidate, ...]:
    if bridge is None:
        return ()
    candidates: list[TokenCandidate] = []
    seen_ids: set[int] = set()
    for surface in bridge_candidate_surfaces(bridge):
        token_ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(token_ids) == 1:
            token_id = int(token_ids[0])
            if token_id not in seen_ids:
                seen_ids.add(token_id)
                candidates.append(TokenCandidate(surface=surface, token_id=token_id))
    return tuple(candidates)


def _provenance(
    prepared: PreparedPrompt,
    anchors: Sequence[Anchor],
    padding_positions: Sequence[int],
    positions: Sequence[int],
) -> dict[int, str]:
    anchor_labels: dict[int, list[str]] = {}
    for anchor in anchors:
        anchor_labels.setdefault(anchor.position, []).append(anchor.label)
    fact_positions = {
        position
        for span in prepared.fact_token_spans
        for position in range(span.start, span.end)
    }
    padding = set(padding_positions)
    final = len(prepared.input_ids) - 1
    result: dict[int, str] = {}
    for position in positions:
        if position in anchor_labels:
            result[position] = "|".join(anchor_labels[position])
        elif position in fact_positions:
            result[position] = "fact"
        elif position == final:
            result[position] = "final_prompt"
        elif position in padding:
            result[position] = "padding"
        else:
            result[position] = "selected"
    return result


def _prompt_batch(
    prepared: PreparedPrompt,
    *,
    bridge: str | None,
) -> pa.RecordBatch:
    prompt = prepared.prompt
    conditions = derive_conditions(prepared)
    return record_batch(
        "prompts",
        {
            "prompt_id": [prompt.prompt_id],
            "canonical_index": [prompt.canonical_index],
            "problem_id": [prompt.problem_id],
            "task": [prompt.task],
            "label": [prompt.label],
            "final_text_hash": [compute_prompt_id(prompt.text)],
            "token_hash": [prepared.token_signature],
            "n_tokens": [len(prepared.input_ids)],
            "ctx_size_declared": [prompt.ctx_size_declared],
            "padding_type_declared": [list(prompt.padding_type_declared)],
            "dispersion_declared": [list(prompt.dispersion_declared)],
            "padding_type_effective": [conditions.padding_type_effective],
            "dispersion_effective": [conditions.dispersion_effective],
            "frac_padding_before": [conditions.frac_padding_before],
            "frac_padding_between": [conditions.frac_padding_between],
            "frac_padding_after": [conditions.frac_padding_after],
            "n_padding_tokens": [conditions.n_padding_tokens],
            "bridge": [bridge],
        },
    )


def _source_rows_batch(
    prepared: PreparedPrompt,
    source_rows: Mapping[int, FlenqaRow],
) -> pa.RecordBatch:
    try:
        rows = tuple(source_rows[row_id] for row_id in prepared.prompt.source_row_ids)
    except KeyError as exc:
        raise ValueError(f"Missing source-row provenance for ID {exc.args[0]}") from exc
    return record_batch(
        "source_rows",
        {
            "prompt_id": [prepared.prompt.prompt_id] * len(rows),
            "source_row_id": [row.source_row_id for row in rows],
            "problem_id": [row.problem_id for row in rows],
            "sample_id": [row.sample_id for row in rows],
            "task": [row.task for row in rows],
            "label": [row.label for row in rows],
            "ctx_size_declared": [row.ctx_size_declared for row in rows],
            "padding_type_declared": [row.padding_type_declared for row in rows],
            "dispersion_declared": [row.dispersion_declared for row in rows],
        },
    )


def _spans_batch(prepared: PreparedPrompt) -> pa.RecordBatch:
    diagnostics = prepared.diagnostics
    return record_batch(
        "spans",
        {
            "prompt_id": [prepared.prompt.prompt_id] * len(diagnostics),
            "span_kind": [diagnostic.kind for diagnostic in diagnostics],
            "ordinal": [diagnostic.ordinal for diagnostic in diagnostics],
            "fact_ordinal": [diagnostic.fact_ordinal for diagnostic in diagnostics],
            "surface": [diagnostic.surface for diagnostic in diagnostics],
            "span_status": [diagnostic.status.value for diagnostic in diagnostics],
            "span_match_count": [
                diagnostic.match_count for diagnostic in diagnostics
            ],
            "char_start": [diagnostic.char_start for diagnostic in diagnostics],
            "char_end": [diagnostic.char_end for diagnostic in diagnostics],
            "token_start": [diagnostic.token_start for diagnostic in diagnostics],
            "token_end": [diagnostic.token_end for diagnostic in diagnostics],
        },
    )


def _anchors_batch(prompt_id: str, anchors: Sequence[Anchor]) -> pa.RecordBatch:
    return record_batch(
        "anchors",
        {
            "prompt_id": [prompt_id] * len(anchors),
            "anchor_label": [anchor.label for anchor in anchors],
            "position": [anchor.position for anchor in anchors],
        },
    )


def _topk_batch(reductions: Sequence[ReadoutReduction]) -> pa.RecordBatch:
    rows = tuple(value for reduction in reductions for value in reduction.topk)
    if not rows:
        return empty_batch("topk")
    return record_batch(
        "topk",
        {
            "prompt_id": [row.prompt_id for row in rows],
            "layer": [row.layer for row in rows],
            "position": [row.position for row in rows],
            "anchor_label": [row.anchor_label for row in rows],
            "lens_kind": [row.lens_kind for row in rows],
            "rank": [row.rank for row in rows],
            "token_id": [row.token_id for row in rows],
            "logit": [row.logit for row in rows],
        },
    )


def _bridge_batch(reductions: Sequence[ReadoutReduction]) -> pa.RecordBatch:
    rows = tuple(value for reduction in reductions for value in reduction.targets)
    if not rows:
        return empty_batch("bridge")
    return record_batch(
        "bridge",
        {
            "prompt_id": [row.prompt_id for row in rows],
            "layer": [row.layer for row in rows],
            "position": [row.position for row in rows],
            "anchor_label": [row.anchor_label for row in rows],
            "lens_kind": [row.lens_kind for row in rows],
            "surface": [row.surface for row in rows],
            "token_id": [row.token_id for row in rows],
            "rank": [row.rank for row in rows],
            "logit": [row.logit for row in rows],
        },
    )


def _summary_batch(
    reductions: Sequence[ReadoutReduction],
    provenance: Mapping[int, str],
) -> pa.RecordBatch:
    rows = tuple(value for reduction in reductions for value in reduction.summary)
    if not rows:
        return empty_batch("summary")
    return record_batch(
        "summary",
        {
            "prompt_id": [row.prompt_id for row in rows],
            "layer": [row.layer for row in rows],
            "position": [row.position for row in rows],
            "provenance": [provenance[row.position] for row in rows],
            "lens_kind": [row.lens_kind for row in rows],
            "entropy": [row.entropy for row in rows],
            "max_logit": [row.max_logit for row in rows],
            "top1_token_id": [row.top1_token_id for row in rows],
        },
    )


def _scoring_batch(
    prepared: PreparedPrompt,
    *,
    tokenizer: Any,
    model_logits: torch.Tensor,
    positions: Sequence[int],
    generated_text: str | None,
) -> pa.RecordBatch:
    final_position = len(prepared.input_ids) - 1
    try:
        final_index = tuple(positions).index(final_position)
    except ValueError as exc:
        raise RuntimeError("Final prompt position was not selected") from exc
    score = score_binary_answer(
        model_logits[final_index],
        tokenizer=tokenizer,
        label=prepared.prompt.label,
        generated_text=generated_text,
    )
    return record_batch(
        "scoring",
        {
            "prompt_id": [prepared.prompt.prompt_id],
            "logit_true": [score.logit_true],
            "logit_false": [score.logit_false],
            "rank_true": [score.rank_true],
            "rank_false": [score.rank_false],
            "predicted": [score.predicted],
            "correct": [score.correct],
            "generated_text": [score.generated_text],
            "extracted": [score.extracted],
            "generated_correct": [score.generated_correct],
            "agrees": [score.agrees],
        },
    )


def run_prompt(
    prepared: PreparedPrompt,
    *,
    source_rows: Mapping[int, FlenqaRow],
    jacobian_runner: LensRunner,
    logit_runner: LensRunner,
    tokenizer: Any,
    generate: Callable[[str], str] | None = None,
) -> tuple[tuple[str, pa.RecordBatch], ...]:
    """Run and reduce one prompt into one typed batch per required table."""
    padding_positions = build_padding_positions(prepared)
    seed = prompt_seed(prepared.prompt.prompt_id)
    anchors = select_anchors(
        prepared,
        padding_positions=padding_positions,
        seed=seed,
    )
    positions = select_summary_positions(
        prepared,
        anchors=anchors,
        padding_positions=padding_positions,
        seed=seed,
    )
    jacobian = jacobian_runner.run(
        prepared.prompt.text,
        positions=positions,
        max_seq_len=MAX_SEQ_LEN,
    )
    logit = logit_runner.run(
        prepared.prompt.text,
        positions=positions,
        max_seq_len=MAX_SEQ_LEN,
    )
    prepared_ids = prepared.input_ids
    jacobian_ids = _input_ids(jacobian.input_ids)
    logit_ids = _input_ids(logit.input_ids)
    if jacobian_ids != prepared_ids or logit_ids != prepared_ids:
        raise RuntimeError("Prepared and lens token IDs differ")
    if jacobian_ids != logit_ids:
        raise RuntimeError("Jacobian and logit-lens token IDs differ")
    if not torch.equal(jacobian.model_logits, logit.model_logits):
        raise RuntimeError("Jacobian and logit-lens model logits differ")
    if (
        jacobian.model_logits.ndim != 2
        or jacobian.model_logits.shape[0] != len(positions)
    ):
        raise RuntimeError("Lens model logits rows must match selected positions")

    bridge = _bridge(prepared)
    candidates = _candidates(tokenizer, bridge)
    jacobian_reduction = reduce_readout(
        prompt_id=prepared.prompt.prompt_id,
        lens_kind="jacobian",
        logits_by_layer=jacobian.logits_by_layer,
        positions=positions,
        anchors=anchors,
        candidates=candidates,
        top_k=TOP_K,
    )
    logit_reduction = reduce_readout(
        prompt_id=prepared.prompt.prompt_id,
        lens_kind="logit",
        logits_by_layer=logit.logits_by_layer,
        positions=positions,
        anchors=anchors,
        candidates=candidates,
        top_k=TOP_K,
    )
    reductions = (jacobian_reduction, logit_reduction)
    provenance = _provenance(
        prepared,
        anchors,
        padding_positions,
        positions,
    )
    generated_text = None if generate is None else generate(prepared.prompt.text)
    batches = {
        "prompts": _prompt_batch(prepared, bridge=bridge),
        "source_rows": _source_rows_batch(prepared, source_rows),
        "spans": _spans_batch(prepared),
        "anchors": _anchors_batch(prepared.prompt.prompt_id, anchors),
        "topk": _topk_batch(reductions),
        "bridge": _bridge_batch(reductions),
        "summary": _summary_batch(reductions, provenance),
        "scoring": _scoring_batch(
            prepared,
            tokenizer=tokenizer,
            model_logits=jacobian.model_logits,
            positions=positions,
            generated_text=generated_text,
        ),
    }
    return tuple((table, batches[table]) for table in REQUIRED_SHARD_TABLES)
