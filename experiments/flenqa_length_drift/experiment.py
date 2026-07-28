"""Testable library run loop for the FLenQA length-drift readout."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
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
from experiments.flenqa_length_drift.constants import (
    ANCHOR_BUDGET,
    MAX_SEQ_LEN,
    SHARD_SIZE,
    TOP_K,
)
from experiments.flenqa_length_drift.gate import BridgeGateResult, bridge_gate
from experiments.flenqa_length_drift.readout import (
    ReadoutReduction,
    TokenCandidate,
    reduce_readout,
)
from experiments.flenqa_length_drift.scoring import score_binary_answer
from experiments.flenqa_length_drift.tables import (
    GLOBAL_SCHEMAS,
    REQUIRED_SHARD_TABLES,
    TABLE_SCHEMAS,
    empty_batch,
    record_batch,
)
from jlens_reasoning.benchmarks.flenqa import FlenqaRow, deduplicate
from jlens_reasoning.benchmarks.flenqa_conditions import (
    assert_unpadded_prompt_count,
    build_padding_positions,
    derive_conditions,
)
from jlens_reasoning.benchmarks.flenqa_preparation import (
    PreparedPrompt,
    prepare_prompt,
)
from jlens_reasoning.benchmarks.flenqa_prompts import compute_prompt_id
from jlens_reasoning.experiments_utils.storage import (
    ShardManifest,
    ShardWriter,
    is_shard_complete,
    read_shard_manifest,
    validate_shard_manifest,
)


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


@dataclass(frozen=True, slots=True)
class LensRunners:
    jacobian: LensRunner
    logit: LensRunner


@dataclass(frozen=True, slots=True)
class ApplyLensRunner:
    """Adapter around ``JacobianLens.apply`` for one lens mode."""

    lens: Any
    model: Any
    use_jacobian: bool
    layers: tuple[int, ...] | None = None

    def run(
        self,
        prompt: str,
        *,
        positions: Sequence[int],
        max_seq_len: int,
    ) -> LensPassResult:
        logits, model_logits, input_ids = self.lens.apply(
            self.model,
            prompt,
            layers=self.layers,
            positions=tuple(positions),
            max_seq_len=max_seq_len,
            use_jacobian=self.use_jacobian,
        )
        return LensPassResult(
            logits_by_layer=logits,
            model_logits=model_logits,
            input_ids=input_ids,
        )


@dataclass(frozen=True, slots=True)
class PromptShard:
    shard_id: int
    prompt_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RunConfig:
    model_name: str
    lens_revision: str
    tokenizer_name: str
    template_hash: str
    code_revision: str
    top_k: int = TOP_K
    anchor_budget: int = ANCHOR_BUDGET
    schema_version: str = "1"
    shard_size: int = SHARD_SIZE
    expected_bridge_problems: int = 200
    expected_unpadded_prompts: int | None = 300
    max_seq_len: int = MAX_SEQ_LEN
    bridge_rule: str = "task-specific-shared-entity-v1"
    dedup_rule: str = "sha256-final-text-first-occurrence-v1"


@dataclass(frozen=True, slots=True)
class RunManifest:
    config_hash: str
    prompt_ids: tuple[str, ...]
    shard_ids: tuple[int, ...]
    bridge_gate: BridgeGateResult


def plan_shards(
    prepared_prompts: Sequence[PreparedPrompt],
    *,
    shard_size: int,
) -> tuple[PromptShard, ...]:
    """Assign immutable shards from the complete canonical prompt sequence."""
    if type(shard_size) is not int or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    canonical_indices = tuple(
        prepared.prompt.canonical_index for prepared in prepared_prompts
    )
    if len(set(canonical_indices)) != len(canonical_indices):
        raise ValueError("Prompt canonical indices must be unique")
    if canonical_indices != tuple(sorted(canonical_indices)):
        raise ValueError("Prepared prompts must be in canonical order")
    return tuple(
        PromptShard(
            shard_id=start // shard_size,
            prompt_indices=canonical_indices[start : start + shard_size],
        )
        for start in range(0, len(canonical_indices), shard_size)
    )


def _config_hash(config: RunConfig) -> str:
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_global_batch(path: Path, batch: pa.RecordBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    if temporary.exists():
        temporary.unlink()
    pq.write_table(
        pa.Table.from_batches([batch], schema=batch.schema),
        temporary,
        compression="zstd",
    )
    os.replace(temporary, path)


def _read_global_table(path: Path, *, table: str) -> pa.Table:
    try:
        result = pq.read_table(path)
    except (OSError, pa.ArrowException) as exc:
        raise RuntimeError(f"Cannot read global {table} table") from exc
    if result.schema != GLOBAL_SCHEMAS[table]:
        raise RuntimeError(f"Global {table} table has the wrong schema")
    return result


def _vocab_batch(tokenizer: Any) -> pa.RecordBatch:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if not isinstance(vocab, Mapping):
            raise ValueError("tokenizer.get_vocab() must return a mapping")
        by_id: dict[int, list[str]] = {}
        for token_text, raw_token_id in vocab.items():
            token_id = int(raw_token_id)
            by_id.setdefault(token_id, []).append(str(token_text))
    else:
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if type(vocab_size) is not int or vocab_size < 0:
            raise ValueError("tokenizer must expose get_vocab() or vocab_size")
        by_id = {token_id: [] for token_id in range(vocab_size)}

    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    token_ids = sorted(by_id)
    token_texts: list[str] = []
    for token_id in token_ids:
        converted = convert(token_id) if callable(convert) else None
        token_texts.append(
            str(converted)
            if converted is not None
            else (sorted(by_id[token_id])[0] if by_id[token_id] else str(token_id))
        )
    return record_batch(
        "vocab",
        {
            "token_id": token_ids,
            "token_text": token_texts,
        },
    )


def _run_meta_batch(config: RunConfig, config_hash: str) -> pa.RecordBatch:
    return record_batch(
        "run_meta",
        {
            "config_hash": [config_hash],
            "model_name": [config.model_name],
            "lens_revision": [config.lens_revision],
            "tokenizer_name": [config.tokenizer_name],
            "template_hash": [config.template_hash],
            "top_k": [config.top_k],
            "anchor_budget": [config.anchor_budget],
            "schema_version": [config.schema_version],
            "code_revision": [config.code_revision],
        },
    )


def _manifest_payload(manifest: RunManifest, root: Path) -> dict[str, Any]:
    return {
        "config_hash": manifest.config_hash,
        "prompt_ids": list(manifest.prompt_ids),
        "shard_ids": list(manifest.shard_ids),
        "bridge_gate": asdict(manifest.bridge_gate),
        "globals": {
            table: {
                "path": f"{table}.parquet",
                "sha256": _file_sha256(root / f"{table}.parquet"),
            }
            for table in GLOBAL_SCHEMAS
        },
    }


def _read_run_manifest(path: Path) -> RunManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        gate = payload["bridge_gate"]
        return RunManifest(
            config_hash=str(payload["config_hash"]),
            prompt_ids=tuple(str(value) for value in payload["prompt_ids"]),
            shard_ids=tuple(int(value) for value in payload["shard_ids"]),
            bridge_gate=BridgeGateResult(
                applicable=int(gate["applicable"]),
                resolved=int(gate["resolved"]),
            ),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Run completion manifest is invalid") from exc


def _validate_run(
    root: Path,
    manifest: RunManifest,
    *,
    expected: RunManifest,
) -> RunManifest:
    if manifest != expected:
        raise RuntimeError("Run completion manifest does not match current plan")
    path = root / "run-manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        globals_payload = payload["globals"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Run completion manifest is invalid") from exc
    for table in GLOBAL_SCHEMAS:
        global_path = root / f"{table}.parquet"
        _read_global_table(global_path, table=table)
        try:
            recorded = globals_payload[table]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Run manifest is missing a global table") from exc
        if recorded.get("path") != f"{table}.parquet" or recorded.get(
            "sha256"
        ) != _file_sha256(global_path):
            raise RuntimeError(f"Run manifest global {table} checksum is invalid")
    for shard_id in manifest.shard_ids:
        if not is_shard_complete(
            root,
            shard_id=shard_id,
            schemas=TABLE_SCHEMAS,
            required_tables=REQUIRED_SHARD_TABLES,
        ):
            raise RuntimeError(f"Run shard {shard_id} is incomplete")
    actual_prompt_ids = tuple(
        prompt_id
        for shard_id in manifest.shard_ids
        for prompt_id in read_shard_manifest(root, shard_id=shard_id).prompt_ids
    )
    if actual_prompt_ids != manifest.prompt_ids:
        raise RuntimeError("Run shard prompt membership does not match manifest")
    return manifest


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
            "span_match_count": [diagnostic.match_count for diagnostic in diagnostics],
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
    if jacobian.model_logits.ndim != 2 or jacobian.model_logits.shape[0] != len(
        positions
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


def run_shard(
    shard: PromptShard,
    prepared_prompts: Sequence[PreparedPrompt],
    *,
    output_dir: Path,
    source_rows: Mapping[int, FlenqaRow],
    runners: LensRunners,
    tokenizer: Any,
    generate: Callable[[str], str] | None = None,
    run_prompt_fn: Callable[..., tuple[tuple[str, pa.RecordBatch], ...]] = run_prompt,
    after_append: Callable[[str], None] | None = None,
) -> ShardManifest:
    """Run one immutable shard, committing its completion manifest last."""
    by_index = {
        prepared.prompt.canonical_index: prepared for prepared in prepared_prompts
    }
    if len(by_index) != len(prepared_prompts):
        raise ValueError("Prepared prompt canonical indices must be unique")
    try:
        selected = tuple(by_index[index] for index in shard.prompt_indices)
    except KeyError as exc:
        raise ValueError(
            f"Shard references unknown canonical index {exc.args[0]}"
        ) from exc
    prompt_ids = tuple(prepared.prompt.prompt_id for prepared in selected)
    root = Path(output_dir)
    if is_shard_complete(
        root,
        shard_id=shard.shard_id,
        schemas=TABLE_SCHEMAS,
        required_tables=REQUIRED_SHARD_TABLES,
    ):
        manifest = read_shard_manifest(root, shard_id=shard.shard_id)
        if manifest.prompt_ids != prompt_ids:
            raise RuntimeError("Completed shard prompt membership does not match plan")
        return validate_shard_manifest(
            root,
            manifest,
            schemas=TABLE_SCHEMAS,
            required_tables=REQUIRED_SHARD_TABLES,
        )

    writer = ShardWriter(
        root,
        shard_id=shard.shard_id,
        schemas=TABLE_SCHEMAS,
        required_tables=REQUIRED_SHARD_TABLES,
        prompt_ids=prompt_ids,
    )
    try:
        for prepared in selected:
            batches = run_prompt_fn(
                prepared,
                source_rows=source_rows,
                jacobian_runner=runners.jacobian,
                logit_runner=runners.logit,
                tokenizer=tokenizer,
                generate=generate,
            )
            if tuple(table for table, _batch in batches) != REQUIRED_SHARD_TABLES:
                raise RuntimeError("run_prompt did not return every required table")
            for table, batch in batches:
                writer.append(table, batch)
                if after_append is not None:
                    after_append(table)
        return writer.commit()
    except BaseException:
        writer.abort()
        raise


def run_experiment(
    rows: Sequence[FlenqaRow],
    *,
    output_dir: Path,
    tokenizer: Any,
    runners: LensRunners,
    config: RunConfig,
    generate: Callable[[str], str] | None = None,
) -> RunManifest:
    """Run or safely resume the complete FLenQA experiment."""
    if (
        config.top_k != TOP_K
        or config.anchor_budget != ANCHOR_BUDGET
        or config.max_seq_len != MAX_SEQ_LEN
    ):
        raise ValueError("RunConfig limits must match the fixed experiment limits")
    if type(config.shard_size) is not int or config.shard_size <= 0:
        raise ValueError("RunConfig shard_size must be a positive integer")
    if (
        type(config.expected_bridge_problems) is not int
        or config.expected_bridge_problems < 0
    ):
        raise ValueError("expected_bridge_problems must be a non-negative integer")
    if config.expected_unpadded_prompts is not None and (
        type(config.expected_unpadded_prompts) is not int
        or config.expected_unpadded_prompts < 0
    ):
        raise ValueError(
            "expected_unpadded_prompts must be a non-negative integer or None"
        )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_hash = _config_hash(config)
    run_meta_path = root / "run_meta.parquet"
    if run_meta_path.exists():
        existing_meta = _read_global_table(run_meta_path, table="run_meta")
        hashes = existing_meta.column("config_hash").to_pylist()
        if hashes != [config_hash]:
            raise RuntimeError("Existing run config does not match requested config")
    elif (root / "manifests").exists() or (root / "run-manifest.json").exists():
        raise RuntimeError("Cannot resume shards without a valid run_meta table")

    source_rows = {row.source_row_id: row for row in rows}
    if len(source_rows) != len(rows):
        raise ValueError("FLenQA source_row_id values must be unique")
    prompts = deduplicate(rows)
    if not prompts:
        raise ValueError("FLenQA experiment requires at least one prompt")
    gate_result = bridge_gate(
        prompts,
        expected_applicable=config.expected_bridge_problems,
    )
    prepared_prompts = tuple(
        prepare_prompt(
            prompt,
            tokenizer,
            max_seq_len=MAX_SEQ_LEN,
            bridge=extract_bridge(prompt),
        )
        for prompt in prompts
    )
    if config.expected_unpadded_prompts is not None:
        assert_unpadded_prompt_count(
            prepared_prompts,
            expected=config.expected_unpadded_prompts,
        )
    shards = plan_shards(prepared_prompts, shard_size=config.shard_size)
    expected_manifest = RunManifest(
        config_hash=config_hash,
        prompt_ids=tuple(prompt.prompt_id for prompt in prompts),
        shard_ids=tuple(shard.shard_id for shard in shards),
        bridge_gate=gate_result,
    )

    global_batches = {
        "run_meta": _run_meta_batch(config, config_hash),
        "vocab": _vocab_batch(tokenizer),
    }
    has_shards = (root / "manifests").exists() and any(
        (root / "manifests").glob("shard-*.json")
    )
    for table in GLOBAL_SCHEMAS:
        path = root / f"{table}.parquet"
        expected_table = pa.Table.from_batches(
            [global_batches[table]],
            schema=GLOBAL_SCHEMAS[table],
        )
        if path.exists():
            if not _read_global_table(path, table=table).equals(expected_table):
                raise RuntimeError(f"Existing global {table} table does not match")
        else:
            if has_shards or (root / "run-manifest.json").exists():
                raise RuntimeError(f"Cannot resume without global {table} table")
            _write_global_batch(path, global_batches[table])

    completion_path = root / "run-manifest.json"
    if completion_path.exists():
        return _validate_run(
            root,
            _read_run_manifest(completion_path),
            expected=expected_manifest,
        )

    for shard in shards:
        run_shard(
            shard,
            prepared_prompts,
            output_dir=root,
            source_rows=source_rows,
            runners=runners,
            tokenizer=tokenizer,
            generate=generate,
        )
    for shard in shards:
        if not is_shard_complete(
            root,
            shard_id=shard.shard_id,
            schemas=TABLE_SCHEMAS,
            required_tables=REQUIRED_SHARD_TABLES,
        ):
            raise RuntimeError(f"Shard {shard.shard_id} is incomplete")

    payload = _manifest_payload(expected_manifest, root)
    temporary = completion_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, completion_path)
    return _validate_run(
        root,
        _read_run_manifest(completion_path),
        expected=expected_manifest,
    )
