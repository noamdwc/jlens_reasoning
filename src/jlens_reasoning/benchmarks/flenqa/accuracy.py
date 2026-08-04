"""Resumable behavioral accuracy evaluation for deduplicated FLenQA prompts."""

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
from tqdm.auto import tqdm

from jlens_reasoning.benchmarks.flenqa.accuracy_storage import (
    REQUIRED_TABLES,
    TABLE_SCHEMAS,
    ShardManifest,
    ShardWriter,
    is_shard_complete,
    read_shard_manifest,
    record_batch,
    reset_incomplete_shard,
    validate_shard_manifest,
)
from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaPrompt,
    FlenqaRow,
    deduplicate,
)
from jlens_reasoning.evaluation import (
    GenerationStatus,
    ModelOutput,
    evaluate_paper_binary,
)


@dataclass(frozen=True, slots=True)
class AccuracyRunConfig:
    model_name: str
    tokenizer_name: str
    code_revision: str
    max_seq_len: int = 4096
    max_new_tokens: int = 64
    shard_size: int = 100
    expected_source_rows: int = 12_000
    expected_prompts: int = 9_862
    decoding_mode: str = "greedy"


class GenerateOutput(Protocol):
    def __call__(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> ModelOutput: ...


@dataclass(frozen=True, slots=True)
class AccuracyShard:
    shard_id: int
    prompt_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AccuracyRunManifest:
    config_hash: str
    prompt_ids: tuple[str, ...]
    shard_ids: tuple[int, ...]


def _validate_config(config: AccuracyRunConfig) -> None:
    for name in ("model_name", "tokenizer_name", "code_revision"):
        value = getattr(config, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    for name in (
        "max_seq_len",
        "max_new_tokens",
        "shard_size",
        "expected_source_rows",
        "expected_prompts",
    ):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.decoding_mode != "greedy":
        raise ValueError("FLenQA accuracy decoding_mode must be 'greedy'")


def _config_hash(config: AccuracyRunConfig) -> str:
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{path.name} must contain a JSON object")
    return payload


def _validate_run_meta(
    root: Path,
    config: AccuracyRunConfig,
    config_hash: str,
) -> None:
    path = root / "run-meta.json"
    expected = {"config": asdict(config), "config_hash": config_hash}
    if path.exists():
        try:
            actual = _read_json(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Existing run configuration is unreadable") from exc
        if actual != expected:
            raise RuntimeError("Existing run configuration does not match")
        return
    _atomic_json(path, expected)


def _input_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise RuntimeError("Tokenizer output must contain input_ids")
    value = encoded["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, Sequence)
        and value
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        if len(value) != 1:
            raise RuntimeError("Tokenizer output must contain one input sequence")
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError("Tokenizer input_ids must be a sequence")
    try:
        token_ids = tuple(int(token_id) for token_id in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Tokenizer input_ids must contain integers") from exc
    if not token_ids:
        raise RuntimeError("Tokenizer input_ids must be nonempty")
    return token_ids


def _ctx_size(prompt: FlenqaPrompt) -> int:
    values = {item.ctx_size for item in prompt.provenance}
    if len(values) != 1:
        raise ValueError("Deduplicated prompt mixes nominal context sizes")
    return next(iter(values))


def run_prompt(
    prompt: FlenqaPrompt,
    *,
    tokenizer: Any,
    generate: GenerateOutput,
    config: AccuracyRunConfig,
) -> pa.RecordBatch:
    """Generate and score one unique FLenQA prompt."""
    input_ids = _input_ids(tokenizer, prompt.text)
    if len(input_ids) > config.max_seq_len:
        raise ValueError(
            f"Prompt {prompt.prompt_id} exceeds maximum sequence length "
            f"{config.max_seq_len}: {len(input_ids)} tokens"
        )
    output = generate(prompt.text, max_new_tokens=config.max_new_tokens)
    if not isinstance(output, ModelOutput):
        raise TypeError("FLenQA generation callback must return ModelOutput")
    if output.generation_status is GenerationStatus.GENERATION_ERROR:
        raise RuntimeError(output.generation_error or "FLenQA generation failed")
    evaluation = evaluate_paper_binary(output, expected=prompt.label)
    return record_batch(
        {
            "prompt_id": [prompt.prompt_id],
            "canonical_index": [prompt.canonical_index],
            "problem_id": [prompt.problem_id],
            "task": [prompt.task],
            "label": [prompt.label],
            "text": [prompt.text],
            "ctx_size": [_ctx_size(prompt)],
            "input_ids": [list(input_ids)],
            "n_input_tokens": [len(input_ids)],
            "provenance": [
                [
                    {
                        "source_row_id": item.source_row_id,
                        "ctx_size": item.ctx_size,
                        "padding_type": item.padding_type,
                        "dispersion": item.dispersion,
                    }
                    for item in prompt.provenance
                ]
            ],
            "generated_token_ids": [list(output.token_ids)],
            "generated_token_pieces": [list(output.token_pieces)],
            "generated_text": [output.text],
            "generation_status": [output.generation_status.value],
            "finish_reason": [output.finish_reason],
            "verdict": [evaluation.verdict],
            "correct": [evaluation.correct],
        }
    )


def plan_shards(
    prompts: Sequence[FlenqaPrompt],
    *,
    shard_size: int,
) -> tuple[AccuracyShard, ...]:
    """Assign immutable shards from canonical prompt order."""
    if type(shard_size) is not int or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    indices = tuple(prompt.canonical_index for prompt in prompts)
    if indices != tuple(sorted(set(indices))):
        raise ValueError("Prompt canonical indices must be sorted and unique")
    return tuple(
        AccuracyShard(
            shard_id=start // shard_size,
            prompt_indices=indices[start : start + shard_size],
        )
        for start in range(0, len(indices), shard_size)
    )


def run_shard(
    shard: AccuracyShard,
    prompts: Sequence[FlenqaPrompt],
    *,
    output_dir: Path,
    tokenizer: Any,
    generate: GenerateOutput,
    config: AccuracyRunConfig,
    on_prompts_completed: Callable[[int], None] | None = None,
) -> ShardManifest:
    """Run or resume one manifest-committed accuracy shard."""
    if type(shard.shard_id) is not int or shard.shard_id < 0:
        raise ValueError("shard_id must be a non-negative integer")
    by_index = {prompt.canonical_index: prompt for prompt in prompts}
    if len(by_index) != len(prompts):
        raise ValueError("Prompt canonical indices must be unique")
    try:
        selected = tuple(by_index[index] for index in shard.prompt_indices)
    except KeyError as exc:
        raise ValueError(
            f"Shard references unknown canonical index {exc.args[0]}"
        ) from exc
    prompt_ids = tuple(prompt.prompt_id for prompt in selected)
    root = Path(output_dir)
    if is_shard_complete(
        root,
        shard_id=shard.shard_id,
        schemas=TABLE_SCHEMAS,
        required_tables=REQUIRED_TABLES,
    ):
        manifest = read_shard_manifest(root, shard_id=shard.shard_id)
        if manifest.prompt_ids != prompt_ids:
            raise RuntimeError("Completed accuracy shard membership does not match")
        validated = validate_shard_manifest(
            root,
            manifest,
            schemas=TABLE_SCHEMAS,
            required_tables=REQUIRED_TABLES,
        )
        if on_prompts_completed is not None:
            on_prompts_completed(len(selected))
        return validated

    reset_incomplete_shard(root, shard_id=shard.shard_id)
    writer = ShardWriter(
        root,
        shard_id=shard.shard_id,
        schemas=TABLE_SCHEMAS,
        required_tables=REQUIRED_TABLES,
        prompt_ids=prompt_ids,
    )
    try:
        for prompt in selected:
            writer.append(
                "results",
                run_prompt(
                    prompt,
                    tokenizer=tokenizer,
                    generate=generate,
                    config=config,
                ),
            )
            if on_prompts_completed is not None:
                on_prompts_completed(1)
        return writer.commit()
    except BaseException:
        writer.abort()
        raise


def _manifest_payload(manifest: AccuracyRunManifest) -> dict[str, Any]:
    return {
        "config_hash": manifest.config_hash,
        "prompt_ids": list(manifest.prompt_ids),
        "shard_ids": list(manifest.shard_ids),
    }


def _read_run_manifest(path: Path) -> AccuracyRunManifest:
    payload = _read_json(path)
    return AccuracyRunManifest(
        config_hash=str(payload["config_hash"]),
        prompt_ids=tuple(str(value) for value in payload["prompt_ids"]),
        shard_ids=tuple(int(value) for value in payload["shard_ids"]),
    )


def _run_shards_complete(root: Path, manifest: AccuracyRunManifest) -> bool:
    actual_prompt_ids: list[str] = []
    for shard_id in manifest.shard_ids:
        if not is_shard_complete(
            root,
            shard_id=shard_id,
            schemas=TABLE_SCHEMAS,
            required_tables=REQUIRED_TABLES,
        ):
            return False
        actual_prompt_ids.extend(
            read_shard_manifest(root, shard_id=shard_id).prompt_ids
        )
    return tuple(actual_prompt_ids) == manifest.prompt_ids


def run_accuracy(
    rows: Sequence[FlenqaRow],
    *,
    output_dir: Path,
    tokenizer: Any,
    generate: GenerateOutput,
    config: AccuracyRunConfig,
    show_progress: bool = True,
) -> AccuracyRunManifest:
    """Run or safely resume behavioral accuracy on every unique prompt."""
    _validate_config(config)
    if len(rows) != config.expected_source_rows:
        raise ValueError(
            f"Expected {config.expected_source_rows} source rows; found {len(rows)}"
        )
    prompts = deduplicate(rows)
    if len(prompts) != config.expected_prompts:
        raise ValueError(
            f"Expected {config.expected_prompts} unique prompts; found {len(prompts)}"
        )
    root = Path(output_dir)
    config_hash = _config_hash(config)
    _validate_run_meta(root, config, config_hash)
    shards = plan_shards(prompts, shard_size=config.shard_size)
    prompt_ids = tuple(prompt.prompt_id for prompt in prompts)
    shard_ids = tuple(shard.shard_id for shard in shards)

    run_manifest_path = root / "run-manifest.json"
    if run_manifest_path.exists():
        try:
            completed = _read_run_manifest(run_manifest_path)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            completed = None
        if completed is not None:
            if completed.config_hash != config_hash:
                raise RuntimeError("Completed run configuration does not match")
            if completed.prompt_ids != prompt_ids or completed.shard_ids != shard_ids:
                raise RuntimeError("Completed accuracy run membership does not match")
            if _run_shards_complete(root, completed):
                return completed

    manifests: list[ShardManifest] = []
    with tqdm(
        total=len(prompts),
        desc="FLenQA accuracy prompts",
        unit="prompt",
        disable=not show_progress,
    ) as progress:
        for shard in shards:
            manifests.append(
                run_shard(
                    shard,
                    prompts,
                    output_dir=root,
                    tokenizer=tokenizer,
                    generate=generate,
                    config=config,
                    on_prompts_completed=progress.update,
                )
            )
    manifest = AccuracyRunManifest(
        config_hash=config_hash,
        prompt_ids=prompt_ids,
        shard_ids=tuple(item.shard_id for item in manifests),
    )
    if not _run_shards_complete(root, manifest):
        raise RuntimeError("Completed accuracy run has invalid shards")
    _atomic_json(run_manifest_path, _manifest_payload(manifest))
    return manifest


def load_accuracy_results(
    root: Path,
    manifest: AccuracyRunManifest,
) -> pa.Table:
    """Load and validate every completed accuracy shard in run order."""
    root = Path(root)
    try:
        metadata = _read_json(root / "run-meta.json")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Accuracy run configuration is unreadable") from exc
    if metadata.get("config_hash") != manifest.config_hash:
        raise RuntimeError("Accuracy run configuration does not match manifest")
    if not _run_shards_complete(root, manifest):
        raise RuntimeError("Accuracy run shards are incomplete or invalid")
    tables = [
        pq.read_table(root / "results" / f"shard-{shard_id:05d}.parquet")
        for shard_id in manifest.shard_ids
    ]
    if not tables:
        raise RuntimeError("Accuracy run must contain at least one shard")
    table = pa.concat_tables(tables)
    if table.schema != TABLE_SCHEMAS["results"]:
        raise RuntimeError("Accuracy results use the wrong schema")
    actual_prompt_ids = tuple(str(value) for value in table["prompt_id"].to_pylist())
    if len(set(actual_prompt_ids)) != len(actual_prompt_ids):
        raise RuntimeError("Accuracy results contain duplicate prompt IDs")
    if actual_prompt_ids != manifest.prompt_ids:
        raise RuntimeError("Accuracy result prompt order does not match manifest")
    return table


__all__ = [
    "AccuracyRunConfig",
    "AccuracyRunManifest",
    "AccuracyShard",
    "GenerateOutput",
    "load_accuracy_results",
    "plan_shards",
    "run_accuracy",
    "run_prompt",
    "run_shard",
]
