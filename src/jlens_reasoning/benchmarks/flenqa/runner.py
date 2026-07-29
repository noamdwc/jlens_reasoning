"""Run paired Jacobian and Logit Lens passes at semantic FLenQA positions."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaRow,
    deduplicate,
)
from jlens_reasoning.benchmarks.flenqa.positions import (
    BridgeGateResult,
    PreparedPrompt,
    bridge_gate,
    prepare_prompt,
    validate_prepared_prompt,
)
from jlens_reasoning.benchmarks.flenqa.storage import (
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


@dataclass(frozen=True, slots=True)
class RunConfig:
    model_name: str
    lens_revision: str
    tokenizer_name: str
    code_revision: str
    layers: tuple[int, ...] | None = None
    top_k: int = 25
    padding_sample_seed: int = 1729
    shard_size: int = 500
    max_seq_len: int = 4096
    logits_rtol: float = 1e-5
    logits_atol: float = 1e-6
    expected_source_rows: int = 12_000
    expected_bridge_problems: int = 200


@dataclass(frozen=True, slots=True)
class PromptShard:
    shard_id: int
    prompt_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RunManifest:
    config_hash: str
    prompt_ids: tuple[str, ...]
    shard_ids: tuple[int, ...]
    returned_layers: tuple[int, ...]
    max_abs_logit_diff: float
    bridge_gate: BridgeGateResult


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
class RankedToken:
    rank: int
    token_id: int
    logit: float


def deterministic_topk(
    logits: torch.Tensor,
    *,
    k: int,
) -> tuple[RankedToken, ...]:
    """Order by descending logit and use lower token IDs to break ties."""
    if logits.ndim != 1:
        raise ValueError("deterministic_topk expects one logits vector")
    if type(k) is not int or k < 0:
        raise ValueError("top-k must be a non-negative integer")
    if torch.isnan(logits).any():
        raise ValueError("top-k logits must not contain NaN")
    count = min(k, logits.numel())
    if count == 0:
        return ()

    threshold = torch.topk(logits, k=count, sorted=False).values.min()
    strict_ids = torch.nonzero(logits > threshold, as_tuple=False).flatten()
    remaining = count - strict_ids.numel()
    boundary_ids = torch.nonzero(logits == threshold, as_tuple=False).flatten()[
        :remaining
    ]
    selected_ids = torch.cat((strict_ids, boundary_ids)).sort().values
    selected_logits = logits[selected_ids]
    order = torch.argsort(selected_logits, descending=True, stable=True)
    ordered_ids = selected_ids[order]
    return tuple(
        RankedToken(
            rank=rank,
            token_id=int(token_id),
            logit=float(logits[token_id].item()),
        )
        for rank, token_id in enumerate(ordered_ids.tolist(), start=1)
    )


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


def _validate_layer_logits(
    logits_by_layer: Mapping[int, torch.Tensor],
    *,
    position_count: int,
) -> int:
    if not logits_by_layer:
        raise RuntimeError("Lens pass returned no layer logits")
    vocabulary_sizes: set[int] = set()
    for layer, logits in logits_by_layer.items():
        if type(layer) is not int:
            raise RuntimeError("Lens layer keys must be integers")
        if logits.ndim != 2 or logits.shape[0] != position_count:
            raise RuntimeError(
                f"Layer {layer} logits rows must match unique positions"
            )
        vocabulary_sizes.add(int(logits.shape[1]))
    if len(vocabulary_sizes) != 1:
        raise RuntimeError("Lens layers must use one vocabulary size")
    vocabulary_size = next(iter(vocabulary_sizes))
    if vocabulary_size <= 0:
        raise RuntimeError("Lens vocabulary must be nonempty")
    return vocabulary_size


def _topk_columns(
    *,
    prompt_id: str,
    lens_kind: str,
    logits_by_layer: Mapping[int, torch.Tensor],
    positions: Sequence[int],
    top_k: int,
) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {
        "prompt_id": [],
        "lens_kind": [],
        "layer": [],
        "position": [],
        "rank": [],
        "token_id": [],
        "logit": [],
    }
    for layer, layer_logits in sorted(logits_by_layer.items()):
        for position, row in zip(positions, layer_logits, strict=True):
            for value in deterministic_topk(row, k=top_k):
                columns["prompt_id"].append(prompt_id)
                columns["lens_kind"].append(lens_kind)
                columns["layer"].append(layer)
                columns["position"].append(position)
                columns["rank"].append(value.rank)
                columns["token_id"].append(value.token_id)
                columns["logit"].append(value.logit)
    return columns


def _merge_columns(
    first: Mapping[str, Sequence[Any]],
    second: Mapping[str, Sequence[Any]],
) -> dict[str, list[Any]]:
    return {
        name: [*first[name], *second[name]]
        for name in first
    }


def run_prompt(
    prepared: PreparedPrompt,
    *,
    runners: LensRunners,
    config: RunConfig,
) -> dict[str, pa.RecordBatch]:
    """Run both lens modes once at each unique meaningful position."""
    positions = prepared.unique_positions
    if not positions:
        raise ValueError("Prepared prompt has no execution positions")
    jacobian = runners.jacobian.run(
        prepared.prompt.text,
        positions=positions,
        max_seq_len=config.max_seq_len,
    )
    logit = runners.logit.run(
        prepared.prompt.text,
        positions=positions,
        max_seq_len=config.max_seq_len,
    )
    if _input_ids(jacobian.input_ids) != prepared.input_ids:
        raise RuntimeError("Jacobian Lens input IDs differ from prepared input IDs")
    if _input_ids(logit.input_ids) != prepared.input_ids:
        raise RuntimeError("Logit Lens input IDs differ from prepared input IDs")

    jacobian_layers = tuple(sorted(jacobian.logits_by_layer))
    logit_layers = tuple(sorted(logit.logits_by_layer))
    if jacobian_layers != logit_layers:
        raise RuntimeError("Jacobian and Logit Lens layer keys differ")
    jacobian_vocab = _validate_layer_logits(
        jacobian.logits_by_layer,
        position_count=len(positions),
    )
    logit_vocab = _validate_layer_logits(
        logit.logits_by_layer,
        position_count=len(positions),
    )
    if jacobian_vocab != logit_vocab:
        raise RuntimeError("Jacobian and Logit Lens vocabulary sizes differ")
    if jacobian.model_logits.shape != logit.model_logits.shape:
        raise RuntimeError("Jacobian and Logit Lens model-logit shapes differ")
    if (
        jacobian.model_logits.ndim != 2
        or jacobian.model_logits.shape[0] != len(positions)
    ):
        raise RuntimeError("Lens model-logit rows must match unique positions")
    if jacobian.model_logits.numel() == 0:
        raise RuntimeError("Lens model logits must be nonempty")
    if not torch.allclose(
        jacobian.model_logits,
        logit.model_logits,
        rtol=config.logits_rtol,
        atol=config.logits_atol,
    ):
        raise RuntimeError("Jacobian and Logit Lens model logits are not allclose")
    max_abs_logit_diff = (
        jacobian.model_logits - logit.model_logits
    ).abs().max().item()

    prompt = prepared.prompt
    prompts = record_batch(
        "prompts",
        {
            "prompt_id": [prompt.prompt_id],
            "canonical_index": [prompt.canonical_index],
            "problem_id": [prompt.problem_id],
            "task": [prompt.task],
            "label": [prompt.label],
            "text": [prompt.text],
            "input_ids": [list(prepared.input_ids)],
            "bridge": [prepared.bridge],
            "max_abs_logit_diff": [max_abs_logit_diff],
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
        },
    )
    position_batch = record_batch(
        "positions",
        {
            "prompt_id": [prompt.prompt_id] * len(prepared.positions),
            "position": [item.position for item in prepared.positions],
            "label": [item.label for item in prepared.positions],
        },
    )
    jacobian_topk = _topk_columns(
        prompt_id=prompt.prompt_id,
        lens_kind="jacobian",
        logits_by_layer=jacobian.logits_by_layer,
        positions=positions,
        top_k=config.top_k,
    )
    logit_topk = _topk_columns(
        prompt_id=prompt.prompt_id,
        lens_kind="logit",
        logits_by_layer=logit.logits_by_layer,
        positions=positions,
        top_k=config.top_k,
    )
    topk = record_batch(
        "topk",
        _merge_columns(jacobian_topk, logit_topk),
    )
    expected_topk_rows = (
        2
        * len(jacobian_layers)
        * len(positions)
        * min(config.top_k, jacobian_vocab)
    )
    if topk.num_rows != expected_topk_rows:
        raise RuntimeError("Top-k row count does not match unique positions")
    return {
        "prompts": prompts,
        "positions": position_batch,
        "topk": topk,
    }


def _validate_config(config: RunConfig) -> None:
    if type(config.top_k) is not int or config.top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if type(config.shard_size) is not int or config.shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    if type(config.max_seq_len) is not int or config.max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer")
    if (
        type(config.expected_source_rows) is not int
        or config.expected_source_rows <= 0
    ):
        raise ValueError("expected_source_rows must be a positive integer")
    if (
        type(config.expected_bridge_problems) is not int
        or config.expected_bridge_problems < 0
    ):
        raise ValueError("expected_bridge_problems must be non-negative")
    if config.logits_rtol < 0 or config.logits_atol < 0:
        raise ValueError("logit tolerances must be non-negative")
    if config.layers is not None and (
        any(type(layer) is not int for layer in config.layers)
        or len(set(config.layers)) != len(config.layers)
    ):
        raise ValueError("configured layers must be unique integers")


def plan_shards(
    prepared_prompts: Sequence[PreparedPrompt],
    *,
    shard_size: int,
) -> tuple[PromptShard, ...]:
    """Assign immutable shards from the canonical prompt order."""
    if type(shard_size) is not int or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    indices = tuple(
        prepared.prompt.canonical_index for prepared in prepared_prompts
    )
    if indices != tuple(sorted(set(indices))):
        raise ValueError("Prepared prompt indices must be sorted and unique")
    return tuple(
        PromptShard(
            shard_id=start // shard_size,
            prompt_indices=indices[start : start + shard_size],
        )
        for start in range(0, len(indices), shard_size)
    )


def run_shard(
    shard: PromptShard,
    prepared_prompts: Sequence[PreparedPrompt],
    *,
    output_dir: Path,
    runners: LensRunners,
    config: RunConfig,
) -> ShardManifest:
    """Run or resume one manifest-committed FLenQA shard."""
    if type(shard.shard_id) is not int or shard.shard_id < 0:
        raise ValueError("shard_id must be a non-negative integer")
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
        required_tables=REQUIRED_TABLES,
    ):
        manifest = read_shard_manifest(root, shard_id=shard.shard_id)
        if manifest.prompt_ids != prompt_ids:
            raise RuntimeError("Completed shard prompt membership does not match")
        return validate_shard_manifest(
            root,
            manifest,
            schemas=TABLE_SCHEMAS,
            required_tables=REQUIRED_TABLES,
        )

    reset_incomplete_shard(root, shard_id=shard.shard_id)
    writer = ShardWriter(
        root,
        shard_id=shard.shard_id,
        schemas=TABLE_SCHEMAS,
        required_tables=REQUIRED_TABLES,
        prompt_ids=prompt_ids,
    )
    try:
        for prepared in selected:
            batches = run_prompt(prepared, runners=runners, config=config)
            if tuple(batches) != REQUIRED_TABLES:
                raise RuntimeError("run_prompt did not return every required table")
            for table in REQUIRED_TABLES:
                writer.append(table, batches[table])
        return writer.commit()
    except BaseException:
        writer.abort()
        raise


def _config_hash(config: RunConfig) -> str:
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


def _run_meta_payload(
    config: RunConfig,
    *,
    config_hash: str,
    returned_layers: Sequence[int] | None,
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "config_hash": config_hash,
        "requested_layers": (
            None if config.layers is None else list(config.layers)
        ),
        "returned_layers": (
            None if returned_layers is None else list(returned_layers)
        ),
    }


def _manifest_payload(manifest: RunManifest) -> dict[str, Any]:
    return {
        "config_hash": manifest.config_hash,
        "prompt_ids": list(manifest.prompt_ids),
        "shard_ids": list(manifest.shard_ids),
        "returned_layers": list(manifest.returned_layers),
        "max_abs_logit_diff": manifest.max_abs_logit_diff,
        "bridge_gate": asdict(manifest.bridge_gate),
    }


def _read_run_manifest(path: Path) -> RunManifest:
    payload = _read_json(path)
    gate = payload["bridge_gate"]
    if not isinstance(gate, Mapping):
        raise RuntimeError("run manifest bridge_gate must be an object")
    return RunManifest(
        config_hash=str(payload["config_hash"]),
        prompt_ids=tuple(str(value) for value in payload["prompt_ids"]),
        shard_ids=tuple(int(value) for value in payload["shard_ids"]),
        returned_layers=tuple(int(value) for value in payload["returned_layers"]),
        max_abs_logit_diff=float(payload["max_abs_logit_diff"]),
        bridge_gate=BridgeGateResult(
            applicable=int(gate["applicable"]),
            resolved=int(gate["resolved"]),
        ),
    )


def _scan_run_outputs(
    root: Path,
    shard_ids: Sequence[int],
) -> tuple[tuple[int, ...], float]:
    layer_sets: dict[str, set[int]] = {}
    max_differences: list[float] = []
    for shard_id in shard_ids:
        stem = f"shard-{shard_id:05d}.parquet"
        prompts = pq.read_table(root / "prompts" / stem).to_pydict()
        max_differences.extend(
            float(value) for value in prompts["max_abs_logit_diff"]
        )
        topk = pq.read_table(root / "topk" / stem).to_pydict()
        for prompt_id, layer in zip(
            topk["prompt_id"],
            topk["layer"],
            strict=True,
        ):
            layer_sets.setdefault(str(prompt_id), set()).add(int(layer))
    if not max_differences or not layer_sets:
        raise RuntimeError("Completed run outputs must be nonempty")
    distinct = {tuple(sorted(layers)) for layers in layer_sets.values()}
    if len(distinct) != 1:
        raise RuntimeError("Returned layer keys differ between prompts")
    return next(iter(distinct)), max(max_differences)


def _validate_run_shards(
    root: Path,
    manifest: RunManifest,
) -> None:
    actual_prompt_ids: list[str] = []
    for shard_id in manifest.shard_ids:
        if not is_shard_complete(
            root,
            shard_id=shard_id,
            schemas=TABLE_SCHEMAS,
            required_tables=REQUIRED_TABLES,
        ):
            raise RuntimeError(f"Run shard {shard_id} is incomplete")
        actual_prompt_ids.extend(
            read_shard_manifest(root, shard_id=shard_id).prompt_ids
        )
    if tuple(actual_prompt_ids) != manifest.prompt_ids:
        raise RuntimeError("Run shard prompt membership does not match manifest")


def run_benchmark(
    rows: Sequence[FlenqaRow],
    *,
    output_dir: Path,
    tokenizer: Any,
    runners: LensRunners,
    config: RunConfig,
) -> RunManifest:
    """Run or safely resume the complete prepared FLenQA benchmark."""
    _validate_config(config)
    if len(rows) != config.expected_source_rows:
        raise ValueError(
            f"Expected {config.expected_source_rows} source rows; found {len(rows)}"
        )
    prompts = deduplicate(rows)
    if not prompts:
        raise ValueError("FLenQA benchmark requires at least one prompt")
    gate_result = bridge_gate(
        prompts,
        expected_applicable=config.expected_bridge_problems,
    )
    prepared_prompts = tuple(
        validate_prepared_prompt(
            prepare_prompt(
                prompt,
                tokenizer,
                max_seq_len=config.max_seq_len,
                sample_seed=config.padding_sample_seed,
            )
        )
        for prompt in prompts
    )
    shards = plan_shards(prepared_prompts, shard_size=config.shard_size)
    prompt_ids = tuple(prompt.prompt_id for prompt in prompts)
    shard_ids = tuple(shard.shard_id for shard in shards)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_hash = _config_hash(config)
    meta_path = root / "run-meta.json"
    if meta_path.exists():
        existing_meta = _read_json(meta_path)
        if existing_meta.get("config_hash") != config_hash:
            raise RuntimeError("Existing run configuration does not match")
    else:
        if (root / "run-manifest.json").exists() or (root / "manifests").exists():
            raise RuntimeError("Cannot resume shards without run metadata")
        _atomic_json(
            meta_path,
            _run_meta_payload(
                config,
                config_hash=config_hash,
                returned_layers=None,
            ),
        )

    completion_path = root / "run-manifest.json"
    if completion_path.exists():
        manifest = _read_run_manifest(completion_path)
        if (
            manifest.config_hash != config_hash
            or manifest.prompt_ids != prompt_ids
            or manifest.shard_ids != shard_ids
            or manifest.bridge_gate != gate_result
        ):
            raise RuntimeError("Existing run manifest does not match requested run")
        _validate_run_shards(root, manifest)
        return manifest

    for shard in shards:
        run_shard(
            shard,
            prepared_prompts,
            output_dir=root,
            runners=runners,
            config=config,
        )
    provisional = RunManifest(
        config_hash=config_hash,
        prompt_ids=prompt_ids,
        shard_ids=shard_ids,
        returned_layers=(),
        max_abs_logit_diff=0.0,
        bridge_gate=gate_result,
    )
    _validate_run_shards(root, provisional)
    returned_layers, max_abs_logit_diff = _scan_run_outputs(root, shard_ids)
    if config.layers is not None and returned_layers != tuple(sorted(config.layers)):
        raise RuntimeError("Returned layer keys do not match configured layers")
    manifest = RunManifest(
        config_hash=config_hash,
        prompt_ids=prompt_ids,
        shard_ids=shard_ids,
        returned_layers=returned_layers,
        max_abs_logit_diff=max_abs_logit_diff,
        bridge_gate=gate_result,
    )
    _atomic_json(
        meta_path,
        _run_meta_payload(
            config,
            config_hash=config_hash,
            returned_layers=returned_layers,
        ),
    )
    _atomic_json(completion_path, _manifest_payload(manifest))
    return manifest
