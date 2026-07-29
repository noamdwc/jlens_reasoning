"""Run paired Jacobian and Logit Lens passes at semantic FLenQA positions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import pyarrow as pa
import torch

from jlens_reasoning.benchmarks.flenqa.positions import PreparedPrompt
from jlens_reasoning.benchmarks.flenqa.storage import record_batch


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
