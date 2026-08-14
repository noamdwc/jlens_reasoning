"""Apply paired lenses to one prepared FLenQA prompt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import pyarrow as pa
import torch

from jlens_reasoning.benchmarks.flenqa.positions import PreparedPrompt
from jlens_reasoning.benchmarks.flenqa.storage import record_batch


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
        return LensPassResult(logits, model_logits, input_ids)


@dataclass(frozen=True, slots=True)
class RankedToken:
    rank: int
    token_id: int
    logit: float


@dataclass(frozen=True, slots=True)
class PromptResult:
    batches: Mapping[str, pa.RecordBatch]
    returned_layers: tuple[int, ...]
    max_abs_logit_diff: float


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
    boundary_ids = torch.nonzero(logits == threshold, as_tuple=False).flatten()[:remaining]
    selected_ids = torch.cat((strict_ids, boundary_ids)).sort().values
    selected_logits = logits[selected_ids]
    order = torch.argsort(selected_logits, descending=True, stable=True)
    ordered_ids = selected_ids[order]
    return tuple(
        RankedToken(rank, int(token_id), float(logits[token_id].item()))
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


def _validate_pass(
    name: str,
    result: LensPassResult,
    *,
    prepared_input_ids: tuple[int, ...],
    position_count: int,
) -> tuple[tuple[int, ...], int]:
    if _input_ids(result.input_ids) != prepared_input_ids:
        raise RuntimeError(f"{name} Lens input IDs differ from prepared input IDs")
    if not result.logits_by_layer:
        raise RuntimeError(f"{name} Lens returned no layer logits")

    vocabulary_sizes: set[int] = set()
    for layer, logits in result.logits_by_layer.items():
        if type(layer) is not int:
            raise RuntimeError("Lens layer keys must be integers")
        if logits.ndim != 2 or logits.shape[0] != position_count:
            raise RuntimeError(f"Layer {layer} logits rows must match unique positions")
        vocabulary_sizes.add(int(logits.shape[1]))
    if len(vocabulary_sizes) != 1:
        raise RuntimeError("Lens layers must use one vocabulary size")
    vocabulary_size = next(iter(vocabulary_sizes))
    if vocabulary_size <= 0:
        raise RuntimeError("Lens vocabulary must be nonempty")
    if result.model_logits.ndim != 2 or result.model_logits.shape[0] != position_count:
        raise RuntimeError("Lens model-logit rows must match unique positions")
    if result.model_logits.numel() == 0:
        raise RuntimeError("Lens model logits must be nonempty")
    return tuple(sorted(result.logits_by_layer)), vocabulary_size


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
            for token in deterministic_topk(row, k=top_k):
                columns["prompt_id"].append(prompt_id)
                columns["lens_kind"].append(lens_kind)
                columns["layer"].append(layer)
                columns["position"].append(position)
                columns["rank"].append(token.rank)
                columns["token_id"].append(token.token_id)
                columns["logit"].append(token.logit)
    return columns


def _merge_columns(
    first: Mapping[str, Sequence[Any]],
    second: Mapping[str, Sequence[Any]],
) -> dict[str, list[Any]]:
    return {name: [*first[name], *second[name]] for name in first}


def run_prompt(
    prepared: PreparedPrompt,
    *,
    runners: LensRunners,
    top_k: int,
    max_seq_len: int,
    logits_rtol: float,
    logits_atol: float,
) -> PromptResult:
    """Run both lens modes once at each unique meaningful position."""
    positions = prepared.unique_positions
    if not positions:
        raise ValueError("Prepared prompt has no execution positions")

    jacobian = runners.jacobian.run(
        prepared.prompt.text,
        positions=positions,
        max_seq_len=max_seq_len,
    )
    logit = runners.logit.run(
        prepared.prompt.text,
        positions=positions,
        max_seq_len=max_seq_len,
    )
    jacobian_layers, jacobian_vocab = _validate_pass(
        "Jacobian",
        jacobian,
        prepared_input_ids=prepared.input_ids,
        position_count=len(positions),
    )
    logit_layers, logit_vocab = _validate_pass(
        "Logit",
        logit,
        prepared_input_ids=prepared.input_ids,
        position_count=len(positions),
    )
    if jacobian_layers != logit_layers:
        raise RuntimeError("Jacobian and Logit Lens layer keys differ")
    if jacobian_vocab != logit_vocab:
        raise RuntimeError("Jacobian and Logit Lens vocabulary sizes differ")
    if jacobian.model_logits.shape != logit.model_logits.shape:
        raise RuntimeError("Jacobian and Logit Lens model-logit shapes differ")
    if not torch.allclose(
        jacobian.model_logits,
        logit.model_logits,
        rtol=logits_rtol,
        atol=logits_atol,
    ):
        raise RuntimeError("Jacobian and Logit Lens model logits are not allclose")
    max_abs_logit_diff = float(
        (jacobian.model_logits - logit.model_logits).abs().max().item()
    )

    prompt = prepared.prompt
    labeled_positions = tuple(
        (label, position)
        for label, label_positions in prepared.positions.items()
        for position in label_positions
    )
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
            "prompt_id": [prompt.prompt_id] * len(labeled_positions),
            "position": [position for _, position in labeled_positions],
            "label": [label for label, _ in labeled_positions],
        },
    )
    topk = record_batch(
        "topk",
        _merge_columns(
            _topk_columns(
                prompt_id=prompt.prompt_id,
                lens_kind="jacobian",
                logits_by_layer=jacobian.logits_by_layer,
                positions=positions,
                top_k=top_k,
            ),
            _topk_columns(
                prompt_id=prompt.prompt_id,
                lens_kind="logit",
                logits_by_layer=logit.logits_by_layer,
                positions=positions,
                top_k=top_k,
            ),
        ),
    )
    return PromptResult(
        batches={"prompts": prompts, "positions": position_batch, "topk": topk},
        returned_layers=jacobian_layers,
        max_abs_logit_diff=max_abs_logit_diff,
    )
