"""Readout-only sanity checks for the public Qwen Jacobian lens."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from jlens.hooks import ActivationRecorder
from torch import nn

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
TOP_K = 25


@dataclass(frozen=True, slots=True)
class ReadoutCase:
    key: str
    prompt: str
    expected_answers: tuple[str, ...]
    target_concepts: tuple[str, ...]
    literal_argument: str | None = None


@dataclass(frozen=True, slots=True)
class SwapCase:
    key: str
    source_surface: str
    target_surface: str
    target_answers: tuple[str, ...]


READOUT_CASES = (
    ReadoutCase(
        key="spider",
        prompt="The number of legs on the animal that spins webs is",
        expected_answers=("8", "eight"),
        target_concepts=("spider",),
    ),
    ReadoutCase(
        key="france_capital",
        prompt="The capital of France is the city of",
        expected_answers=("Paris",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_language",
        prompt="Most people in France speak",
        expected_answers=("French",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_continent",
        prompt="France is a country on the continent of",
        expected_answers=("Europe",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_currency",
        prompt="The single-word name for the currency now used in France is the",
        expected_answers=("Euro",),
        target_concepts=("France",),
        literal_argument="France",
    ),
)


SWAP_CASES = (
    SwapCase("spider", " spider", " ant", ("6", "six")),
    SwapCase("france_capital", " France", " China", ("Beijing",)),
    SwapCase("france_language", " France", " China", ("Chinese",)),
    SwapCase("france_continent", " France", " China", ("Asia",)),
    SwapCase("france_currency", " France", " China", ("Yuan",)),
)


@dataclass(frozen=True, slots=True)
class TokenVariant:
    token_id: int
    surface: str


def single_token_surface(tokenizer: Any, surface: str) -> TokenVariant:
    token_ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Configured swap surface {surface!r} must encode as exactly one token"
        )
    return TokenVariant(token_id=token_ids[0], surface=surface)


def _concept_surfaces(concept: str) -> tuple[str, ...]:
    bases = (concept, concept.lower(), concept.capitalize(), concept.upper())
    ordered: list[str] = []
    for base in bases:
        for surface in (base, f" {base}"):
            if surface not in ordered:
                ordered.append(surface)
    return tuple(ordered)


def concept_token_variants(
    tokenizer: Any, concepts: Sequence[str]
) -> tuple[TokenVariant, ...]:
    variants: list[TokenVariant] = []
    seen_ids: set[int] = set()
    for concept in concepts:
        for surface in _concept_surfaces(concept):
            token_ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(token_ids) == 1 and token_ids[0] not in seen_ids:
                seen_ids.add(token_ids[0])
                variants.append(TokenVariant(token_id=token_ids[0], surface=surface))
    if not variants:
        raise ValueError(f"No single-token variants found for {tuple(concepts)!r}")
    return tuple(variants)


def jlens_vector(
    lens: Any,
    unembedding_weight: torch.Tensor,
    *,
    layer: int,
    token_id: int,
) -> torch.Tensor:
    jacobian = lens.jacobians[layer].to(
        device=unembedding_weight.device,
        dtype=torch.float32,
    )
    unembedding_row = unembedding_weight[token_id].to(dtype=torch.float32)
    return jacobian.T @ unembedding_row


def coordinate_swap(
    hidden: torch.Tensor,
    source_vector: torch.Tensor,
    target_vector: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if hidden.shape[-1] != source_vector.numel():
        raise ValueError("Source vector width does not match hidden width")
    if source_vector.shape != target_vector.shape:
        raise ValueError("Source and target vectors must have the same shape")

    working = hidden.float()
    vectors = torch.stack(
        (
            source_vector.to(device=hidden.device, dtype=torch.float32),
            target_vector.to(device=hidden.device, dtype=torch.float32),
        ),
        dim=-1,
    )
    coordinates = working @ torch.linalg.pinv(vectors).T
    delta = (coordinates.flip(-1) - coordinates) @ vectors.T
    return (working + float(alpha) * delta).to(dtype=hidden.dtype)


class LensCoordinateSwapper:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        vectors_by_layer: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        *,
        alpha: float,
    ) -> None:
        self._blocks = blocks
        self._vectors_by_layer = dict(vectors_by_layer)
        self._alpha = alpha
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, layer: int):
        source_vector, target_vector = self._vectors_by_layer[layer]

        def patch(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            hidden = output if torch.is_tensor(output) else output[0]
            patched = coordinate_swap(
                hidden,
                source_vector,
                target_vector,
                alpha=self._alpha,
            )
            if torch.is_tensor(output):
                return patched
            return (patched, *output[1:])

        return patch

    def __enter__(self) -> LensCoordinateSwapper:
        try:
            for layer in sorted(self._vectors_by_layer):
                self._handles.append(
                    self._blocks[layer].register_forward_hook(self._hook(layer))
                )
        except Exception:
            self._remove()
            raise
        return self

    def _remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __exit__(self, *exc: Any) -> None:
        self._remove()


def find_last_subsequence(
    sequence: Sequence[int], patterns: Iterable[Sequence[int]]
) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    for pattern in patterns:
        width = len(pattern)
        if not width:
            continue
        for start in range(len(sequence) - width + 1):
            if list(sequence[start : start + width]) == list(pattern):
                matches.append((start, start + width))
    if not matches:
        raise ValueError("Literal argument token span was not found in prompt")
    return max(matches, key=lambda span: (span[0], span[1]))


def positions_after_literal(
    tokenizer: Any, input_ids: torch.Tensor, literal: str
) -> list[int]:
    sequence = input_ids[0].tolist()
    patterns = [
        tokenizer.encode(surface, add_special_tokens=False)
        for surface in _concept_surfaces(literal)
    ]
    _, end = find_last_subsequence(sequence, patterns)
    positions = list(range(end, len(sequence)))
    if not positions:
        raise ValueError(f"No positions remain after literal argument {literal!r}")
    return positions


def positions_from_literal(
    tokenizer: Any,
    input_ids: torch.Tensor,
    literal: str,
) -> list[int]:
    sequence = input_ids[0].tolist()
    patterns = [
        tokenizer.encode(surface, add_special_tokens=False)
        for surface in _concept_surfaces(literal)
    ]
    start, _ = find_last_subsequence(sequence, patterns)
    return list(range(start, len(sequence)))


def best_target_rank(logits: torch.Tensor, target_ids: Sequence[int]) -> int:
    if logits.ndim != 1:
        raise ValueError("best_target_rank expects one logits vector")
    if not target_ids:
        raise ValueError("best_target_rank needs at least one target token")
    token_ids = torch.arange(logits.numel(), device=logits.device)
    ranks = []
    for target_id in target_ids:
        target_logit = logits[target_id]
        higher = (logits > target_logit).sum()
        earlier_ties = ((logits == target_logit) & (token_ids < target_id)).sum()
        ranks.append(1 + int(higher.item()) + int(earlier_ties.item()))
    return min(ranks)


def top_tokens(logits: torch.Tensor, tokenizer: Any, *, k: int = TOP_K) -> list[dict]:
    values, indices = torch.topk(logits, k=min(k, logits.numel()))
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode(
                [int(token_id)], clean_up_tokenization_spaces=False
            ),
            "logit": float(value),
        }
        for value, token_id in zip(values.tolist(), indices.tolist(), strict=True)
    ]


def _next_token_payload(
    logits: torch.Tensor,
    target_ids: Sequence[int],
    tokenizer: Any,
    *,
    top_k: int,
) -> dict[str, Any]:
    logits = logits.detach().float().cpu()
    top1_id = int(logits.argmax().item())
    return {
        "top1_id": top1_id,
        "top1_token": tokenizer.decode([top1_id], clean_up_tokenization_spaces=False),
        "target_rank": best_target_rank(logits, target_ids),
        "top_tokens": top_tokens(logits, tokenizer, k=top_k),
    }


def prepare_scoring_input(
    input_ids: torch.Tensor,
    *,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    tokenizer: Any,
    max_formatting_tokens: int = 2,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    scoring_input = input_ids
    prefix: list[dict[str, Any]] = []
    for _ in range(max_formatting_tokens):
        logits = forward_next_token(scoring_input)
        token_id = int(logits.argmax().item())
        surface = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if surface.strip():
            break
        prefix.append({"token_id": token_id, "token": surface})
        next_id = torch.tensor(
            [[token_id]],
            device=scoring_input.device,
            dtype=scoring_input.dtype,
        )
        scoring_input = torch.cat((scoring_input, next_id), dim=1)
    return scoring_input, prefix


def summarize_swap_logits(
    clean_logits: torch.Tensor,
    intervened_logits: Mapping[float, torch.Tensor],
    *,
    clean_answers: Sequence[str],
    target_answers: Sequence[str],
    tokenizer: Any,
    top_k: int,
) -> dict[str, Any]:
    expected_variants = concept_token_variants(tokenizer, clean_answers)
    expected_ids = tuple(variant.token_id for variant in expected_variants)
    target_variants = concept_token_variants(tokenizer, target_answers)
    target_ids = tuple(variant.token_id for variant in target_variants)
    normalized_clean = clean_logits.detach().float().cpu()
    clean = _next_token_payload(normalized_clean, target_ids, tokenizer, top_k=top_k)
    clean["expected_rank"] = best_target_rank(normalized_clean, expected_ids)
    clean["expected_top1"] = clean["expected_rank"] == 1
    interventions = {
        str(alpha): _next_token_payload(logits, target_ids, tokenizer, top_k=top_k)
        for alpha, logits in sorted(intervened_logits.items())
    }
    best_rank = min(item["target_rank"] for item in interventions.values())
    return {
        "clean_answers": list(clean_answers),
        "clean_answer_variants": [asdict(variant) for variant in expected_variants],
        "target_answers": list(target_answers),
        "target_variants": [asdict(variant) for variant in target_variants],
        "clean": clean,
        "interventions": interventions,
        "best_intervened_rank": best_rank,
        "improved": best_rank < clean["target_rank"],
        "target_top1": best_rank == 1,
    }


def aggregate_capability_checks(
    read_results: Sequence[Mapping[str, Any]],
    swap_results: Sequence[Mapping[str, Any]],
    *,
    minimum_improvements: int = 3,
) -> tuple[dict[str, bool], list[str]]:
    clean_baselines = all(
        bool(case["checks"]["baseline_top1"]) for case in read_results
    )
    spider = next((case for case in read_results if case["key"] == "spider"), None)
    spider_read = bool(spider and spider["checks"].get("read_capability", False))
    improved_count = sum(bool(case["improved"]) for case in swap_results)
    top1_count = sum(bool(case["target_top1"]) for case in swap_results)
    checks = {
        "clean_baselines": clean_baselines,
        "spider_read": spider_read,
        "swap_rank_improvements": improved_count >= minimum_improvements,
        "swap_target_top1": top1_count >= 1,
    }
    failures: list[str] = []
    if not clean_baselines:
        failures.append("one or more clean baseline answers are not top-1")
    if not spider_read:
        failures.append("spider readout did not satisfy the Qwen capability gate")
    if not checks["swap_rank_improvements"]:
        failures.append(
            f"coordinate swaps improved {improved_count}/{len(swap_results)} "
            f"target ranks; need at least {minimum_improvements}"
        )
    if not checks["swap_target_top1"]:
        failures.append("no coordinate swap placed its target answer at top-1")
    return checks, failures


def workspace_loading(
    activations_by_layer: Mapping[int, torch.Tensor],
    vectors_by_layer: Mapping[int, torch.Tensor],
    *,
    positions: Sequence[int],
) -> float:
    similarities = []
    for layer, vector in vectors_by_layer.items():
        hidden = activations_by_layer[layer][0, list(positions)].float()
        direction = vector.to(hidden.device, dtype=torch.float32).expand_as(hidden)
        similarities.append(
            torch.nn.functional.cosine_similarity(hidden, direction, dim=-1)
        )
    return float(torch.cat(similarities).mean().item())


def workspace_layers(n_layers: int, source_layers: Iterable[int]) -> list[int]:
    lower = math.ceil(0.35 * n_layers)
    upper = math.floor(0.80 * n_layers)
    return [layer for layer in source_layers if lower <= layer <= upper]


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value


def write_results(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_model_lens(model: Any, lens: Any) -> None:
    if model.d_model != lens.d_model:
        raise ValueError(
            f"Model/lens residual width mismatch: {model.d_model} != {lens.d_model}"
        )
    invalid = [layer for layer in lens.source_layers if not 0 <= layer < model.n_layers]
    if invalid:
        raise ValueError(
            f"Lens fitted layers {invalid} are outside model depth {model.n_layers}"
        )


def _summarize_lens(
    logits_by_layer: Mapping[int, torch.Tensor],
    *,
    layers: Sequence[int],
    positions: Sequence[int],
    target_ids: Sequence[int],
) -> dict[str, int]:
    candidates = [
        (
            best_target_rank(logits_by_layer[layer][position], target_ids),
            layer,
            position,
        )
        for layer in layers
        for position in positions
    ]
    rank, layer, position = min(candidates)
    return {"best_rank": rank, "layer": layer, "position": position}


def _readout_payload(
    logits_by_layer: Mapping[int, torch.Tensor],
    tokenizer: Any,
    target_ids: Sequence[int],
    *,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(layer): [
            {
                "position": position,
                "target_rank": best_target_rank(position_logits, target_ids),
                "top_tokens": top_tokens(position_logits, tokenizer, k=top_k),
            }
            for position, position_logits in enumerate(layer_logits)
        ]
        for layer, layer_logits in sorted(logits_by_layer.items())
    }


def analyze_case(
    case: ReadoutCase,
    *,
    model: Any,
    lens: Any,
    tokenizer: Any,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    validate_model_lens(model, lens)
    jacobian_logits, model_logits, input_ids = lens.apply(
        model, case.prompt, positions=None
    )
    logit_logits, repeated_model_logits, repeated_input_ids = lens.apply(
        model, case.prompt, positions=None, use_jacobian=False
    )
    if not torch.equal(input_ids, repeated_input_ids):
        raise RuntimeError("J-Lens and logit-lens tokenization differed")
    if not torch.equal(model_logits, repeated_model_logits):
        raise RuntimeError("J-Lens and logit-lens baseline logits differed")

    target_variants = concept_token_variants(tokenizer, case.target_concepts)
    target_ids = tuple(variant.token_id for variant in target_variants)
    baseline_top1_id = int(model_logits[-1].argmax().item())
    scored_positions = (
        list(range(input_ids.shape[1]))
        if case.literal_argument is None
        else positions_after_literal(tokenizer, input_ids, case.literal_argument)
    )
    scored_layers = workspace_layers(model.n_layers, lens.source_layers)
    if not scored_layers:
        raise ValueError("No fitted layers fall inside the workspace range")

    summaries = {
        "jacobian_lens": _summarize_lens(
            jacobian_logits,
            layers=scored_layers,
            positions=scored_positions,
            target_ids=target_ids,
        ),
        "logit_lens": _summarize_lens(
            logit_logits,
            layers=scored_layers,
            positions=scored_positions,
            target_ids=target_ids,
        ),
    }
    checks: dict[str, bool] = {}
    if case.key == "spider":
        jacobian_rank = summaries["jacobian_lens"]["best_rank"]
        logit_rank = summaries["logit_lens"]["best_rank"]
        checks.update(
            {
                "paper_top1_hit": jacobian_rank == 1,
                "read_capability": jacobian_rank <= 5 and jacobian_rank < logit_rank,
            }
        )
    return {
        "key": case.key,
        "prompt": case.prompt,
        "expected_answers": list(case.expected_answers),
        "target_concepts": list(case.target_concepts),
        "input_ids": input_ids[0].tolist(),
        "input_tokens": [
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            for token_id in input_ids[0].tolist()
        ],
        "target_variants": [asdict(variant) for variant in target_variants],
        "baseline": {
            "top1_id": baseline_top1_id,
            "top1_token": tokenizer.decode(
                [baseline_top1_id], clean_up_tokenization_spaces=False
            ),
            "top_tokens": top_tokens(model_logits[-1], tokenizer, k=top_k),
        },
        "workspace_layers": scored_layers,
        "scored_positions": scored_positions,
        "summary": summaries,
        "readouts": {
            "jacobian_lens": _readout_payload(
                jacobian_logits, tokenizer, target_ids, top_k=top_k
            ),
            "logit_lens": _readout_payload(
                logit_logits, tokenizer, target_ids, top_k=top_k
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def analyze_swap_case(
    case: SwapCase,
    *,
    read_case: ReadoutCase,
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    layers: Sequence[int],
    alphas: Sequence[float],
    top_k: int,
) -> dict[str, Any]:
    source = single_token_surface(tokenizer, case.source_surface)
    target = single_token_surface(tokenizer, case.target_surface)
    input_ids = model.encode(read_case.prompt)
    scoring_input, formatting_prefix = prepare_scoring_input(
        input_ids,
        forward_next_token=forward_next_token,
        tokenizer=tokenizer,
    )
    vectors_by_layer = {
        layer: (
            jlens_vector(
                lens,
                unembedding_weight,
                layer=layer,
                token_id=source.token_id,
            ),
            jlens_vector(
                lens,
                unembedding_weight,
                layer=layer,
                token_id=target.token_id,
            ),
        )
        for layer in layers
    }
    loading = None
    if read_case.literal_argument is not None:
        with (
            torch.inference_mode(),
            ActivationRecorder(
                model.layers,
                at=layers,
            ) as recorder,
        ):
            forward_next_token(input_ids)
        loading = workspace_loading(
            recorder.activations,
            {layer: vectors_by_layer[layer][0] for layer in layers},
            positions=positions_from_literal(
                tokenizer,
                input_ids,
                read_case.literal_argument,
            ),
        )
    with torch.inference_mode():
        clean_logits = forward_next_token(scoring_input)
        intervened_logits: dict[float, torch.Tensor] = {}
        for alpha in alphas:
            with LensCoordinateSwapper(model.layers, vectors_by_layer, alpha=alpha):
                intervened_logits[alpha] = forward_next_token(scoring_input)

    summary = summarize_swap_logits(
        clean_logits,
        intervened_logits,
        clean_answers=read_case.expected_answers,
        target_answers=case.target_answers,
        tokenizer=tokenizer,
        top_k=top_k,
    )
    return {
        "key": case.key,
        "prompt": read_case.prompt,
        "source": asdict(source),
        "target": asdict(target),
        "formatting_prefix": formatting_prefix,
        "workspace_loading": loading,
        "workspace_layers": list(layers),
        **summary,
    }


def run_readout_sanity(
    *,
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    cases: Sequence[ReadoutCase] = READOUT_CASES,
    swap_cases: Sequence[SwapCase] = SWAP_CASES,
    alphas: Sequence[float] = (1.0, 2.0),
    minimum_improvements: int = 3,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    validate_model_lens(model, lens)
    layers = workspace_layers(model.n_layers, lens.source_layers)
    if not layers:
        raise ValueError("No fitted layers fall inside the workspace range")

    read_results = [
        analyze_case(case, model=model, lens=lens, tokenizer=tokenizer, top_k=top_k)
        for case in cases
    ]
    read_cases_by_key = {case.key: case for case in cases}
    swap_results = [
        analyze_swap_case(
            swap_case,
            read_case=read_cases_by_key[swap_case.key],
            model=model,
            lens=lens,
            tokenizer=tokenizer,
            unembedding_weight=unembedding_weight,
            forward_next_token=forward_next_token,
            layers=layers,
            alphas=alphas,
            top_k=top_k,
        )
        for swap_case in swap_cases
    ]
    swaps_by_key = {case["key"]: case for case in swap_results}
    for read_result in read_results:
        swap_result = swaps_by_key[read_result["key"]]
        read_result["baseline"]["formatting_prefix"] = swap_result["formatting_prefix"]
        read_result["baseline"]["expected_rank"] = swap_result["clean"]["expected_rank"]
        read_result["checks"]["baseline_top1"] = swap_result["clean"]["expected_top1"]
        read_result["passed"] = all(read_result["checks"].values())

    checks, failures = aggregate_capability_checks(
        read_results,
        swap_results,
        minimum_improvements=minimum_improvements,
    )
    return {
        "model": MODEL_NAME,
        "lens": {
            "repo": LENS_REPO,
            "revision": LENS_REVISION,
            "file": LENS_FILE,
            "n_prompts": lens.n_prompts,
            "d_model": lens.d_model,
            "source_layers": list(lens.source_layers),
        },
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "top_k": top_k,
        "intervention_strengths": list(alphas),
        "cases": read_results,
        "swaps": swap_results,
        "checks": checks,
        "failures": failures,
        "passed": all(checks.values()),
    }
