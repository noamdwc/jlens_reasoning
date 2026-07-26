# J-Lens Readout Sanity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Colab-GPU experiment that loads the released Qwen3.5-4B Jacobian lens and checks readouts for the paper's spider and France examples without causal intervention.

**Architecture:** A pure experiment module owns prompt definitions, token-variant resolution, target ranking, model/lens validation, case analysis, and JSON serialization. A Colab notebook performs authenticated model/lens loading, delegates analysis to that module, saves reproducible artifacts, and renders two upstream interactive slices. CPU tests exercise all deterministic analysis behavior with synthetic tensors and fake tokenizers; no automated test downloads weights.

**Tech Stack:** Python 3.11, PyTorch, Transformers, Hugging Face Hub, Anthropic `jlens`, Google Colab, `nbformat`, pytest, Ruff.

**Design reference:** `docs/superpowers/specs/2026-07-14-jlens-readout-sanity-design.md`

---

## File Structure

- Create `src/jlens_reasoning/experiments/__init__.py`: mark the experiment package and re-export no mutable state.
- Create `src/jlens_reasoning/experiments/readout_sanity.py`: own released artifact coordinates, prompt cases, pure ranking/span helpers, compatibility validation, case execution, and JSON writing.
- Create `tests/test_readout_sanity.py`: CPU-only tests for the experiment module, including fake lens/model integration.
- Create `notebooks/01_jlens_readout_sanity.ipynb`: orchestrate the real Colab GPU run and render interactive slices.
- Modify `tests/test_notebooks.py`: include and validate the experiment notebook.
- Modify `README.md`: document how to launch the readout sanity run and interpret its scope.

## Task 1: Define the released artifacts and experiment cases

**Files:**

- Create: `src/jlens_reasoning/experiments/__init__.py`
- Create: `src/jlens_reasoning/experiments/readout_sanity.py`
- Create: `tests/test_readout_sanity.py`

- [ ] **Step 1: Write failing tests for immutable experiment definitions**

Create `tests/test_readout_sanity.py` with:

```python
from jlens_reasoning.experiments.readout_sanity import (
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    READOUT_CASES,
)


def test_released_artifact_coordinates_match_upstream_walkthrough() -> None:
    assert MODEL_NAME == "Qwen/Qwen3.5-4B"
    assert LENS_REPO == "neuronpedia/jacobian-lens"
    assert LENS_REVISION == "qwen-n1000"
    assert LENS_FILE.endswith("Qwen3.5-4B_jacobian_lens_n1000.pt")


def test_cases_cover_exact_spider_and_france_prompts() -> None:
    cases = {case.key: case for case in READOUT_CASES}

    assert cases["spider"].prompt == (
        "The number of legs on the animal that spins webs is"
    )
    assert cases["spider"].expected_answers == ("8", "eight")
    assert cases["spider"].target_concepts == ("spider",)
    assert cases["spider"].literal_argument is None

    france = [case for case in READOUT_CASES if case.key.startswith("france_")]
    assert len(france) == 4
    assert {case.expected_answers[0] for case in france} == {
        "Paris",
        "French",
        "Europe",
        "Euro",
    }
    assert all(case.target_concepts == ("France",) for case in france)
    assert all(case.literal_argument == "France" for case in france)
```

- [ ] **Step 2: Run the definition tests and verify they fail**

Run:

```bash
uv run pytest tests/test_readout_sanity.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jlens_reasoning.experiments'`.

- [ ] **Step 3: Add the experiment package and definitions**

Create `src/jlens_reasoning/experiments/__init__.py`:

```python
"""Reproducible Jacobian Lens experiments."""
```

Create the start of `src/jlens_reasoning/experiments/readout_sanity.py`:

```python
"""Readout-only sanity checks for the public Qwen Jacobian lens."""

from __future__ import annotations

from dataclasses import dataclass

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
TOP_K = 25


@dataclass(frozen=True, slots=True)
class ReadoutCase:
    key: str
    prompt: str
    expected_answers: tuple[str, ...]
    target_concepts: tuple[str, ...]
    literal_argument: str | None = None


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
```

- [ ] **Step 4: Run the definition tests**

Run:

```bash
uv run pytest tests/test_readout_sanity.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit the experiment definitions**

```bash
git add src/jlens_reasoning/experiments/__init__.py src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_sanity.py
git commit -m "feat: define J-Lens readout sanity cases"
```

## Task 2: Implement deterministic token and rank analysis

**Files:**

- Modify: `src/jlens_reasoning/experiments/readout_sanity.py`
- Modify: `tests/test_readout_sanity.py`

- [ ] **Step 1: Add failing helper tests**

Move the new standard-library and third-party imports into the existing import
block at the top of `tests/test_readout_sanity.py`, extend the existing
`readout_sanity` import, and append the fake tokenizer and tests:

```python
import json
from pathlib import Path

import torch

from jlens_reasoning.experiments.readout_sanity import (
    best_target_rank,
    concept_token_variants,
    find_last_subsequence,
    positions_after_literal,
    top_tokens,
    workspace_layers,
    write_results,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.pieces = {
            "France": [7],
            " France": [17],
            "france": [8],
            " france": [18],
            "FRANCE": [9, 10],
            " FRANCE": [19, 20],
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.pieces.get(text, [99, 100])

    def decode(
        self, token_ids: list[int], *, clean_up_tokenization_spaces: bool = False
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return f"token-{token_ids[0]}"


def test_concept_variants_keep_single_tokens_and_deduplicate() -> None:
    variants = concept_token_variants(FakeTokenizer(), ("France", "france"))

    assert [(variant.token_id, variant.surface) for variant in variants] == [
        (7, "France"),
        (17, " France"),
        (8, "france"),
        (18, " france"),
    ]


def test_find_last_subsequence_and_positions_after_literal() -> None:
    assert find_last_subsequence([1, 17, 2, 17, 3], ([7], [17])) == (3, 4)
    assert positions_after_literal(
        FakeTokenizer(), torch.tensor([[1, 17, 2, 3]]), "France"
    ) == [2, 3]


def test_rank_is_one_based_best_variant_and_stable_for_ties() -> None:
    logits = torch.tensor([0.0, 3.0, 3.0, 1.0])

    assert best_target_rank(logits, (2, 3)) == 2
    assert best_target_rank(logits, (3,)) == 3


def test_top_tokens_preserve_token_ids_and_logits() -> None:
    assert top_tokens(torch.tensor([0.0, 2.0, 1.0]), FakeTokenizer(), k=2) == [
        {"token_id": 1, "token": "token-1", "logit": 2.0},
        {"token_id": 2, "token": "token-2", "logit": 1.0},
    ]


def test_workspace_layers_use_inclusive_ceil_and_floor_bounds() -> None:
    assert workspace_layers(20, range(20)) == list(range(7, 17))
    assert workspace_layers(20, [0, 6, 7, 12, 16, 17, 19]) == [7, 12, 16]


def test_results_round_trip_without_tensors(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    write_results(output, {"rank": torch.tensor(3), "layers": torch.tensor([7, 8])})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "layers": [7, 8],
        "rank": 3,
    }
```

- [ ] **Step 2: Run the helper tests and verify they fail**

Run:

```bash
uv run pytest tests/test_readout_sanity.py -v
```

Expected: collection FAILS because the helper functions do not exist.

- [ ] **Step 3: Implement token variants, spans, ranks, and serialization**

Merge these imports into the import block at the top of `readout_sanity.py`,
change the existing dataclass import to `from dataclasses import asdict,
dataclass`, and add the definitions after `READOUT_CASES`:

```python
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class TokenVariant:
    token_id: int
    surface: str


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
```

- [ ] **Step 4: Run helper tests and lint**

Run:

```bash
uv run pytest tests/test_readout_sanity.py -v
uv run ruff check src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_sanity.py
```

Expected: all tests PASS and Ruff reports no errors.

- [ ] **Step 5: Commit deterministic analysis helpers**

```bash
git add src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_sanity.py
git commit -m "feat: analyze J-Lens token ranks"
```

## Task 3: Add model/lens validation and the readout runner

**Files:**

- Modify: `src/jlens_reasoning/experiments/readout_sanity.py`
- Modify: `tests/test_readout_sanity.py`

- [ ] **Step 1: Write failing validation and integration tests**

Merge `SimpleNamespace`, `pytest`, and the new experiment symbols into the
existing top-level imports, then append the test doubles and tests:

```python
from types import SimpleNamespace

import pytest

from jlens_reasoning.experiments.readout_sanity import (
    ReadoutCase,
    analyze_case,
    run_readout_sanity,
    validate_model_lens,
)


class RunnerTokenizer(FakeTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.pieces.update(
            {
                "spider": [2],
                " spider": [2],
                "Spider": [2],
                " Spider": [2],
                "SPIDER": [2],
                " SPIDER": [2],
                "8": [4],
                " 8": [4],
                "eight": [4],
                " eight": [4],
                "Eight": [4],
                " Eight": [4],
                "EIGHT": [4],
                " EIGHT": [4],
            }
        )


class FakeLens:
    d_model = 4
    source_layers = [0, 1, 2, 3]
    n_prompts = 1000

    def apply(self, model, prompt, *, use_jacobian=True, **kwargs):
        del model, prompt, kwargs
        input_ids = torch.tensor([[0, 1, 3]])
        model_logits = torch.zeros(3, 6)
        model_logits[-1, 4] = 9.0
        layer_logits = {}
        for layer in self.source_layers:
            logits = torch.zeros(3, 6)
            logits[:, 2] = 8.0 if use_jacobian and layer == 2 else -1.0
            layer_logits[layer] = logits
        return layer_logits, model_logits, input_ids


def test_validate_model_lens_rejects_width_and_layer_mismatches() -> None:
    with pytest.raises(ValueError, match="residual width"):
        validate_model_lens(SimpleNamespace(n_layers=4, d_model=5), FakeLens())

    lens = FakeLens()
    lens.source_layers = [0, 4]
    with pytest.raises(ValueError, match="fitted layers"):
        validate_model_lens(SimpleNamespace(n_layers=4, d_model=4), lens)


def test_analyze_case_grades_baseline_and_spider_readout() -> None:
    case = ReadoutCase(
        key="spider",
        prompt="prompt",
        expected_answers=("8", "eight"),
        target_concepts=("spider",),
    )

    result = analyze_case(
        case,
        model=SimpleNamespace(n_layers=4, d_model=4),
        lens=FakeLens(),
        tokenizer=RunnerTokenizer(),
        top_k=3,
    )

    assert result["checks"] == {"baseline_top1": True, "target_top_k": True}
    assert result["summary"]["jacobian_lens"]["best_rank"] == 1
    assert result["summary"]["jacobian_lens"]["layer"] == 2
    assert result["summary"]["logit_lens"]["best_rank"] > 1
    assert result["passed"] is True


def test_run_readout_sanity_keeps_failed_case_details() -> None:
    case = ReadoutCase(
        key="wrong-baseline",
        prompt="prompt",
        expected_answers=("missing",),
        target_concepts=("spider",),
    )
    tokenizer = RunnerTokenizer()
    tokenizer.pieces["missing"] = [5]
    tokenizer.pieces[" missing"] = [5]

    result = run_readout_sanity(
        model=SimpleNamespace(n_layers=4, d_model=4),
        lens=FakeLens(),
        tokenizer=tokenizer,
        cases=(case,),
        top_k=3,
    )

    assert result["passed"] is False
    assert result["cases"][0]["checks"]["baseline_top1"] is False
    assert result["failures"] == ["wrong-baseline: baseline top-1 mismatch"]
```

- [ ] **Step 2: Run the runner tests and verify they fail**

Run:

```bash
uv run pytest tests/test_readout_sanity.py -v
```

Expected: collection FAILS because the validation and runner functions do not exist.

- [ ] **Step 3: Implement compatibility validation and case analysis**

Append to `readout_sanity.py`:

```python
def validate_model_lens(model: Any, lens: Any) -> None:
    if model.d_model != lens.d_model:
        raise ValueError(
            f"Model/lens residual width mismatch: {model.d_model} != {lens.d_model}"
        )
    invalid = [
        layer for layer in lens.source_layers if not 0 <= layer < model.n_layers
    ]
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
        (best_target_rank(logits_by_layer[layer][position], target_ids), layer, position)
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
    answer_variants = concept_token_variants(tokenizer, case.expected_answers)
    answer_ids = {variant.token_id for variant in answer_variants}
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
    checks = {
        "baseline_top1": baseline_top1_id in answer_ids,
        "target_top_k": summaries["jacobian_lens"]["best_rank"] <= top_k,
    }
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


def run_readout_sanity(
    *,
    model: Any,
    lens: Any,
    tokenizer: Any,
    cases: Sequence[ReadoutCase] = READOUT_CASES,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    validate_model_lens(model, lens)
    case_results = [
        analyze_case(
            case, model=model, lens=lens, tokenizer=tokenizer, top_k=top_k
        )
        for case in cases
    ]
    failures: list[str] = []
    for case in case_results:
        if not case["checks"]["baseline_top1"]:
            failures.append(f"{case['key']}: baseline top-1 mismatch")
        if not case["checks"]["target_top_k"]:
            failures.append(f"{case['key']}: target outside J-Lens top {top_k}")
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
        "cases": case_results,
        "failures": failures,
        "passed": not failures,
    }
```

- [ ] **Step 4: Run runner tests and the complete CPU suite**

Run:

```bash
uv run pytest tests/test_readout_sanity.py -v
uv run pytest
```

Expected: all new tests PASS and the complete existing suite remains green.

- [ ] **Step 5: Commit the readout runner**

```bash
git add src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_sanity.py
git commit -m "feat: run readout-only J-Lens checks"
```

## Task 4: Add and structurally verify the Colab notebook

**Files:**

- Create: `notebooks/01_jlens_readout_sanity.ipynb`
- Modify: `tests/test_notebooks.py`

- [ ] **Step 1: Write failing notebook-structure tests**

Add `Path("notebooks/01_jlens_readout_sanity.ipynb")` to `NOTEBOOKS` in `tests/test_notebooks.py`, then append:

```python
def test_readout_sanity_notebook_has_pinned_gpu_workflow() -> None:
    notebook = load_notebook(Path("notebooks/01_jlens_readout_sanity.ipynb"))
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "initialize_colab(enable_wandb=False, require_cuda=True)" in source
    assert 'MODEL_NAME = "Qwen/Qwen3.5-4B"' not in source
    assert "from jlens_reasoning.experiments.readout_sanity import" in source
    assert "JacobianLens.from_pretrained" in source
    assert "run_readout_sanity" in source
    assert "write_results" in source
    assert "compute_slice" in source
    assert "mode=\"embed\"" in source
    assert "raise RuntimeError" in source
```

- [ ] **Step 2: Run the notebook tests and verify they fail**

Run:

```bash
uv run pytest tests/test_notebooks.py -v
```

Expected: FAIL because `notebooks/01_jlens_readout_sanity.ipynb` does not exist.

- [ ] **Step 3: Create the notebook with the canonical loader and exact cells**

Create `notebooks/01_jlens_readout_sanity.ipynb` as nbformat 4. Copy the first
cell from `notebooks/_template.ipynb` byte-for-byte. Add these code cells, each
with `execution_count: null` and `outputs: []`:

Initialization cell:

```python
from jlens_reasoning.environments.colab import initialize_colab

context = initialize_colab(enable_wandb=False, require_cuda=True)
context
```

Imports and model/lens loading cell:

```python
import importlib.metadata
import subprocess

import jlens
import torch
import transformers

from jlens_reasoning.experiments.readout_sanity import (
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    READOUT_CASES,
    concept_token_variants,
    run_readout_sanity,
    validate_model_lens,
    write_results,
)

hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
).to(context.device)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf_model, tokenizer)
lens = jlens.JacobianLens.from_pretrained(
    LENS_REPO,
    filename=LENS_FILE,
    revision=LENS_REVISION,
)
validate_model_lens(model, lens)
model, lens
```

Run and persist cell:

```python
result = run_readout_sanity(model=model, lens=lens, tokenizer=tokenizer)
result["provenance"] = {
    "project_commit": subprocess.run(
        ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "jlens": importlib.metadata.version("jlens"),
}

run_dir = context.runs_dir / "jlens-readout-sanity"
result_path = run_dir / "result.json"
write_results(result_path, result)

for case in result["cases"]:
    summary = case["summary"]["jacobian_lens"]
    print(
        case["key"],
        f"baseline={case['baseline']['top1_token']!r}",
        f"best_rank={summary['best_rank']}",
        f"layer={summary['layer']}",
        f"position={summary['position']}",
        f"passed={case['passed']}",
    )
print(f"Saved: {result_path}")
```

Interactive-slice cell:

```python
from IPython.display import display
from jlens.vis import build_page, compute_slice, notebook_iframe

for case in (READOUT_CASES[0], READOUT_CASES[1]):
    pinned = {
        variant.token_id
        for variant in concept_token_variants(tokenizer, case.target_concepts)
    }
    slice_data = compute_slice(
        model,
        lens,
        case.prompt,
        top_n=25,
        pinned_token_ids=pinned,
        mask_display=True,
    )
    page, _, _ = build_page(
        slice_data,
        case.prompt,
        title=f"J-Lens readout sanity: {case.key}",
        description="Readout-only open-model sanity check.",
        mode="embed",
    )
    html_path = run_dir / f"{case.key}.html"
    html_path.write_text(page, encoding="utf-8")
    print(f"Saved: {html_path}")
    display(notebook_iframe(page))
```

Final grading cell:

```python
if not result["passed"]:
    raise RuntimeError("Readout sanity checks failed: " + "; ".join(result["failures"]))

print("All J-Lens readout sanity checks passed.")
```

- [ ] **Step 4: Run notebook tests and validate notebook JSON**

Run:

```bash
uv run pytest tests/test_notebooks.py -v
uv run python -m json.tool notebooks/01_jlens_readout_sanity.ipynb >/dev/null
```

Expected: notebook tests PASS and `json.tool` exits 0.

- [ ] **Step 5: Commit the experiment notebook**

```bash
git add notebooks/01_jlens_readout_sanity.ipynb tests/test_notebooks.py
git commit -m "feat: add Colab J-Lens readout notebook"
```

## Task 5: Document the run and perform local verification

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Add the readout sanity instructions**

Append this section to `README.md`:

````markdown
## J-Lens readout sanity experiment

`notebooks/01_jlens_readout_sanity.ipynb` is the first model-backed experiment.
Open it through the IDE's Colab integration with a GPU runtime and run all cells.
It uses the released `Qwen/Qwen3.5-4B` Jacobian lens, disables W&B, and writes
results beneath:

```text
runs/jlens-readout-sanity/
├── result.json
├── spider.html
└── france_capital.html
```

The experiment checks whether the J-Lens surfaces the unspoken `spider`
intermediate and preserves `France` after the argument token across four factual
operations. It is a readout-only open-model sanity check, not a reproduction of
the paper's causal spider→ant or France→China swaps.
````

- [ ] **Step 2: Run focused and full verification**

Run:

```bash
uv sync --locked --extra experiment
uv run pytest tests/test_readout_sanity.py tests/test_notebooks.py -v
uv run pytest
uv run ruff format --check .
uv run ruff check .
git diff --check
```

Expected: locked sync succeeds; all tests pass; both Ruff commands and
`git diff --check` exit 0. No command downloads a model or lens.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain J-Lens readout sanity run"
```

## Task 6: Execute and validate the real Colab GPU run

**Files:**

- Runtime artifacts only: `runs/jlens-readout-sanity/result.json`
- Runtime artifacts only: `runs/jlens-readout-sanity/spider.html`
- Runtime artifacts only: `runs/jlens-readout-sanity/france_capital.html`

- [ ] **Step 1: Launch the notebook on a CUDA Colab runtime**

Open `notebooks/01_jlens_readout_sanity.ipynb` through the IDE's Colab
integration, set `PROJECT_REF` to the implementation commit, and run all cells.

Expected: initialization reports a CUDA device; Qwen3.5-4B and the pinned
`qwen-n1000` lens load from Hugging Face; the notebook reaches the summary cell.

- [ ] **Step 2: Inspect the saved machine-readable result**

Confirm `result.json` contains five cases, complete per-layer/per-position
readouts for both lens types, package provenance, and `passed: true`.

Expected: every baseline is top-1 correct, `spider` reaches J-Lens top 25 in a
workspace-range layer, and `France` reaches J-Lens top 25 strictly after its
literal span in each France prompt.

- [ ] **Step 3: Inspect the two interactive slices**

Open `spider.html` and `france_capital.html` from the persistent artifact
directory and pin the tracked concept tokens.

Expected: both pages render without an external model process, show the complete
position × layer grid, and their rank charts agree with `result.json`.

- [ ] **Step 4: Record genuine open-model deviations without weakening checks**

If the notebook saves artifacts but ends with a qualitative failure, preserve
`result.json`, quote the failing case and its best rank/layer/position, and open
the corresponding readout data before changing prompts, layer bounds, or the
top-25 criterion. Any threshold or prompt change requires a spec amendment.

- [ ] **Step 5: Report the verified milestone**

Report the project commit, artifact paths, baseline answers, best J-Lens ranks,
and whether each case passed. State explicitly that causal swaps remain deferred.
