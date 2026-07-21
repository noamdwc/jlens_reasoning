# Experiment Package Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current monolithic `jlens_reasoning.experiments` package with reusable `jlens_reasoning.experiments_utils` mechanics and a self-contained, importable `experiments.jlens_readout_sanity` package beside its Colab notebook.

**Architecture:** Move policy-free token, artifact, validation, intervention, and deterministic-control primitives into `src/jlens_reasoning/experiments_utils`. Move all fixed J-Lens prompts, thresholds, result assembly, control orchestration, and notebook-facing APIs into `experiments/jlens_readout_sanity`, package both roots in the wheel, and remove the old import paths without compatibility shims. Keep the notebook explicit: setup, model loading, execution, saving, reporting, visualization, and final gating remain independently rerunnable cells.

**Tech Stack:** Python 3.11, setuptools, PyTorch, `jlens`, Transformers tokenizer interfaces, pytest, nbformat, Ruff, uv.

**Design reference:** `docs/superpowers/specs/2026-07-21-experiment-package-architecture-design.md`

---

## File Structure

### Create

- `src/jlens_reasoning/experiments_utils/__init__.py`: package marker only; consumers import focused submodules directly.
- `src/jlens_reasoning/experiments_utils/artifacts.py`: JSON conversion and stable result writing.
- `src/jlens_reasoning/experiments_utils/tokens.py`: token variants, ranks, token spans, top-token payloads, and formatting-prefix preparation.
- `src/jlens_reasoning/experiments_utils/interventions.py`: J-Lens vector construction, coordinate swapping, hook ownership, and the common executor.
- `src/jlens_reasoning/experiments_utils/controls.py`: deterministic control math, random-vector generation, exclusions, target selection, and generic exact-key validation.
- `src/jlens_reasoning/experiments_utils/validation.py`: model/lens compatibility, workspace layers, and workspace loading.
- `experiments/__init__.py`: importable top-level experiment namespace with no registry.
- `experiments/jlens_readout_sanity/__init__.py`: experiment package marker with no eager heavyweight imports.
- `experiments/jlens_readout_sanity/constants.py`: all fixed J-Lens policy and artifact coordinates.
- `experiments/jlens_readout_sanity/types.py`: case, resolved-case, token, and intervention-context dataclasses.
- `experiments/jlens_readout_sanity/runner.py`: per-case readout/swap analysis, complete experiment orchestration, and result assembly.
- `experiments/jlens_readout_sanity/controls.py`: the four experiment-specific controls and global gate aggregation.
- `experiments/jlens_readout_sanity/utils.py`: small notebook-facing facade.
- `experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`: moved and import-updated Colab notebook.
- `tests/experiments_utils/test_artifacts.py`
- `tests/experiments_utils/test_tokens.py`
- `tests/experiments_utils/test_interventions.py`
- `tests/experiments_utils/test_controls.py`
- `tests/experiments_utils/test_validation.py`
- `tests/experiments/jlens_readout_sanity/test_constants.py`
- `tests/experiments/jlens_readout_sanity/test_runner.py`
- `tests/experiments/jlens_readout_sanity/test_controls.py`
- `tests/experiments/jlens_readout_sanity/test_package.py`
- `tests/test_package_discovery.py`

### Modify

- `pyproject.toml`: discover both `src/jlens_reasoning*` and root `experiments*` packages.
- `tests/test_notebooks.py`: keep shared notebooks explicit and discover experiment notebooks automatically.
- `tests/test_imports.py`: assert the new shared and experiment packages import.
- `README.md`: update only the J-Lens notebook path; preserve the pre-existing unstaged evaluation-policy edit.

### Delete after migration

- `src/jlens_reasoning/experiments/__init__.py`
- `src/jlens_reasoning/experiments/intervention_utils.py`
- `src/jlens_reasoning/experiments/readout_cases.py`
- `src/jlens_reasoning/experiments/readout_constants.py`
- `src/jlens_reasoning/experiments/readout_controls.py`
- `src/jlens_reasoning/experiments/readout_sanity.py`
- `src/jlens_reasoning/experiments/readout_utils.py`
- `src/jlens_reasoning/experiments/sanity_constants.py`
- `src/jlens_reasoning/experiments/sanity_controls.py`
- `notebooks/01_jlens_readout_sanity.ipynb`
- `tests/test_experiment_constants.py`
- `tests/test_readout_module_structure.py`
- `tests/test_readout_sanity.py`
- `tests/test_sanity_controls.py`

---

## Task 1: Establish dual-root package discovery

**Files:**

- Create: `experiments/__init__.py`
- Create: `experiments/jlens_readout_sanity/__init__.py`
- Create: `src/jlens_reasoning/experiments_utils/__init__.py`
- Create: `tests/test_package_discovery.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_imports.py`

- [ ] **Step 1: Write failing package-discovery tests**

Create `tests/test_package_discovery.py`:

```python
from pathlib import Path
import tomllib

from setuptools.config.expand import find_packages


ROOT = Path(__file__).resolve().parents[1]


def test_setuptools_discovers_library_and_experiment_packages() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    find_config = config["tool"]["setuptools"]["packages"]["find"]

    assert find_config == {
        "where": ["src", "."],
        "include": ["jlens_reasoning*", "experiments*"],
        "namespaces": False,
    }
    discovered = set(
        find_packages(
            where=find_config["where"],
            include=find_config["include"],
            namespaces=find_config["namespaces"],
            root_dir=ROOT,
        )
    )
    assert "jlens_reasoning" in discovered
    assert "jlens_reasoning.experiments_utils" in discovered
    assert "experiments" in discovered
    assert "experiments.jlens_readout_sanity" in discovered
    assert not any(name.startswith("tests") for name in discovered)
```

Append to `tests/test_imports.py`:

```python
def test_new_experiment_package_roots_import() -> None:
    import experiments
    import experiments.jlens_readout_sanity
    import jlens_reasoning.experiments_utils

    assert experiments.__name__ == "experiments"
    assert experiments.jlens_readout_sanity.__name__ == (
        "experiments.jlens_readout_sanity"
    )
    assert jlens_reasoning.experiments_utils.__name__ == (
        "jlens_reasoning.experiments_utils"
    )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_package_discovery.py tests/test_imports.py -q
```

Expected: package discovery or imports fail because neither new package root exists and `pyproject.toml` still searches only `src`.

- [ ] **Step 3: Create package markers and configure discovery**

Create the three `__init__.py` files with docstrings only:

```python
"""Reusable mechanics for model experiments."""
```

```python
"""Repository-owned experiment packages."""
```

```python
"""J-Lens read-and-change sanity experiment."""
```

Replace the setuptools discovery table in `pyproject.toml` with:

```toml
[tool.setuptools.packages.find]
where = ["src", "."]
include = ["jlens_reasoning*", "experiments*"]
namespaces = false
```

Do not add individual experiment names to `pyproject.toml`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_package_discovery.py tests/test_imports.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add pyproject.toml experiments/__init__.py experiments/jlens_readout_sanity/__init__.py src/jlens_reasoning/experiments_utils/__init__.py tests/test_package_discovery.py tests/test_imports.py
git commit -m "build: package experiment modules from both roots"
```

## Task 2: Extract artifact and token utilities

**Files:**

- Create: `src/jlens_reasoning/experiments_utils/artifacts.py`
- Create: `src/jlens_reasoning/experiments_utils/tokens.py`
- Create: `tests/experiments_utils/test_artifacts.py`
- Create: `tests/experiments_utils/test_tokens.py`

- [ ] **Step 1: Write failing artifact tests**

Move the stable result round-trip coverage from `tests/test_readout_sanity.py`
and `tests/test_sanity_controls.py` into `tests/experiments_utils/test_artifacts.py`:

```python
import json
from pathlib import Path

import torch

from jlens_reasoning.experiments_utils.artifacts import write_results


def test_write_results_is_json_ready_and_byte_stable(tmp_path: Path) -> None:
    result = {
        "rank": torch.tensor(3),
        "layers": torch.tensor([7, 8]),
        "path": Path("runs/result.json"),
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_results(first, result)
    write_results(second, result)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == {
        "layers": [7, 8],
        "path": "runs/result.json",
        "rank": 3,
    }
```

- [ ] **Step 2: Write failing token utility tests**

Create `tests/experiments_utils/test_tokens.py` by moving the existing fake
tokenizer tests for strict surfaces, concept variants, subsequences, ranks,
top tokens, and formatting prefixes. Import only these public symbols:

```python
from jlens_reasoning.experiments_utils.tokens import (
    TokenVariant,
    best_target_rank,
    concept_surfaces,
    concept_token_variants,
    find_last_subsequence,
    positions_after_literal,
    positions_from_literal,
    prepare_scoring_input,
    single_token_surface,
    top_tokens,
)
```

Keep the existing numerical expectations, and call generic helpers with
explicit policy:

```python
assert top_tokens(logits, tokenizer, k=2) == expected

scoring_input, prefix = prepare_scoring_input(
    input_ids,
    forward_next_token=forward_next_token,
    tokenizer=tokenizer,
    max_formatting_tokens=2,
)
```

Add a direct policy-independence assertion:

```python
def test_concept_surfaces_has_no_experiment_case_dependency() -> None:
    assert concept_surfaces("France") == (
        "France",
        " France",
        "france",
        " france",
        "FRANCE",
        " FRANCE",
    )
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/experiments_utils/test_artifacts.py tests/experiments_utils/test_tokens.py -q
```

Expected: collection fails because `artifacts` and `tokens` do not exist.

- [ ] **Step 4: Implement `artifacts.py`**

Move `_jsonable` and `write_results` from `readout_utils.py` without changing
serialization behavior. Keep `_jsonable` private and export only
`write_results`:

```python
__all__ = ["write_results"]
```

- [ ] **Step 5: Implement `tokens.py`**

Move and rename the policy-free token helpers as follows:

```text
readout_cases.TokenVariant             -> tokens.TokenVariant
readout_cases._concept_surfaces        -> tokens.concept_surfaces
readout_cases.single_token_surface     -> tokens.single_token_surface
readout_cases.concept_token_variants   -> tokens.concept_token_variants
readout_utils.find_last_subsequence    -> tokens.find_last_subsequence
readout_utils.positions_after_literal  -> tokens.positions_after_literal
readout_utils.positions_from_literal   -> tokens.positions_from_literal
readout_utils.best_target_rank         -> tokens.best_target_rank
readout_utils.top_tokens               -> tokens.top_tokens
intervention_utils._next_token_payload -> tokens.next_token_payload
readout_utils.prepare_scoring_input    -> tokens.prepare_scoring_input
```

Use these signatures so experiment policy is supplied by callers:

```python
def top_tokens(
    logits: torch.Tensor,
    tokenizer: Any,
    *,
    k: int,
) -> list[dict[str, Any]]:
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


def next_token_payload(
    logits: torch.Tensor,
    target_ids: Sequence[int],
    tokenizer: Any,
    *,
    top_k: int,
) -> dict[str, Any]:
    normalized = logits.detach().float().cpu()
    top1_id = int(normalized.argmax().item())
    return {
        "top1_id": top1_id,
        "top1_token": tokenizer.decode(
            [top1_id], clean_up_tokenization_spaces=False
        ),
        "target_rank": best_target_rank(normalized, target_ids),
        "top_tokens": top_tokens(normalized, tokenizer, k=top_k),
    }


def prepare_scoring_input(
    input_ids: torch.Tensor,
    *,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    tokenizer: Any,
    max_formatting_tokens: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    scoring_input = input_ids
    prefix: list[dict[str, Any]] = []
    for _ in range(max_formatting_tokens):
        logits = forward_next_token(scoring_input)
        token_id = int(logits.argmax().item())
        surface = tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False
        )
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
```

Update every internal call from `_concept_surfaces` to `concept_surfaces`.
Do not import J-Lens constants or case definitions.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/experiments_utils/test_artifacts.py tests/experiments_utils/test_tokens.py -q
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/jlens_reasoning/experiments_utils/artifacts.py src/jlens_reasoning/experiments_utils/tokens.py tests/experiments_utils/test_artifacts.py tests/experiments_utils/test_tokens.py
git commit -m "refactor: extract reusable experiment artifact and token utilities"
```

## Task 3: Extract validation and intervention mechanics

**Files:**

- Create: `src/jlens_reasoning/experiments_utils/validation.py`
- Create: `src/jlens_reasoning/experiments_utils/interventions.py`
- Create: `tests/experiments_utils/test_validation.py`
- Create: `tests/experiments_utils/test_interventions.py`

- [ ] **Step 1: Write failing validation tests**

Move the current model/lens, workspace-layer, and workspace-loading tests into
`tests/experiments_utils/test_validation.py`. Import:

```python
from jlens_reasoning.experiments_utils.validation import (
    validate_model_lens,
    workspace_layers,
    workspace_loading,
)
```

Make workspace policy explicit:

```python
def test_workspace_layers_use_caller_bounds() -> None:
    assert workspace_layers(
        20,
        range(20),
        lower_fraction=0.35,
        upper_fraction=0.80,
    ) == list(range(7, 17))
```

Preserve the existing mismatch and cosine-loading assertions.

- [ ] **Step 2: Write failing intervention tests**

Move the existing real tensor tests into
`tests/experiments_utils/test_interventions.py`. Import:

```python
from jlens_reasoning.experiments_utils.interventions import (
    LensCoordinateSwapper,
    coordinate_swap,
    execute_intervention,
    jlens_vector,
)
```

Retain tests for:

- `J[layer].T @ W_U[token]` vector construction;
- alpha 0, 1, and 2 coordinate behavior;
- shape and dtype preservation;
- patching all activation positions;
- tuple output preservation;
- hook cleanup on context exit and forward exceptions.

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/experiments_utils/test_validation.py tests/experiments_utils/test_interventions.py -q
```

Expected: collection fails because both modules are missing.

- [ ] **Step 4: Implement `validation.py`**

Move `validate_model_lens`, `workspace_loading`, and `workspace_layers` from
`readout_utils.py`. Replace imported fixed fractions with required keyword
arguments:

```python
def workspace_layers(
    n_layers: int,
    source_layers: Iterable[int],
    *,
    lower_fraction: float,
    upper_fraction: float,
) -> list[int]:
    if not 0.0 <= lower_fraction <= upper_fraction <= 1.0:
        raise ValueError("Workspace fractions must satisfy 0 <= lower <= upper <= 1")
    lower = math.ceil(lower_fraction * n_layers)
    upper = math.floor(upper_fraction * n_layers)
    return [layer for layer in source_layers if lower <= layer <= upper]
```

- [ ] **Step 5: Implement `interventions.py`**

Move these implementations unchanged except for imports and module paths:

```text
intervention_utils.jlens_vector
intervention_utils.coordinate_swap
intervention_utils.LensCoordinateSwapper
intervention_utils.execute_intervention
intervention_utils._single_token_vectors_by_layer
intervention_utils._token_vectors_by_layer
```

Promote the final two helpers to public names:

```python
single_token_vectors_by_layer
token_vectors_by_layer
```

Do not move `InterventionContext`, `analyze_identity_case`,
`summarize_swap_logits`, `_prepare_intervention_context`,
`_rank_gain_payload`, or `analyze_swap_case`; they depend on the J-Lens result
contract and move to the experiment package later.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/experiments_utils/test_validation.py tests/experiments_utils/test_interventions.py -q
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/jlens_reasoning/experiments_utils/validation.py src/jlens_reasoning/experiments_utils/interventions.py tests/experiments_utils/test_validation.py tests/experiments_utils/test_interventions.py
git commit -m "refactor: extract reusable validation and intervention mechanics"
```

## Task 4: Make deterministic control primitives policy-free

**Files:**

- Create: `src/jlens_reasoning/experiments_utils/controls.py`
- Create: `tests/experiments_utils/test_controls.py`

- [ ] **Step 1: Write failing generic control tests**

Move the pure calculations and tokenizer stub from
`tests/test_sanity_controls.py` into
`tests/experiments_utils/test_controls.py`. Import:

```python
from jlens_reasoning.experiments_utils.controls import (
    build_random_target_exclusions,
    derive_subseed,
    log_rank_gain,
    matched_random_vectors,
    mean,
    percentile,
    percentile_label,
    require_exact_case_keys,
    select_random_targets,
    strict_percentile_gate,
)
```

Define test policy locally:

```python
SEEDS = (11, 29, 47, 71, 101, 131, 167, 199, 239, 281, 331, 379, 431, 487, 547, 607)
VECTOR_NAMESPACE = "jlens-control-v1"
TARGET_NAMESPACE = "jlens-random-target-v1"
```

Update calls to provide every experiment policy value explicitly:

```python
assert derive_subseed(
    11,
    7,
    "source",
    namespace=VECTOR_NAMESPACE,
) == 3688398498245801101

generated, norms = matched_random_vectors(
    real,
    base_seed=29,
    namespace=VECTOR_NAMESPACE,
    norm_atol=1e-6,
    norm_rtol=1e-5,
    low_precision_norm_atol=1e-2,
    low_precision_norm_rtol=1e-2,
    max_attempts=1024,
)

gate = strict_percentile_gate(
    real_score,
    control_scores,
    quantile=0.95,
    interpretation="deterministic sanity check; not statistical significance",
)

selected = select_random_targets(
    tokenizer,
    excluded_ids=exclusions["all"],
    seeds=SEEDS,
    output_vocab_size=32,
    namespace=TARGET_NAMESPACE,
)
```

Add exact-key validation without fixed case names:

```python
def test_exact_case_keys_are_caller_owned() -> None:
    results = [{"key": "a"}, {"key": "b"}]
    require_exact_case_keys(results, expected_keys=("a", "b"))
    with pytest.raises(ValueError, match="Expected exact case keys"):
        require_exact_case_keys(results, expected_keys=("b", "a"))
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/experiments_utils/test_controls.py -q
```

Expected: collection fails because the generic control module does not exist.

- [ ] **Step 3: Implement generic score and percentile functions**

Move `log_rank_gain`, `mean`, `percentile`, and `percentile_label`. Implement:

```python
def strict_percentile_gate(
    real_score: float,
    control_scores: Sequence[float],
    *,
    quantile: float,
    interpretation: str,
) -> dict[str, Any]:
    threshold = percentile(control_scores, quantile)
    return {
        "real_score": float(real_score),
        "percentile": quantile,
        "threshold": threshold,
        "comparison": "real_score > threshold",
        "interpretation": interpretation,
        "passed": float(real_score) > threshold,
    }
```

- [ ] **Step 4: Implement namespaced deterministic random helpers**

Use these signatures:

```python
def derive_subseed(
    base_seed: int,
    layer_index: int,
    role: str,
    *,
    namespace: str,
) -> int:
    if role not in {"source", "target"}:
        raise ValueError("Role must be 'source' or 'target'")
    if not namespace:
        raise ValueError("Random namespace must be non-empty")
    payload = f"{namespace}:{base_seed}:{layer_index}:{role}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _norm_tolerances(
    dtype: torch.dtype,
    *,
    norm_atol: float,
    norm_rtol: float,
    low_precision_norm_atol: float,
    low_precision_norm_rtol: float,
) -> tuple[float, float]:
    if dtype in {torch.float16, torch.bfloat16}:
        return low_precision_norm_atol, low_precision_norm_rtol
    return norm_atol, norm_rtol


def _matched_random_vector(
    real_vector: torch.Tensor,
    *,
    base_seed: int,
    layer_index: int,
    role: str,
    namespace: str,
    norm_atol: float,
    norm_rtol: float,
    low_precision_norm_atol: float,
    low_precision_norm_rtol: float,
    max_attempts: int,
) -> torch.Tensor:
    real_cpu = real_vector.detach().to(device="cpu", dtype=torch.float32)
    if real_cpu.numel() == 0:
        raise ValueError("Cannot match the norm of an empty vector")
    real_norm = torch.linalg.vector_norm(real_cpu)
    if real_norm.item() == 0.0:
        return torch.zeros_like(real_vector)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        derive_subseed(
            base_seed,
            layer_index,
            role,
            namespace=namespace,
        )
    )
    atol, rtol = _norm_tolerances(
        real_vector.dtype,
        norm_atol=norm_atol,
        norm_rtol=norm_rtol,
        low_precision_norm_atol=low_precision_norm_atol,
        low_precision_norm_rtol=low_precision_norm_rtol,
    )
    for _ in range(max_attempts):
        random_vector = torch.randn(
            real_cpu.shape,
            generator=generator,
            dtype=torch.float32,
        )
        random_norm = torch.linalg.vector_norm(random_vector)
        if random_norm.item() == 0.0:
            random_vector.zero_()
            random_vector.reshape(-1)[0] = 1.0
            random_norm = torch.linalg.vector_norm(random_vector)
        matched_cpu = random_vector * (real_norm / random_norm)
        converted = matched_cpu.to(
            device=real_vector.device,
            dtype=real_vector.dtype,
        )
        converted_norm = torch.linalg.vector_norm(converted.detach().float())
        if torch.isfinite(converted).all() and math.isclose(
            real_norm.item(),
            converted_norm.item(),
            abs_tol=atol,
            rel_tol=rtol,
        ):
            return converted
    raise RuntimeError(
        "Unable to generate a finite norm-matched random vector after "
        f"{max_attempts} conversion attempts"
    )


def matched_random_vectors(
    real_vectors: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    base_seed: int,
    namespace: str,
    norm_atol: float,
    norm_rtol: float,
    low_precision_norm_atol: float,
    low_precision_norm_rtol: float,
    max_attempts: int,
) -> tuple[
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[str, dict[str, Any]]],
]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    generated: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    report: dict[str, dict[str, dict[str, Any]]] = {}
    for layer in sorted(real_vectors):
        real_pair = real_vectors[layer]
        random_pair = tuple(
            _matched_random_vector(
                real_vector,
                base_seed=base_seed,
                layer_index=layer,
                role=role,
                namespace=namespace,
                norm_atol=norm_atol,
                norm_rtol=norm_rtol,
                low_precision_norm_atol=low_precision_norm_atol,
                low_precision_norm_rtol=low_precision_norm_rtol,
                max_attempts=max_attempts,
            )
            for real_vector, role in zip(
                real_pair,
                ("source", "target"),
                strict=True,
            )
        )
        generated[layer] = random_pair
        report[str(layer)] = {}
        for real_vector, random_vector, role in zip(
            real_pair,
            random_pair,
            ("source", "target"),
            strict=True,
        ):
            real_norm = torch.linalg.vector_norm(real_vector.detach().float()).item()
            random_norm = torch.linalg.vector_norm(
                random_vector.detach().float()
            ).item()
            atol, rtol = _norm_tolerances(
                real_vector.dtype,
                norm_atol=norm_atol,
                norm_rtol=norm_rtol,
                low_precision_norm_atol=low_precision_norm_atol,
                low_precision_norm_rtol=low_precision_norm_rtol,
            )
            report[str(layer)][role] = {
                "real_norm": real_norm,
                "random_norm": random_norm,
                "atol": atol,
                "rtol": rtol,
                "matched": math.isclose(
                    real_norm,
                    random_norm,
                    abs_tol=atol,
                    rel_tol=rtol,
                ),
                "device": str(random_vector.device),
                "dtype": str(random_vector.dtype),
                "device_matches": random_vector.device == real_vector.device,
                "dtype_matches": random_vector.dtype == real_vector.dtype,
            }
    return generated, report
```

Preserve the current exact seed outputs.

- [ ] **Step 5: Implement generic token exclusions and target selection**

Move `decode_token`, `_encoded_ids`, and `build_random_target_exclusions`
without experiment imports. Add a namespace argument to selection:

```python
def select_random_targets(
    tokenizer: Any,
    *,
    excluded_ids: Iterable[int],
    seeds: Sequence[int],
    output_vocab_size: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if not namespace:
        raise ValueError("Random namespace must be non-empty")
    if output_vocab_size < 1:
        raise ValueError("Model output vocabulary must contain at least one token")
    excluded = {int(token_id) for token_id in excluded_ids}
    eligible = sorted(
        {
            int(token_id)
            for token_id in tokenizer.get_vocab().values()
            if 0 <= int(token_id) < output_vocab_size
            and int(token_id) not in excluded
        }
    )
    if not eligible:
        raise ValueError("No eligible random-target tokens remain")
    remaining = list(eligible)
    selected = []
    for seed in seeds:
        if not remaining:
            remaining = list(eligible)
        digest = hashlib.sha256(f"{namespace}:{seed}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(remaining)
        token_id = remaining.pop(index)
        selected.append(
            {
                "seed": int(seed),
                "token_id": token_id,
                "token": decode_token(tokenizer, token_id),
            }
        )
    return selected
```

Implement generic exact-key validation:

```python
def require_exact_case_keys(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_keys: Sequence[str],
) -> None:
    actual_keys = [str(result["key"]) for result in results]
    expected = list(expected_keys)
    if actual_keys != expected:
        raise ValueError(f"Expected exact case keys {expected!r}, got {actual_keys!r}")
```

Do not move `CONTROL_SEEDS`, `CONTROL_CASE_KEYS`, `CONTROL_CHECK_MAP`, fixed
tolerances, `summarize_wrong_concept`, `controls_passed`, `_control_failure`, or
`aggregate_all_checks`; those are experiment policy.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/experiments_utils/test_controls.py -q
```

Expected: all focused tests pass with the current exact deterministic IDs and
sub-seed values.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/jlens_reasoning/experiments_utils/controls.py tests/experiments_utils/test_controls.py
git commit -m "refactor: extract policy-free experiment control primitives"
```

## Task 5: Define the J-Lens experiment policy and types

**Files:**

- Create: `experiments/jlens_readout_sanity/types.py`
- Create: `experiments/jlens_readout_sanity/constants.py`
- Create: `tests/experiments/jlens_readout_sanity/test_constants.py`

- [ ] **Step 1: Write failing policy tests**

Create `tests/experiments/jlens_readout_sanity/test_constants.py` with the
existing exact coordinate, case, alpha, threshold, seed, and tolerance
assertions. Import from the new paths:

```python
from experiments.jlens_readout_sanity.constants import (
    CONTROL_ALPHA,
    CONTROL_CASE_KEYS,
    CONTROL_CHECK_MAP,
    CONTROL_SEEDS,
    DEFAULT_INTERVENTION_STRENGTHS,
    DEFAULT_MAX_FORMATTING_TOKENS,
    DEFAULT_MINIMUM_IMPROVEMENTS,
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    PERCENTILE_INTERPRETATION,
    PERCENTILE_QUANTILE,
    RANDOM_TARGET_NAMESPACE,
    RANDOM_VECTOR_NAMESPACE,
    READOUT_CASES,
    SWAP_CASES,
    TOP_K,
    WORKSPACE_LAYER_LOWER_FRACTION,
    WORKSPACE_LAYER_UPPER_FRACTION,
    WRONG_CONCEPT_REQUIRED_CASE_WINS,
)
```

Retain the exact current values and add:

```python
assert RANDOM_VECTOR_NAMESPACE == "jlens-control-v1"
assert RANDOM_TARGET_NAMESPACE == "jlens-random-target-v1"
assert CONTROL_CASE_KEYS == tuple(case.key for case in SWAP_CASES)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/experiments/jlens_readout_sanity/test_constants.py -q
```

Expected: collection fails because `types.py` and `constants.py` are absent.

- [ ] **Step 3: Implement experiment types**

Move these dataclasses into `types.py`:

```python
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


@dataclass(frozen=True, slots=True)
class ResolvedSwapCase:
    case: SwapCase
    read_case: ReadoutCase
    source: TokenVariant
    target: TokenVariant


@dataclass(slots=True)
class InterventionContext:
    resolved: ResolvedSwapCase
    input_ids: torch.Tensor
    scoring_input: torch.Tensor
    formatting_prefix: list[dict[str, Any]]
    clean_logits: torch.Tensor
    target_ids: tuple[int, ...]
    real_vectors_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]]
    workspace_loading: float | None
```

Import `TokenVariant` from `jlens_reasoning.experiments_utils.tokens` rather
than redefining it.

- [ ] **Step 4: Consolidate constants and cases**

Move all values from `readout_constants.py`, `sanity_constants.py`, and the
fixed `READOUT_CASES`/`SWAP_CASES` tuples into `constants.py`. Preserve their
values exactly. Add:

```python
RANDOM_VECTOR_NAMESPACE = "jlens-control-v1"
RANDOM_TARGET_NAMESPACE = "jlens-random-target-v1"
```

Derive rather than duplicate:

```python
CONTROL_CASE_KEYS = tuple(case.key for case in SWAP_CASES)
```

Keep `CONTROL_CHECK_MAP` as the one authoritative ordered mapping. The module
may construct immutable dataclass constants, but it must not import torch,
write files, register hooks, or execute controls.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/experiments/jlens_readout_sanity/test_constants.py -q
```

Expected: all exact policy tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add experiments/jlens_readout_sanity/types.py experiments/jlens_readout_sanity/constants.py tests/experiments/jlens_readout_sanity/test_constants.py
git commit -m "refactor: define J-Lens experiment policy package"
```

## Task 6: Move J-Lens execution and control orchestration

**Files:**

- Create: `experiments/jlens_readout_sanity/runner.py`
- Create: `experiments/jlens_readout_sanity/controls.py`
- Create: `experiments/jlens_readout_sanity/utils.py`
- Create: `tests/experiments/jlens_readout_sanity/test_runner.py`
- Create: `tests/experiments/jlens_readout_sanity/test_controls.py`
- Create: `tests/experiments/jlens_readout_sanity/test_package.py`

- [ ] **Step 1: Write failing runner tests at the new imports**

Move the experiment-policy and per-case tests from
`tests/test_readout_sanity.py` into
`tests/experiments/jlens_readout_sanity/test_runner.py`. Import notebook-facing
symbols from `utils.py` and internal test targets from `runner.py` or
`controls.py` according to their defining module:

```python
from experiments.jlens_readout_sanity.runner import (
    analyze_case,
    analyze_swap_case,
    resolve_swap_cases,
    summarize_swap_logits,
)
from experiments.jlens_readout_sanity.controls import analyze_identity_case
from experiments.jlens_readout_sanity.utils import run_readout_sanity
```

Retain exact tests for:

- swap resolution and duplicate/missing keys;
- readout summaries and capability gates;
- identity invariance and tolerance failure;
- real alpha 1/2 swap schema and best-strength behavior;
- strict surface validation before lens forwards;
- configured alpha requirement;
- full five-case behavior with current stub classes.

Update monkeypatch targets to the actual defining modules, not the facade.

- [ ] **Step 2: Write failing experiment-control tests**

Move experiment-specific tests from `tests/test_sanity_controls.py` and the
complete five-case control integration test into
`tests/experiments/jlens_readout_sanity/test_controls.py`. Import:

```python
from experiments.jlens_readout_sanity.controls import (
    aggregate_all_checks,
    controls_passed,
    run_negative_controls,
    summarize_wrong_concept,
)
```

Retain assertions for exact five-case ordering, four strict wrong-concept wins,
all four global check keys, actionable failures, 16 repeated results, original
target scoring, transient logits, and 180 total stub interventions.

- [ ] **Step 3: Write failing package-boundary tests**

Create `tests/experiments/jlens_readout_sanity/test_package.py`:

```python
from experiments.jlens_readout_sanity import controls, runner, utils


def test_utils_is_the_small_notebook_facade() -> None:
    assert utils.run_readout_sanity is runner.run_readout_sanity
    assert utils.write_results.__module__ == (
        "jlens_reasoning.experiments_utils.artifacts"
    )
    assert utils.validate_model_lens.__module__ == (
        "jlens_reasoning.experiments_utils.validation"
    )


def test_control_orchestration_remains_experiment_local() -> None:
    assert controls.run_negative_controls.__module__ == (
        "experiments.jlens_readout_sanity.controls"
    )
```

- [ ] **Step 4: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/experiments/jlens_readout_sanity -q
```

Expected: collection fails because runner, controls, and facade modules do not
exist.

- [ ] **Step 5: Implement `runner.py` with shared mechanics**

Move these J-Lens-specific implementations into `runner.py`:

```text
readout_cases.resolve_swap_cases
readout_sanity._summarize_lens
readout_sanity._readout_payload
readout_sanity.analyze_case
intervention_utils.summarize_swap_logits
intervention_utils._prepare_intervention_context
intervention_utils.analyze_swap_case
readout_utils.aggregate_capability_checks
readout_sanity.run_readout_sanity
```

Replace old imports with the focused shared modules. Every call to generic
helpers must pass local policy explicitly:

```python
layers = workspace_layers(
    model.n_layers,
    lens.source_layers,
    lower_fraction=WORKSPACE_LAYER_LOWER_FRACTION,
    upper_fraction=WORKSPACE_LAYER_UPPER_FRACTION,
)

scoring_input, formatting_prefix = prepare_scoring_input(
    input_ids,
    forward_next_token=forward_next_token,
    tokenizer=tokenizer,
    max_formatting_tokens=DEFAULT_MAX_FORMATTING_TOKENS,
)
```

Use public `token_vectors_by_layer` from shared interventions. Preserve every
existing result field, alpha key normalization, validation ordering, transient
logit behavior, and failure message. Import `run_negative_controls` and
`aggregate_all_checks` from local `controls.py` only at module level; local
`controls.py` must not import `runner.py`.

- [ ] **Step 6: Implement experiment-local `controls.py`**

Move `analyze_identity_case`, `_rank_gain_payload`,
`_intervention_payload_at_alpha`, `run_negative_controls`,
`summarize_wrong_concept`, `controls_passed`, `_control_failure`, and
`aggregate_all_checks` into the local module. Import `InterventionContext` from
local `types.py`, and import token ranking plus intervention execution from the
shared utility package. Do not import `runner.py`. Replace fixed shared-policy
imports with `constants.py`. Call generic functions with the explicit local
values:

```python
random_vectors, norm_report = matched_random_vectors(
    context.real_vectors_by_layer,
    base_seed=seed,
    namespace=RANDOM_VECTOR_NAMESPACE,
    norm_atol=NORM_ATOL,
    norm_rtol=NORM_RTOL,
    low_precision_norm_atol=LOW_PRECISION_NORM_ATOL,
    low_precision_norm_rtol=LOW_PRECISION_NORM_RTOL,
    max_attempts=MAX_RANDOM_VECTOR_ATTEMPTS,
)

random_vector_gate = strict_percentile_gate(
    real_mean,
    random_vector_means,
    quantile=PERCENTILE_QUANTILE,
    interpretation=PERCENTILE_INTERPRETATION,
)

selected_targets = select_random_targets(
    tokenizer,
    excluded_ids=exclusions["all"],
    seeds=CONTROL_SEEDS,
    output_vocab_size=output_vocab_size,
    namespace=RANDOM_TARGET_NAMESPACE,
)
```

Wrap generic key validation locally:

```python
def require_exact_cases(results: Sequence[Mapping[str, Any]]) -> None:
    require_exact_case_keys(results, expected_keys=CONTROL_CASE_KEYS)
```

Preserve the current result schema and control/global pass separation exactly.

- [ ] **Step 7: Implement the notebook facade**

Create `utils.py` as imports plus `__all__`, with no copied implementations:

```python
from experiments.jlens_readout_sanity.runner import run_readout_sanity
from jlens_reasoning.experiments_utils.artifacts import write_results
from jlens_reasoning.experiments_utils.tokens import concept_token_variants
from jlens_reasoning.experiments_utils.validation import validate_model_lens

__all__ = [
    "concept_token_variants",
    "run_readout_sanity",
    "validate_model_lens",
    "write_results",
]
```

- [ ] **Step 8: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/experiments_utils tests/experiments/jlens_readout_sanity -q
```

Expected: all shared and experiment-local tests pass while the old package
still exists temporarily.

- [ ] **Step 9: Commit Task 6**

```bash
git add experiments/jlens_readout_sanity/runner.py experiments/jlens_readout_sanity/controls.py experiments/jlens_readout_sanity/utils.py tests/experiments/jlens_readout_sanity/test_runner.py tests/experiments/jlens_readout_sanity/test_controls.py tests/experiments/jlens_readout_sanity/test_package.py
git commit -m "refactor: move J-Lens sanity into its experiment package"
```

## Task 7: Remove old modules and complete the clean import migration

**Files:**

- Delete: `src/jlens_reasoning/experiments/`
- Delete: `tests/test_experiment_constants.py`
- Delete: `tests/test_readout_module_structure.py`
- Delete: `tests/test_readout_sanity.py`
- Delete: `tests/test_sanity_controls.py`
- Modify: `tests/test_imports.py`
- Modify: all remaining Python imports found by `rg`

- [ ] **Step 1: Add failing clean-migration assertions**

Append to `tests/test_imports.py`:

```python
from pathlib import Path
import importlib.util


def test_old_experiments_package_is_removed() -> None:
    assert not Path("src/jlens_reasoning/experiments").exists()
    assert importlib.util.find_spec("jlens_reasoning.experiments") is None
```

- [ ] **Step 2: Run the assertion and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_imports.py::test_old_experiments_package_is_removed -q
```

Expected: failure because the old directory and package still exist.

- [ ] **Step 3: Verify replacement coverage before deleting**

Run:

```bash
rg -n "^def |^class " src/jlens_reasoning/experiments
rg -n "jlens_reasoning\.experiments" experiments src tests
```

For every old public or tested symbol, confirm a new defining module or an
intentional experiment-local replacement exists. The second command may show
only old tests and old modules at this point; it must show no new module using
the old path.

- [ ] **Step 4: Delete the old package and superseded tests**

Remove the nine files under `src/jlens_reasoning/experiments`, then remove the
four superseded top-level test files only after their assertions exist in the
new mirrored test directories.

Do not delete `src/jlens_reasoning/environments`, `evaluation.py`, runtime,
tracking, configuration, or unrelated tests.

- [ ] **Step 5: Update remaining imports and verify none remain**

Update `tests/test_imports.py` module expectations to include:

```python
EXPECTED_MODULES = (
    "experiments.jlens_readout_sanity.constants",
    "experiments.jlens_readout_sanity.controls",
    "experiments.jlens_readout_sanity.runner",
    "experiments.jlens_readout_sanity.utils",
    "jlens_reasoning.experiments_utils.artifacts",
    "jlens_reasoning.experiments_utils.controls",
    "jlens_reasoning.experiments_utils.interventions",
    "jlens_reasoning.experiments_utils.tokens",
    "jlens_reasoning.experiments_utils.validation",
)
```

Run:

```bash
rg -n "jlens_reasoning\.experiments" experiments src tests
```

Expected: exit 1 with no matches. Do not add an old-path compatibility module.

- [ ] **Step 6: Run the full Python test suite and verify GREEN**

Run:

```bash
.venv/bin/pytest -q
```

Expected: all tests pass with no imports from the deleted package.

- [ ] **Step 7: Commit Task 7**

```bash
git add src/jlens_reasoning tests experiments
git commit -m "refactor: remove legacy experiments package"
```

## Task 8: Move and update the explicit Colab notebook

**Files:**

- Create by move: `experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`
- Delete by move: `notebooks/01_jlens_readout_sanity.ipynb`
- Modify: `tests/test_notebooks.py`
- Modify: `README.md` only at the notebook path reference

- [ ] **Step 1: Write failing notebook discovery tests**

Replace the manually maintained notebook list in `tests/test_notebooks.py` with:

```python
SHARED_NOTEBOOKS = [
    Path("notebooks/_template.ipynb"),
    Path("notebooks/00_environment_check.ipynb"),
]
EXPERIMENT_NOTEBOOKS = sorted(Path("experiments").glob("*/*.ipynb"))
NOTEBOOKS = [*SHARED_NOTEBOOKS, *EXPERIMENT_NOTEBOOKS]
```

Add:

```python
def test_experiment_notebooks_are_discovered_without_a_registry() -> None:
    assert EXPERIMENT_NOTEBOOKS == [
        Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb")
    ]
    assert not Path("notebooks/01_jlens_readout_sanity.ipynb").exists()
```

Update the J-Lens notebook contract test to load the new path and assert:

```python
assert "from experiments.jlens_readout_sanity.constants import" in source
assert "from experiments.jlens_readout_sanity.utils import" in source
assert "run_readout_sanity" in source
assert "write_results" in source
assert "JacobianLens.from_pretrained" in source
assert "forward_next_token" in source
assert "compute_slice" in source
assert "raise RuntimeError" in source
```

- [ ] **Step 2: Run notebook tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_notebooks.py -q
```

Expected: the experiment notebook discovery assertion fails because the
notebook remains at its old path.

- [ ] **Step 3: Move the notebook without changing the canonical loader cell**

Move `notebooks/01_jlens_readout_sanity.ipynb` to
`experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb` with nbformat or
a JSON-aware move. Preserve null execution counts, empty outputs, cell IDs, and
the exact first cell source from `_template.ipynb`.

- [ ] **Step 4: Update notebook imports while preserving debuggable cells**

In the import/model-loading cell, import fixed values from `constants.py` and
notebook helpers from `utils.py`:

```python
from experiments.jlens_readout_sanity.constants import (
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    READOUT_CASES,
)
from experiments.jlens_readout_sanity.utils import (
    concept_token_variants,
    run_readout_sanity,
    validate_model_lens,
    write_results,
)
```

Keep separate visible cells for environment initialization, model/lens loading,
the next-token adapter and experiment run, result/provenance writing, concise
summaries, visualization, and the final gate. Do not replace them with a single
`run_experiment(context)` call.

- [ ] **Step 5: Update the README path without staging the user's existing hunk**

Change only:

```text
notebooks/01_jlens_readout_sanity.ipynb
```

to:

```text
experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb
```

The worktree already contains an unrelated unstaged README change in the LLM
answer-evaluation paragraph. Inspect `git diff -- README.md`, stage only the
notebook-path hunk with patch staging, and verify the unrelated paragraph does
not appear in `git diff --cached -- README.md`.

- [ ] **Step 6: Run notebook and formatting tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_notebooks.py -q
.venv/bin/ruff check experiments tests/test_notebooks.py
.venv/bin/ruff format --check experiments tests/test_notebooks.py
```

Expected: notebook discovery and all contract checks pass; Ruff reports no
issues.

- [ ] **Step 7: Commit Task 8 with only the intended README hunk**

Stage the notebook move and test normally. Stage only the README notebook-path
hunk, then inspect the cached diff:

```bash
git add experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb notebooks/01_jlens_readout_sanity.ipynb tests/test_notebooks.py
git add -p README.md
git diff --cached -- README.md
git commit -m "refactor: colocate J-Lens notebook with its experiment"
```

Expected after commit: `git status --short` still shows the pre-existing
unstaged README evaluation-policy edit.

## Task 9: Verify the installed wheel and complete the migration

**Files:**

- Verify all changed files.
- Modify only files required to fix migration-caused failures.

- [ ] **Step 1: Run focused architecture and experiment tests**

```bash
.venv/bin/pytest tests/test_package_discovery.py tests/test_imports.py tests/experiments_utils tests/experiments/jlens_readout_sanity tests/test_notebooks.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the full test suite**

```bash
.venv/bin/pytest -q
```

Expected: all tests pass. Any unrelated pre-existing failure must be shown to
reproduce without the migration before it is reported as unrelated.

- [ ] **Step 3: Run repository Ruff checks**

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Expected: both commands exit zero.

- [ ] **Step 4: Build the wheel from the repository**

Run these commands in one shell session from the repository worktree:

```bash
JLENS_REPO_ROOT="$PWD"
JLENS_WHEEL_CHECK_DIR="$(mktemp -d)"
uv build --wheel --clear --out-dir "$JLENS_WHEEL_CHECK_DIR" .
JLENS_WHEEL_PATH="$(find "$JLENS_WHEEL_CHECK_DIR" -maxdepth 1 -type f -name '*.whl' -print -quit)"
test -n "$JLENS_WHEEL_PATH"
```

Expected: exactly one wheel is produced.

- [ ] **Step 5: Verify imports from the wheel without the repository on `sys.path`**

Install the wheel into a fresh target directory without dependencies, reusing
the current verified environment only for third-party dependencies:

```bash
export JLENS_WHEEL_TARGET="$(mktemp -d)"
"$JLENS_REPO_ROOT/.venv/bin/pip" install --no-deps --target "$JLENS_WHEEL_TARGET" "$JLENS_WHEEL_PATH"
```

Change to `/private/tmp`, insert the wheel target at index zero, and assert both
imported files come from it rather than the editable checkout:

```bash
cd /private/tmp
"$JLENS_REPO_ROOT/.venv/bin/python" -c "import os,sys; target=os.environ['JLENS_WHEEL_TARGET']; sys.path.insert(0,target); import experiments.jlens_readout_sanity.utils as e; import jlens_reasoning.experiments_utils.controls as c; assert e.__file__.startswith(target); assert c.__file__.startswith(target)"
cd "$JLENS_REPO_ROOT"
```

Expected: the command exits zero. Inspect the wheel archive:

```bash
"$JLENS_REPO_ROOT/.venv/bin/python" -m zipfile -l "$JLENS_WHEEL_PATH"
```

Confirm it contains both:

```text
experiments/jlens_readout_sanity/
jlens_reasoning/experiments_utils/
```

It must not contain `jlens_reasoning/experiments/` or test packages.

- [ ] **Step 6: Check whitespace, imports, paths, and scope**

```bash
git diff --check
rg -n "jlens_reasoning\.experiments|notebooks/01_jlens_readout_sanity\.ipynb" experiments src tests README.md
git status --short
```

Expected: `git diff --check` exits zero; ripgrep exits 1 with no old import or
notebook path; status contains only intended migration changes and the
pre-existing unstaged README evaluation-policy edit.

- [ ] **Step 7: Review every acceptance criterion**

Confirm from the final tree and tests:

- no old `src/jlens_reasoning/experiments` package or compatibility shim;
- shared modules accept policy arguments and import no J-Lens experiment
  constants;
- J-Lens constants, cases, gates, failure formatting, and orchestration are
  experiment-local;
- current prompts, alphas, control outputs, result schema, artifact names, and
  scientific gates are unchanged;
- notebook cells expose setup, loading, execution, saving, reporting,
  visualization, and final failure independently;
- shared notebooks remain under `notebooks/`;
- experiment notebook discovery requires no registry or central list edit;
- wheel imports work outside the repository.

- [ ] **Step 8: Request code review and rerun verification after fixes**

Invoke `superpowers:requesting-code-review`. Address only verified issues, then
rerun Steps 1-6 before claiming completion.
