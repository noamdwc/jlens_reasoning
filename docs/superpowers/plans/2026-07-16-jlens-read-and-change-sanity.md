# J-Lens Read-and-Change Sanity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom readout-only gate with a paper-aligned sanity experiment that reads the spider intermediate and causally tests spider-to-ant and France-to-China J-Lens coordinate swaps on Qwen3.5-4B.

**Architecture:** Keep prompt/readout, intervention math, hook management, next-token ranking, and capability aggregation in `readout_sanity.py`; keep GPU model loading, the Hugging Face forward callback, provenance, persistence, and HTML rendering in the Colab notebook. Implement the paper's pseudoinverse coordinate swap as a pure tensor function, apply it through exception-safe block hooks, and grade clean/alpha-1/alpha-2 next-token distributions with an aggregate three-of-five improvement plus any-top-1 gate.

**Tech Stack:** Python 3.11, PyTorch, `jlens`, Hugging Face Transformers, pytest, nbformat, Ruff, Jupyter/Colab.

---

## File Structure

- Modify `src/jlens_reasoning/experiments/readout_sanity.py`: case definitions, token resolution, vector math, hooks, clean/intervened inference summaries, workspace loading, and aggregate checks.
- Modify `tests/test_readout_sanity.py`: CPU-only red-green coverage for definitions, math, hooks, metrics, and orchestration.
- Modify `notebooks/01_jlens_readout_sanity.ipynb`: remove long-form generation grading and invoke next-token coordinate swaps.
- Modify `tests/test_notebooks.py`: enforce the new notebook contract and reject the removed generation/evaluator path.
- Modify `README.md`: describe the read-and-change experiment without staging the user's pre-existing evaluator-policy edit.

Use `/tmp/jlens-reasoning-uv-cache` as `UV_CACHE_DIR` for every `uv` command
because the default host cache is outside the sandbox. Each command below sets
it explicitly.

### Task 1: Define the released swap cases and strict swap tokens

**Files:**

- Modify: `tests/test_readout_sanity.py:8-95`
- Modify: `src/jlens_reasoning/experiments/readout_sanity.py:21-99`

- [ ] **Step 1: Write failing definition and token-resolution tests**

Extend the import block with `SWAP_CASES`, `SwapCase`, and
`single_token_surface`. Replace the existing prompt-definition test with:

```python
def test_cases_cover_released_read_and_swap_examples() -> None:
    read_cases = {case.key: case for case in READOUT_CASES}
    swap_cases = {case.key: case for case in SWAP_CASES}

    assert read_cases["spider"].prompt == (
        "The number of legs on the animal that spins webs is"
    )
    assert read_cases["spider"].expected_answers == ("8", "eight")
    assert read_cases["spider"].target_concepts == ("spider",)

    assert [(case.key, case.target_answers[0]) for case in SWAP_CASES] == [
        ("spider", "6"),
        ("france_capital", "Beijing"),
        ("france_language", "Chinese"),
        ("france_continent", "Asia"),
        ("france_currency", "Yuan"),
    ]
    assert swap_cases["spider"].source_surface == " spider"
    assert swap_cases["spider"].target_surface == " ant"
    france_swaps = [case for case in SWAP_CASES if case.key.startswith("france_")]
    assert all(case.source_surface == " France" for case in france_swaps)
    assert all(case.target_surface == " China" for case in france_swaps)


def test_single_token_surface_is_strict() -> None:
    tokenizer = FakeTokenizer()

    assert single_token_surface(tokenizer, " France") == TokenVariant(
        token_id=17,
        surface=" France",
    )
    with pytest.raises(ValueError, match="exactly one token"):
        single_token_surface(tokenizer, " FRANCE")
```

Also import `TokenVariant` so the equality assertion uses the public value
object.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run pytest tests/test_readout_sanity.py::test_cases_cover_released_read_and_swap_examples tests/test_readout_sanity.py::test_single_token_surface_is_strict -v
```

Expected: collection fails because `SWAP_CASES`, `SwapCase`, and
`single_token_surface` do not exist.

- [ ] **Step 3: Add the minimal swap definitions and resolver**

Add after `ReadoutCase`:

```python
@dataclass(frozen=True, slots=True)
class SwapCase:
    key: str
    source_surface: str
    target_surface: str
    target_answers: tuple[str, ...]
```

Add after `READOUT_CASES`:

```python
SWAP_CASES = (
    SwapCase("spider", " spider", " ant", ("6", "six")),
    SwapCase("france_capital", " France", " China", ("Beijing",)),
    SwapCase("france_language", " France", " China", ("Chinese",)),
    SwapCase("france_continent", " France", " China", ("Asia",)),
    SwapCase("france_currency", " France", " China", ("Yuan",)),
)
```

Add after `TokenVariant`:

```python
def single_token_surface(tokenizer: Any, surface: str) -> TokenVariant:
    token_ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Configured swap surface {surface!r} must encode as exactly one token"
        )
    return TokenVariant(token_id=token_ids[0], surface=surface)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: both tests pass.

- [ ] **Step 5: Commit the case contract**

```bash
git add src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_sanity.py
git commit -m "test: define paper swap sanity cases"
```

### Task 2: Implement the paper's coordinate-swap tensor math

**Files:**

- Modify: `tests/test_readout_sanity.py`
- Modify: `src/jlens_reasoning/experiments/readout_sanity.py`

- [ ] **Step 1: Write failing vector and coordinate tests**

Import `coordinate_swap` and `jlens_vector`, then add:

```python
def test_jlens_vector_composes_jacobian_and_unembedding() -> None:
    lens = SimpleNamespace(
        jacobians={1: torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    )
    unembedding = torch.tensor([[0.0, 0.0], [5.0, 6.0]])

    assert torch.equal(
        jlens_vector(lens, unembedding, layer=1, token_id=1),
        torch.tensor([23.0, 34.0]),
    )


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        (0.0, [1.0, 0.0, 7.0]),
        (1.0, [0.0, 1.0, 7.0]),
        (2.0, [-1.0, 2.0, 7.0]),
    ],
)
def test_coordinate_swap_strength_and_orthogonal_component(
    alpha: float,
    expected: list[float],
) -> None:
    hidden = torch.tensor([1.0, 0.0, 7.0])
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])

    actual = coordinate_swap(hidden, source, target, alpha=alpha)

    assert torch.allclose(actual, torch.tensor(expected))


def test_coordinate_swap_preserves_shape_and_dtype() -> None:
    hidden = torch.tensor(
        [[[1.0, 0.0], [0.5, 0.25]]],
        dtype=torch.bfloat16,
    )

    actual = coordinate_swap(
        hidden,
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        alpha=1.0,
    )

    assert actual.shape == hidden.shape
    assert actual.dtype == hidden.dtype
```

- [ ] **Step 2: Run the tensor tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run pytest tests/test_readout_sanity.py -k "jlens_vector or coordinate_swap" -v
```

Expected: collection fails because both functions are missing.

- [ ] **Step 3: Implement the minimal pure tensor functions**

Add:

```python
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
```

- [ ] **Step 4: Run the tensor tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit the intervention math**

```bash
git add src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_sanity.py
git commit -m "feat: implement J-Lens coordinate swap math"
```

### Task 3: Apply swaps through exception-safe transformer hooks

**Files:**

- Modify: `tests/test_readout_sanity.py`
- Modify: `src/jlens_reasoning/experiments/readout_sanity.py`

- [ ] **Step 1: Write failing hook lifecycle tests**

Import `LensCoordinateSwapper` and `torch.nn as nn`, then add:

```python
class TensorBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TupleBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, str]:
        return hidden, "cache"


def test_swapper_patches_all_positions_and_preserves_tuple_members() -> None:
    blocks = nn.ModuleList([TensorBlock(), TupleBlock()])
    vectors = {
        0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
        1: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
    }
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    with LensCoordinateSwapper(blocks, vectors, alpha=1.0):
        first = blocks[0](hidden)
        second, cache = blocks[1](hidden)

    assert torch.equal(first, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    assert torch.equal(second, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    assert cache == "cache"
    assert all(not block._forward_hooks for block in blocks)


def test_swapper_removes_hooks_after_exception() -> None:
    blocks = nn.ModuleList([TensorBlock()])
    vectors = {
        0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
    }

    with pytest.raises(RuntimeError, match="stop"):
        with LensCoordinateSwapper(blocks, vectors, alpha=1.0):
            raise RuntimeError("stop")

    assert not blocks[0]._forward_hooks
```

- [ ] **Step 2: Run the hook tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run pytest tests/test_readout_sanity.py -k swapper -v
```

Expected: collection fails because `LensCoordinateSwapper` is missing.

- [ ] **Step 3: Implement the hook context manager**

Add `from torch import nn` and implement:

```python
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
```

- [ ] **Step 4: Run the hook tests and verify GREEN**

Run the command from Step 2. Expected: both hook tests pass.

- [ ] **Step 5: Commit hook management**

```bash
git add src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_sanity.py
git commit -m "feat: clamp coordinate swaps across model layers"
```

### Task 4: Grade clean and intervened next-token distributions

**Files:**

- Modify: `tests/test_readout_sanity.py`
- Modify: `src/jlens_reasoning/experiments/readout_sanity.py`

- [ ] **Step 1: Write failing output-summary and aggregate-gate tests**

Import `aggregate_capability_checks`, `prepare_scoring_input`, and
`summarize_swap_logits`, then add:

```python
def test_swap_summary_uses_best_strength_and_clean_rank() -> None:
    tokenizer = RunnerTokenizer()
    tokenizer.pieces.update(
        {
            "6": [5],
            " 6": [5],
            "six": [5],
            " six": [5],
            "Six": [5],
            " Six": [5],
            "SIX": [5],
            " SIX": [5],
        }
    )
    clean = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
    alpha_1 = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, 3.5])
    alpha_2 = torch.tensor([1.0, 0.0, -1.0, -2.0, -3.0, 5.0])

    result = summarize_swap_logits(
        clean,
        {1.0: alpha_1, 2.0: alpha_2},
        clean_answers=("8", "eight"),
        target_answers=("6", "six"),
        tokenizer=tokenizer,
        top_k=3,
    )

    assert result["clean"]["expected_rank"] == 5
    assert result["clean"]["expected_top1"] is False
    assert result["clean"]["target_rank"] == 6
    assert result["interventions"]["1.0"]["target_rank"] == 2
    assert result["interventions"]["2.0"]["target_rank"] == 1
    assert result["best_intervened_rank"] == 1
    assert result["improved"] is True
    assert result["target_top1"] is True


class FormattingTokenizer(RunnerTokenizer):
    def decode(
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return " " if token_ids[0] == 0 else f"token-{token_ids[0]}"


def test_scoring_input_appends_only_bounded_clean_formatting_tokens() -> None:
    calls: list[list[int]] = []

    def forward_next_token(input_ids: torch.Tensor) -> torch.Tensor:
        calls.append(input_ids[0].tolist())
        logits = torch.zeros(6)
        logits[0 if input_ids.shape[1] == 1 else 4] = 5.0
        return logits

    scoring_input, prefix = prepare_scoring_input(
        torch.tensor([[9]]),
        forward_next_token=forward_next_token,
        tokenizer=FormattingTokenizer(),
        max_formatting_tokens=2,
    )

    assert scoring_input.tolist() == [[9, 0]]
    assert prefix == [{"token_id": 0, "token": " "}]
    assert calls == [[9], [9, 0]]


def test_capability_gate_requires_three_improvements_and_one_top1() -> None:
    read_results = [
        {"key": "spider", "checks": {"baseline_top1": True, "read_capability": True}},
        {"key": "france_capital", "checks": {"baseline_top1": True}},
        {"key": "france_language", "checks": {"baseline_top1": True}},
        {"key": "france_continent", "checks": {"baseline_top1": True}},
        {"key": "france_currency", "checks": {"baseline_top1": True}},
    ]
    swap_results = [
        {"improved": True, "target_top1": True},
        {"improved": True, "target_top1": False},
        {"improved": True, "target_top1": False},
        {"improved": False, "target_top1": False},
        {"improved": False, "target_top1": False},
    ]

    checks, failures = aggregate_capability_checks(read_results, swap_results)

    assert checks == {
        "clean_baselines": True,
        "spider_read": True,
        "swap_rank_improvements": True,
        "swap_target_top1": True,
    }
    assert failures == []


def test_capability_gate_reports_aggregate_swap_failures() -> None:
    read_results = [
        {"key": "spider", "checks": {"baseline_top1": True, "read_capability": True}},
    ]
    swap_results = [
        {"improved": True, "target_top1": False},
        {"improved": True, "target_top1": False},
        {"improved": False, "target_top1": False},
    ]

    checks, failures = aggregate_capability_checks(read_results, swap_results)

    assert checks["swap_rank_improvements"] is False
    assert checks["swap_target_top1"] is False
    assert failures == [
        "coordinate swaps improved 2/3 target ranks; need at least 3",
        "no coordinate swap placed its target answer at top-1",
    ]
```

- [ ] **Step 2: Run the metric tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run pytest tests/test_readout_sanity.py -k "swap_summary or capability_gate" -v
```

Expected: collection fails because the metric helpers are missing.

- [ ] **Step 3: Implement deterministic swap and aggregate summaries**

Add `Callable` to the `collections.abc` imports, then add:

```python
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
        "top1_token": tokenizer.decode(
            [top1_id], clean_up_tokenization_spaces=False
        ),
        "target_rank": best_target_rank(logits, target_ids),
        "top_tokens": top_tokens(logits, tokenizer, k=top_k),
    }


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
    clean = _next_token_payload(clean_logits, target_ids, tokenizer, top_k=top_k)
    clean["expected_rank"] = best_target_rank(clean_logits, expected_ids)
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
```

Add the bounded formatting-prefix helper before `summarize_swap_logits`:

```python
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

- [ ] **Step 4: Run the metric tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Update the read-case checks test-first**

Change `test_analyze_case_grades_baseline_and_spider_readout` to expect:

```python
assert result["checks"] == {
    "paper_top1_hit": True,
    "read_capability": True,
}
```

Run that single test and verify it fails because `analyze_case` still returns
`target_top_k`. Then replace the check construction in `analyze_case` with:

```python
checks = {}
if case.key == "spider":
    jacobian_rank = summaries["jacobian_lens"]["best_rank"]
    logit_rank = summaries["logit_lens"]["best_rank"]
    checks.update(
        {
            "paper_top1_hit": jacobian_rank == 1,
            "read_capability": jacobian_rank <= 5 and jacobian_rank < logit_rank,
        }
    )
```

Run the single test again. Expected: pass.

- [ ] **Step 6: Commit next-token metrics and gates**

```bash
git add src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_sanity.py
git commit -m "feat: grade read and change capability metrics"
```

### Task 5: Orchestrate clean, alpha-1, and alpha-2 interventions

**Files:**

- Modify: `tests/test_readout_sanity.py`
- Modify: `src/jlens_reasoning/experiments/readout_sanity.py`

- [ ] **Step 1: Add a failing tiny-model orchestration test**

Import `analyze_swap_case`. Add a tokenizer with the configured surfaces and a
one-block model whose output changes under the hook:

```python
class SwapTokenizer(RunnerTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.pieces.update(
            {
                " ant": [3],
                "6": [5],
                " 6": [5],
                "six": [5],
                " six": [5],
                "Six": [5],
                " Six": [5],
                "SIX": [5],
                " SIX": [5],
            }
        )


class TinySwapModel:
    n_layers = 4
    d_model = 2

    def __init__(self) -> None:
        self.layers = nn.ModuleList(
            [TensorBlock(), TensorBlock(), TensorBlock(), TensorBlock()]
        )

    def encode(self, prompt: str, *, max_length: int = 512) -> torch.Tensor:
        del prompt, max_length
        return torch.tensor([[0, 1]])


def test_analyze_swap_case_runs_clean_and_both_strengths() -> None:
    model = TinySwapModel()
    lens = SimpleNamespace(
        jacobians={2: torch.eye(2)},
        source_layers=[2],
        d_model=2,
    )
    unembedding = torch.zeros(6, 2)
    unembedding[2] = torch.tensor([1.0, 0.0])
    unembedding[3] = torch.tensor([0.0, 1.0])

    def forward_next_token(input_ids: torch.Tensor) -> torch.Tensor:
        del input_ids
        hidden = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        for block in model.layers:
            hidden = block(hidden)
        logits = torch.zeros(6)
        logits[4] = hidden[0, -1, 0]
        logits[5] = hidden[0, -1, 1]
        return logits

    result = analyze_swap_case(
        SwapCase("spider", " spider", " ant", ("6", "six")),
        read_case=ReadoutCase(
            key="spider",
            prompt="prompt",
            expected_answers=("8", "eight"),
            target_concepts=("spider",),
        ),
        model=model,
        lens=lens,
        tokenizer=SwapTokenizer(),
        unembedding_weight=unembedding,
        forward_next_token=forward_next_token,
        layers=[2],
        alphas=(1.0, 2.0),
        top_k=3,
    )

    assert result["source"] == {"surface": " spider", "token_id": 2}
    assert result["target"] == {"surface": " ant", "token_id": 3}
    assert set(result["interventions"]) == {"1.0", "2.0"}
    assert result["improved"] is True
    assert result["target_top1"] is True
```

- [ ] **Step 2: Run the orchestration test and verify RED**

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run pytest tests/test_readout_sanity.py::test_analyze_swap_case_runs_clean_and_both_strengths -v
```

Expected: collection fails because `analyze_swap_case` is missing.

- [ ] **Step 3: Implement per-case intervention orchestration**

Implement:

```python
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
        "workspace_layers": list(layers),
        **summary,
    }
```

- [ ] **Step 4: Run the orchestration test and verify GREEN**

Run the command from Step 2. Expected: pass.

- [ ] **Step 5: Add workspace-loading coverage and implementation**

Import `workspace_loading` and add this focused pure-function test:

```python
def test_workspace_loading_averages_layers_and_positions() -> None:
    activations = {
        2: torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        3: torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    }
    vectors = {
        2: torch.tensor([1.0, 0.0]),
        3: torch.tensor([1.0, 0.0]),
    }

    assert workspace_loading(activations, vectors, positions=[0, 1]) == pytest.approx(
        0.75
    )
```

Run the test and verify collection fails because the helper is missing. Then
implement:

```python
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
```

Use `jlens.hooks.ActivationRecorder` around the clean forward pass in
`analyze_swap_case`. Add this literal-inclusive position helper:

```python
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
```

Wrap the clean forward on the original prompt as follows before selecting the
formatting prefix:

```python
from jlens.hooks import ActivationRecorder

with torch.inference_mode(), ActivationRecorder(
    model.layers,
    at=layers,
) as recorder:
    forward_next_token(input_ids)

loading = None
if read_case.literal_argument is not None:
    loading = workspace_loading(
        recorder.activations,
        {layer: vectors_by_layer[layer][0] for layer in layers},
        positions=positions_from_literal(
            tokenizer,
            input_ids,
            read_case.literal_argument,
        ),
    )
```

Attach `"workspace_loading": loading` to the returned mapping. For spider the
value is `None`. Run the new test until it passes.

- [ ] **Step 6: Integrate swap results into `run_readout_sanity` test-first**

Replace `test_run_readout_sanity_keeps_failed_case_details` with:

```python
class TinyCompleteLens:
    d_model = 2
    source_layers = [2]
    n_prompts = 1000

    def __init__(self) -> None:
        self.jacobians = {2: torch.eye(2)}

    def apply(self, model, prompt, *, use_jacobian=True, **kwargs):
        del prompt, kwargs
        input_ids = model.encode("prompt")
        model_logits = torch.zeros(input_ids.shape[1], 6)
        model_logits[-1, 4] = 5.0
        readout = torch.zeros(input_ids.shape[1], 6)
        readout[:, 2] = 4.0 if use_jacobian else -1.0
        return {2: readout}, model_logits, input_ids


def test_run_readout_sanity_combines_read_and_change_checks() -> None:
    model = TinySwapModel()
    lens = TinyCompleteLens()
    tokenizer = SwapTokenizer()
    unembedding = torch.zeros(6, 2)
    unembedding[2] = torch.tensor([1.0, 0.0])
    unembedding[3] = torch.tensor([0.0, 1.0])

    def forward_next_token(input_ids: torch.Tensor) -> torch.Tensor:
        del input_ids
        hidden = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        for block in model.layers:
            hidden = block(hidden)
        logits = torch.zeros(6)
        logits[4] = hidden[0, -1, 0]
        logits[5] = hidden[0, -1, 1]
        return logits

    result = run_readout_sanity(
        model=model,
        lens=lens,
        tokenizer=tokenizer,
        unembedding_weight=unembedding,
        forward_next_token=forward_next_token,
        cases=(
            ReadoutCase(
                key="spider",
                prompt="prompt",
                expected_answers=("8", "eight"),
                target_concepts=("spider",),
            ),
        ),
        swap_cases=(SwapCase("spider", " spider", " ant", ("6", "six")),),
        minimum_improvements=1,
        top_k=3,
    )

    assert result["checks"] == {
        "clean_baselines": True,
        "spider_read": True,
        "swap_rank_improvements": True,
        "swap_target_top1": True,
    }
    assert result["cases"][0]["checks"]["baseline_top1"] is True
    assert result["swaps"][0]["target_top1"] is True
    assert result["passed"] is True
```

Run the test and verify it fails against the old runner signature. Then change
`run_readout_sanity` to require `unembedding_weight` and
`forward_next_token`, accept `swap_cases=SWAP_CASES`, use `(1.0, 2.0)` as the
default strengths, and implement:

```python
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
        read_result["baseline"]["formatting_prefix"] = swap_result[
            "formatting_prefix"
        ]
        read_result["baseline"]["expected_rank"] = swap_result["clean"][
            "expected_rank"
        ]
        read_result["checks"]["baseline_top1"] = swap_result["clean"][
            "expected_top1"
        ]
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
```

Remove the now-unused `answer_variants` and `answer_ids` locals from
`analyze_case`; formatted baseline correctness is added by the runner from the
clean swap-scoring distribution.

- [ ] **Step 7: Run the complete experiment-module tests**

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run pytest tests/test_readout_sanity.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit the complete read-and-change runner**

```bash
git add src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_sanity.py
git commit -m "feat: run paper-aligned read and change sanity"
```

### Task 6: Replace notebook generation grading with next-token swaps

**Files:**

- Modify: `tests/test_notebooks.py:52-64`
- Modify: `notebooks/01_jlens_readout_sanity.ipynb`

- [ ] **Step 1: Strengthen the notebook contract test**

Extend `test_readout_sanity_notebook_has_pinned_gpu_workflow` with:

```python
assert "forward_next_token" in source
assert "get_output_embeddings().weight" in source
assert "intervention_strengths" in source
assert 'result["swaps"]' in source
assert "causal_lm.generate" not in source
assert "SimpleFactualEvaluator" not in source
assert "max_new_tokens" not in source
```

- [ ] **Step 2: Run the notebook test and verify RED**

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run pytest tests/test_notebooks.py::test_readout_sanity_notebook_has_pinned_gpu_workflow -v
```

Expected: failure because the notebook still uses generation/evaluator grading
and does not define intervention inference.

- [ ] **Step 3: Update the notebook code cells**

Use `apply_patch` to modify the notebook JSON cell sources while preserving the
canonical loader cell, empty outputs, and null execution counts. The import
cell must remove all imports from `jlens_reasoning.evaluation` and
`evaluation_utils`.

Replace the experiment cell with this logic:

```python
@torch.inference_mode()
def forward_next_token(input_ids):
    return causal_lm(input_ids=input_ids, use_cache=False).logits[0, -1]


result = run_readout_sanity(
    model=model,
    lens=lens,
    tokenizer=tokenizer,
    unembedding_weight=causal_lm.get_output_embeddings().weight,
    forward_next_token=forward_next_token,
)
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
        f"read_rank={summary['best_rank']}",
        f"read_layer={summary['layer']}",
        f"read_position={summary['position']}",
    )
for swap in result["swaps"]:
    print(
        swap["key"],
        f"clean_target_rank={swap['clean']['target_rank']}",
        f"alpha1_rank={swap['interventions']['1.0']['target_rank']}",
        f"alpha2_rank={swap['interventions']['2.0']['target_rank']}",
        f"improved={swap['improved']}",
        f"target_top1={swap['target_top1']}",
    )
print("intervention_strengths", result["intervention_strengths"])
print(f"Saved: {result_path}")
```

Keep the existing HTML cell and final save-before-raise cell. Update the HTML
description to `Paper-aligned read-and-change open-model sanity check.`

- [ ] **Step 4: Run notebook tests and verify GREEN**

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run pytest tests/test_notebooks.py -v
```

Expected: all notebook tests pass and the notebook has no saved outputs.

- [ ] **Step 5: Commit notebook integration**

```bash
git add notebooks/01_jlens_readout_sanity.ipynb tests/test_notebooks.py
git commit -m "feat: run coordinate swaps in sanity notebook"
```

### Task 7: Update documentation without absorbing the existing README edit

**Files:**

- Modify: `README.md:96-113`

- [ ] **Step 1: Update only the experiment description**

Change the heading to `## J-Lens read-and-change sanity experiment` and replace
the old readout-only paragraph with:

```markdown
The experiment checks whether the J-Lens surfaces the unspoken `spider`
intermediate and whether clamped coordinate swaps causally redirect next-token
answers. It runs the paper's `spider`→`ant` example and the same
`France`→`China` swap across capital, language, continent, and currency prompts
at both the standard (`alpha=1`) and double (`alpha=2`) strengths. The result
artifact reports exact per-swap ranks and applies an open-model capability gate;
it does not claim numerical replication of Claude 4.5.
```

Do not alter or revert the pre-existing evaluator-policy hunk below this
section.

- [ ] **Step 2: Check the README diff and whitespace**

```bash
git diff -- README.md
git diff --check
```

Expected: two README hunks are visible—the new experiment-description hunk and
the pre-existing evaluator-policy hunk—with no whitespace errors.

- [ ] **Step 3: Stage only the experiment-description hunk**

Create a patch from `git diff -- README.md`, retain only the heading/experiment
paragraph hunk, and apply it to the index with:

```bash
git apply --cached /tmp/jlens-read-and-change-readme.patch
```

Then verify:

```bash
git diff --cached -- README.md
git diff -- README.md
```

Expected: the cached diff contains only the read-and-change documentation; the
working-tree diff still contains the user's evaluator-policy edit.

- [ ] **Step 4: Commit only the intended documentation hunk**

```bash
git commit -m "docs: describe J-Lens coordinate swap sanity"
```

### Task 8: Run full verification and inspect the final branch state

**Files:**

- Verify all modified source, tests, notebook, and documentation files.

- [ ] **Step 1: Run the complete test suite**

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run pytest -v
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run Ruff formatting checks and lint**

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run ruff check .
```

Expected: formatting check and lint both exit 0.

- [ ] **Step 3: Verify notebook cleanliness and JSON validity**

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv run pytest tests/test_notebooks.py -v
```

Expected: notebook parsing, empty outputs, null execution counts, and the
read-and-change workflow contract all pass.

- [ ] **Step 4: Inspect the branch diff and preserved user change**

```bash
git status --short --branch
git log --oneline --decorate -10
git diff --check origin/codex/jlens-readout-sanity...HEAD
```

Expected: the branch contains the design, plan, implementation, tests,
notebook, and intended README documentation commits. `README.md` remains
modified only by the user's pre-existing evaluator-policy change.

- [ ] **Step 5: Record the GPU verification handoff**

The local suite cannot prove model-level swap success because CI and this macOS
environment do not execute the 4B model with the released lens. Report the
exact Colab action: set `PROJECT_REF` to `codex/jlens-readout-sanity`, run all
cells on CUDA, and return the new `result.json` if the capability gate fails.
