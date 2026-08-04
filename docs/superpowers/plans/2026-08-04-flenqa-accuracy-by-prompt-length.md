# FLenQA Accuracy by Prompt Length Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable Colab notebook that evaluates all 9,862 unique FLenQA prompts and plots paper-compatible accuracy by nominal prompt length.

**Architecture:** Add a paper-compatible binary evaluator to the shared evaluation layer, then build a dedicated FLenQA accuracy runner on the existing atomic Parquet shard primitives. Keep aggregation in a pure analysis module and keep Transformers generation plus plotting visible in a thin Colab notebook.

**Tech Stack:** Python 3.11, PyTorch, Transformers, PyArrow/Parquet, nbformat, matplotlib in Colab, pytest, ruff

---

## File map

- Modify `src/jlens_reasoning/evaluation_utils.py`: gold-blind final binary-verdict extraction.
- Modify `src/jlens_reasoning/evaluation.py`: typed paper-compatible binary evaluation.
- Modify `docs/llm-answer-evaluation.md`: separate paper-compatibility policy.
- Modify `tests/test_evaluation.py`: evaluator regressions.
- Create `src/jlens_reasoning/benchmarks/flenqa/accuracy_storage.py`: result schema and incomplete-shard cleanup.
- Create `tests/benchmarks/flenqa/test_accuracy_storage.py`: schema and cleanup tests.
- Create `src/jlens_reasoning/benchmarks/flenqa/accuracy.py`: generation contracts, sharding, run metadata, resume, and result loading.
- Create `tests/benchmarks/flenqa/test_accuracy.py`: CPU-only runner tests.
- Create `src/jlens_reasoning/benchmarks/flenqa/accuracy_analysis.py`: paper-weighted and unique-prompt summaries.
- Create `tests/benchmarks/flenqa/test_accuracy_analysis.py`: aggregation tests.
- Modify `src/jlens_reasoning/benchmarks/flenqa/__init__.py`: public accuracy exports.
- Create `notebooks/flenqa_accuracy.ipynb`: full Colab driver and plots.
- Modify `tests/test_notebooks.py`: notebook discovery and visible-workflow contract.
- Modify `README.md`: usage and artifact documentation.

### Task 1: Add paper-compatible binary evaluation

**Files:**
- Modify: `src/jlens_reasoning/evaluation_utils.py`
- Modify: `src/jlens_reasoning/evaluation.py`
- Modify: `docs/llm-answer-evaluation.md`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing extraction tests**

Add imports for `extract_last_binary_verdict`, `BinaryVerdictResult`, and
`evaluate_paper_binary`, then add these tests to `tests/test_evaluation.py`:

```python
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("True", True),
        ("FALSE", False),
        ("First true, but finally false.", False),
        ("untrue and falsehood", None),
        ("", None),
    ],
)
def test_extract_last_binary_verdict_is_gold_blind(
    text: str,
    expected: bool | None,
) -> None:
    assert extract_last_binary_verdict(text) is expected


def test_paper_binary_evaluation_scores_the_last_verdict() -> None:
    output = ModelOutput("Initially false. Final answer: TRUE.")

    result = evaluate_paper_binary(output, expected=True)

    assert result == BinaryVerdictResult(
        raw_output=output,
        expected=True,
        verdict=True,
        correct=True,
    )


def test_paper_binary_evaluation_scores_missing_and_truncated_answers() -> None:
    missing = evaluate_paper_binary("I cannot determine it.", expected=False)
    truncated = evaluate_paper_binary(
        ModelOutput(
            "Reasoning... False then True",
            generation_status=GenerationStatus.TRUNCATED,
            finish_reason="length",
        ),
        expected=False,
    )

    assert missing.verdict is None
    assert missing.correct is False
    assert truncated.verdict is True
    assert truncated.correct is False


@pytest.mark.parametrize("expected", [0, 1, "True", None])
def test_paper_binary_evaluation_requires_a_boolean_label(expected: object) -> None:
    with pytest.raises(TypeError, match="boolean"):
        evaluate_paper_binary("True", expected=expected)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest tests/test_evaluation.py -q
```

Expected: collection fails because the new extractor, result type, and entry
point do not exist.

- [ ] **Step 3: Implement the extractor and typed evaluator**

In `evaluation_utils.py`, add a compiled standalone-word regex and this public
function:

```python
_BINARY_VERDICT = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def extract_last_binary_verdict(text: str) -> bool | None:
    """Return the last standalone True/False verdict in response order."""
    if not isinstance(text, str):
        raise TypeError("binary verdict extraction requires text")
    matches = tuple(_BINARY_VERDICT.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).casefold() == "true"
```

In `evaluation.py`, import the extractor and add:

```python
@dataclass(frozen=True, slots=True)
class BinaryVerdictResult:
    raw_output: ModelOutput
    expected: bool
    verdict: bool | None
    correct: bool

    def __post_init__(self) -> None:
        if type(self.expected) is not bool:
            raise TypeError("expected binary verdict must be boolean")
        if self.verdict is not None and type(self.verdict) is not bool:
            raise TypeError("extracted binary verdict must be boolean or None")
        if self.correct is not (self.verdict is self.expected):
            raise ValueError("binary correctness must match verdict and label")


def evaluate_paper_binary(
    output: str | ModelOutput,
    *,
    expected: bool,
) -> BinaryVerdictResult:
    """Apply the FLenQA paper's final-occurrence behavioral scoring rule."""
    if type(expected) is not bool:
        raise TypeError("expected binary verdict must be boolean")
    model_output = ModelOutput(output) if isinstance(output, str) else output
    verdict = extract_last_binary_verdict(model_output.text)
    return BinaryVerdictResult(
        raw_output=model_output,
        expected=expected,
        verdict=verdict,
        correct=verdict is expected,
    )
```

Append a section to `docs/llm-answer-evaluation.md` titled
`FLenQA Paper-Compatible Generated Verdict` that states: final standalone
case-insensitive verdict wins; missing verdict is incorrect; truncated output is
scored from available text; this is behavioral replication and does not replace
constrained-logit scoring.

- [ ] **Step 4: Run evaluator tests and lint**

Run:

```bash
uv run pytest tests/test_evaluation.py -q
uv run ruff check src/jlens_reasoning/evaluation.py src/jlens_reasoning/evaluation_utils.py tests/test_evaluation.py
```

Expected: all evaluator tests pass and ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/jlens_reasoning/evaluation.py src/jlens_reasoning/evaluation_utils.py tests/test_evaluation.py docs/llm-answer-evaluation.md
git commit -m "feat: add paper-compatible FLenQA evaluation"
```

### Task 2: Define accuracy result storage

**Files:**
- Create: `src/jlens_reasoning/benchmarks/flenqa/accuracy_storage.py`
- Create: `tests/benchmarks/flenqa/test_accuracy_storage.py`

- [ ] **Step 1: Write failing storage tests**

Create `tests/benchmarks/flenqa/test_accuracy_storage.py`:

```python
from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from jlens_reasoning.benchmarks.flenqa.accuracy_storage import (
    REQUIRED_TABLES,
    TABLE_SCHEMAS,
    record_batch,
    reset_incomplete_shard,
)


def test_accuracy_result_schema_is_exact_and_typed() -> None:
    schema = TABLE_SCHEMAS["results"]

    assert REQUIRED_TABLES == ("results",)
    assert schema.field("prompt_id").type == pa.string()
    assert schema.field("ctx_size").type == pa.int32()
    assert schema.field("n_input_tokens").type == pa.int32()
    assert schema.field("verdict").nullable
    assert schema.field("provenance").type.value_type.field("dispersion").type == pa.string()


def test_accuracy_record_batch_round_trips_nullable_verdict() -> None:
    batch = record_batch(
        {
            "prompt_id": ["p"],
            "canonical_index": [0],
            "problem_id": [3],
            "task": ["PIR"],
            "label": [True],
            "text": ["prompt"],
            "ctx_size": [250],
            "input_ids": [[1, 2]],
            "n_input_tokens": [2],
            "provenance": [[{"source_row_id": 7, "ctx_size": 250, "padding_type": "books", "dispersion": "random"}]],
            "generated_token_ids": [[9]],
            "generated_token_pieces": [["maybe"]],
            "generated_text": ["maybe"],
            "generation_status": ["complete"],
            "finish_reason": ["eos"],
            "verdict": [None],
            "correct": [False],
        }
    )

    assert batch.schema == TABLE_SCHEMAS["results"]
    assert batch.to_pydict()["verdict"] == [None]


def test_reset_removes_only_one_accuracy_shard(tmp_path: Path) -> None:
    target = tmp_path / "results" / "shard-00002.parquet"
    other = tmp_path / "results" / "shard-00003.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"target")
    other.write_bytes(b"other")

    reset_incomplete_shard(tmp_path, shard_id=2)

    assert not target.exists()
    assert other.read_bytes() == b"other"
```

- [ ] **Step 2: Run the tests and verify RED**

Run `uv run pytest tests/benchmarks/flenqa/test_accuracy_storage.py -q`.

Expected: import fails because `accuracy_storage.py` does not exist.

- [ ] **Step 3: Implement the exact schema and helpers**

Create `accuracy_storage.py` with `from __future__ import annotations`, import
`PROVENANCE_TYPE` from the existing FLenQA storage module, and define one
non-nullable `results` schema in the exact column order used by the test. Make
only `finish_reason` and `verdict` nullable. Re-export `ShardManifest`,
`ShardWriter`, `is_shard_complete`, `read_shard_manifest`, and
`validate_shard_manifest` from `experiments_utils.storage`.

Implement:

```python
REQUIRED_TABLES = ("results",)
TABLE_SCHEMAS = {
    "results": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("canonical_index", pa.int32(), nullable=False),
            pa.field("problem_id", pa.int32(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("label", pa.bool_(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("ctx_size", pa.int32(), nullable=False),
            pa.field("input_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("n_input_tokens", pa.int32(), nullable=False),
            pa.field("provenance", PROVENANCE_TYPE, nullable=False),
            pa.field("generated_token_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("generated_token_pieces", pa.list_(pa.string()), nullable=False),
            pa.field("generated_text", pa.string(), nullable=False),
            pa.field("generation_status", pa.string(), nullable=False),
            pa.field("finish_reason", pa.string()),
            pa.field("verdict", pa.bool_()),
            pa.field("correct", pa.bool_(), nullable=False),
        ]
    )
}


def record_batch(columns: Mapping[str, Sequence[Any]]) -> pa.RecordBatch:
    schema = TABLE_SCHEMAS["results"]
    if set(columns) != set(schema.names):
        raise ValueError(f"results columns must be exactly {schema.names}")
    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    if len({len(array) for array in arrays}) > 1:
        raise ValueError("results columns must have equal lengths")
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def reset_incomplete_shard(root: Path, *, shard_id: int) -> None:
    stem = f"shard-{shard_id:05d}"
    manifest = Path(root) / "manifests" / f"{stem}.json"
    for path in (manifest, manifest.with_suffix(".json.tmp")):
        if path.exists():
            path.unlink()
    final = Path(root) / "results" / f"{stem}.parquet"
    for path in (final, final.with_suffix(".parquet.tmp")):
        if path.exists():
            path.unlink()
```

- [ ] **Step 4: Run storage tests**

Run:

```bash
uv run pytest tests/benchmarks/flenqa/test_accuracy_storage.py -q
uv run ruff check src/jlens_reasoning/benchmarks/flenqa/accuracy_storage.py tests/benchmarks/flenqa/test_accuracy_storage.py
```

Expected: all tests pass and ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/jlens_reasoning/benchmarks/flenqa/accuracy_storage.py tests/benchmarks/flenqa/test_accuracy_storage.py
git commit -m "feat: define FLenQA accuracy storage"
```

### Task 3: Implement resumable accuracy inference

**Files:**
- Create: `src/jlens_reasoning/benchmarks/flenqa/accuracy.py`
- Create: `tests/benchmarks/flenqa/test_accuracy.py`
- Modify: `src/jlens_reasoning/benchmarks/flenqa/__init__.py`

- [ ] **Step 1: Write failing runner tests with CPU fakes**

Create fixtures in `test_accuracy.py` using `FlenqaRow`, `dataclasses.replace`,
and these fakes:

```python
class FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, list[list[int]]]:
        assert kwargs["truncation"] is False
        return {"input_ids": [[*range(len(text))]]}


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, prompt: str, *, max_new_tokens: int) -> ModelOutput:
        self.calls.append((prompt, max_new_tokens))
        return ModelOutput(
            "Final answer: True",
            token_ids=(21, 22),
            token_pieces=(" True", ""),
            finish_reason="eos",
        )
```

Add tests that assert:

```python
def test_run_accuracy_generates_once_per_unique_prompt(tmp_path: Path) -> None:
    generator = RecordingGenerator()
    rows = (_row(), replace(_row(), source_row_id=1))
    manifest = run_accuracy(
        rows,
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generate=generator,
        config=_config(expected_source_rows=2, expected_prompts=1),
    )
    table = load_accuracy_results(tmp_path, manifest)

    assert len(generator.calls) == 1
    assert table.num_rows == 1
    assert table.column("correct").to_pylist() == [True]
    assert table.column("n_input_tokens").to_pylist() == [
        len(generator.calls[0][0])
    ]


def test_completed_run_resumes_without_generation(tmp_path: Path) -> None:
    config = _config()
    first = run_accuracy((_row(),), output_dir=tmp_path, tokenizer=FakeTokenizer(), generate=RecordingGenerator(), config=config)

    def fail(*args: object, **kwargs: object) -> ModelOutput:
        raise AssertionError("completed run must not generate")

    resumed = run_accuracy((_row(),), output_dir=tmp_path, tokenizer=FakeTokenizer(), generate=fail, config=config)
    assert resumed == first


def test_config_mismatch_is_rejected_before_shards_change(tmp_path: Path) -> None:
    run_accuracy((_row(),), output_dir=tmp_path, tokenizer=FakeTokenizer(), generate=RecordingGenerator(), config=_config())
    before = (tmp_path / "results" / "shard-00000.parquet").read_bytes()

    with pytest.raises(RuntimeError, match="configuration"):
        run_accuracy((_row(),), output_dir=tmp_path, tokenizer=FakeTokenizer(), generate=RecordingGenerator(), config=_config(max_new_tokens=8))

    assert (tmp_path / "results" / "shard-00000.parquet").read_bytes() == before


def test_failed_generation_aborts_incomplete_shard(tmp_path: Path) -> None:
    def fail(*args: object, **kwargs: object) -> ModelOutput:
        raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device failure"):
        run_accuracy((_row(),), output_dir=tmp_path, tokenizer=FakeTokenizer(), generate=fail, config=_config())

    assert not (tmp_path / "manifests" / "shard-00000.json").exists()
    assert not (tmp_path / "results" / "shard-00000.parquet").exists()


def test_over_limit_prompt_fails_before_generation(tmp_path: Path) -> None:
    generator = RecordingGenerator()
    with pytest.raises(ValueError, match="maximum sequence length"):
        run_accuracy((_row(),), output_dir=tmp_path, tokenizer=FakeTokenizer(), generate=generator, config=_config(max_seq_len=1))
    assert generator.calls == []
```

Also test deterministic shard membership, progress updates for completed and
new prompts, corrupted-shard rebuild, output metadata preservation, and an
expected-prompt-count mismatch before generation.

- [ ] **Step 2: Run runner tests and verify RED**

Run `uv run pytest tests/benchmarks/flenqa/test_accuracy.py -q`.

Expected: import fails because `accuracy.py` does not exist.

- [ ] **Step 3: Implement typed contracts and one-prompt execution**

Create `accuracy.py` with:

```python
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
```

Implement strict config validation, canonical shard planning, tokenizer output
normalization for a single sequence, and `run_prompt`. `run_prompt` derives the
single unanimous `ctx_size` from provenance, rejects over-limit inputs, calls
the generator once, calls `evaluate_paper_binary`, and returns a one-row Arrow
batch using the storage schema. Preserve the `ModelOutput` fields exactly.

- [ ] **Step 4: Implement atomic shards, run metadata, and resume**

Use `ShardWriter` and the accuracy storage helpers. Follow the existing FLenQA
runner's order:

1. validate config and row count;
2. deduplicate and validate expected prompt count;
3. compute SHA-256 over sorted JSON `asdict(config)`;
4. if `run-meta.json` exists, require the same config and hash before touching
   shards; otherwise atomically create it;
5. plan immutable shards by canonical index;
6. resume validated shards only when manifest prompt IDs match;
7. reset and rebuild incomplete/corrupt shards;
8. abort a writer on every `BaseException`;
9. atomically write `run-manifest.json` only after every shard validates.

Implement `load_accuracy_results(root, manifest)` using
`pyarrow.concat_tables` over shard files in manifest order. Validate schema,
row count, unique prompt IDs, and exact equality to `manifest.prompt_ids`.

- [ ] **Step 5: Export the public accuracy API**

Add these imports and `__all__` names in `benchmarks/flenqa/__init__.py`:

```python
from jlens_reasoning.benchmarks.flenqa.accuracy import (
    AccuracyRunConfig,
    AccuracyRunManifest,
    load_accuracy_results,
    run_accuracy,
)
```

- [ ] **Step 6: Run runner tests and benchmark regressions**

Run:

```bash
uv run pytest tests/benchmarks/flenqa/test_accuracy.py tests/benchmarks/flenqa -q
uv run ruff check src/jlens_reasoning/benchmarks/flenqa tests/benchmarks/flenqa
```

Expected: all FLenQA tests pass and ruff reports no errors.

- [ ] **Step 7: Commit**

```bash
git add src/jlens_reasoning/benchmarks/flenqa/accuracy.py src/jlens_reasoning/benchmarks/flenqa/__init__.py tests/benchmarks/flenqa/test_accuracy.py
git commit -m "feat: run resumable FLenQA accuracy evaluation"
```

### Task 4: Add paper and unique-prompt aggregation

**Files:**
- Create: `src/jlens_reasoning/benchmarks/flenqa/accuracy_analysis.py`
- Create: `tests/benchmarks/flenqa/test_accuracy_analysis.py`
- Modify: `src/jlens_reasoning/benchmarks/flenqa/__init__.py`

- [ ] **Step 1: Write failing aggregation tests**

Construct a small Arrow table with two prompts: one prompt has two random source
rows at 250 and the other has one random plus one first-placement row at 500.
Add:

```python
def test_unique_summary_weights_each_prompt_once() -> None:
    points = summarize_unique_prompts(_table())
    assert points == (
        AccuracyPoint(250, correct=1, total=1, no_verdict=0),
        AccuracyPoint(500, correct=0, total=1, no_verdict=1),
    )


def test_paper_summary_expands_only_random_source_rows() -> None:
    points = summarize_paper_random(_table())
    assert points == (
        AccuracyPoint(250, correct=2, total=2, no_verdict=0),
        AccuracyPoint(500, correct=0, total=1, no_verdict=1),
    )


def test_task_filter_applies_before_aggregation() -> None:
    assert summarize_unique_prompts(_table(), task="PIR") == (
        AccuracyPoint(250, correct=1, total=1, no_verdict=0),
    )


def test_full_unique_counts_are_pinned() -> None:
    assert FULL_UNIQUE_PROMPT_COUNTS == {
        250: 300,
        500: 2_368,
        1000: 2_394,
        2000: 2_400,
        3000: 2_400,
    }
```

Also test duplicate prompt rejection, mixed context-size provenance rejection,
missing prompt IDs, invalid task filters, verdict-frequency counts, and token
length min/median/max.

- [ ] **Step 2: Run tests and verify RED**

Run `uv run pytest tests/benchmarks/flenqa/test_accuracy_analysis.py -q`.

Expected: import fails because `accuracy_analysis.py` does not exist.

- [ ] **Step 3: Implement immutable summaries**

Create:

```python
FULL_UNIQUE_PROMPT_COUNTS = {
    250: 300,
    500: 2_368,
    1000: 2_394,
    2000: 2_400,
    3000: 2_400,
}


@dataclass(frozen=True, slots=True)
class AccuracyPoint:
    ctx_size: int
    correct: int
    total: int
    no_verdict: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total
```

Implement `summarize_unique_prompts(table, *, task=None)` by validating one row
per prompt and grouping rows by `ctx_size`. Implement
`summarize_paper_random(table, *, task=None)` by iterating each row's provenance
and appending one weighted observation for every provenance entry whose
`dispersion == "random"`. Sort results by context size.

Add `summarize_verdicts(table)` returning per-length counts of `True`, `False`,
and `None`, and `summarize_token_lengths(table)` returning per-length minimum,
median, and maximum. Use the standard library only; do not add pandas or scipy.

- [ ] **Step 4: Export and verify analysis**

Export the analysis constants, dataclasses, and functions from the FLenQA
package. Run:

```bash
uv run pytest tests/benchmarks/flenqa/test_accuracy_analysis.py -q
uv run ruff check src/jlens_reasoning/benchmarks/flenqa/accuracy_analysis.py tests/benchmarks/flenqa/test_accuracy_analysis.py
```

Expected: all analysis tests pass and ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/jlens_reasoning/benchmarks/flenqa/accuracy_analysis.py src/jlens_reasoning/benchmarks/flenqa/__init__.py tests/benchmarks/flenqa/test_accuracy_analysis.py
git commit -m "feat: summarize FLenQA accuracy by length"
```

### Task 5: Add the full-run Colab notebook

**Files:**
- Create: `notebooks/flenqa_accuracy.ipynb`
- Modify: `tests/test_notebooks.py`

- [ ] **Step 1: Write failing notebook contract tests**

Add `Path("notebooks/flenqa_accuracy.ipynb")` to `FLENQA_NOTEBOOKS`. Split the
existing benchmark-driver assertion so the lens drivers remain subject to
`run_benchmark`, while the accuracy notebook gets this contract:

```python
def test_flenqa_accuracy_notebook_has_visible_full_run_workflow() -> None:
    path = Path("notebooks/flenqa_accuracy.ipynb")
    cells = notebook_cells_by_id(path)
    source = "\n".join(load_notebook(path).cells[index].source for index in range(len(load_notebook(path).cells)))

    assert "initialize_colab(enable_wandb=False, require_cuda=True)" in source
    assert "normalize_rows(raw_rows, full=True)" in source
    assert "len(prompts) == 9_862" in source
    assert "AccuracyRunConfig(" in source
    assert "expected_source_rows=12_000" in source
    assert "expected_prompts=9_862" in source
    assert "do_sample=False" in cells["define-generation"]
    assert "max_new_tokens=max_new_tokens" in cells["define-generation"]
    assert "run_accuracy(" in cells["run-accuracy"]
    assert "summarize_paper_random" in cells["paper-curve"]
    assert "summarize_unique_prompts" in cells["unique-curve"]
    assert "run_benchmark(" not in source
    assert "JacobianLens" not in source
```

- [ ] **Step 2: Run notebook tests and verify RED**

Run `uv run pytest tests/test_notebooks.py -q`.

Expected: failure because the notebook is missing and the hardcoded notebook
list changed.

- [ ] **Step 3: Create the notebook with the canonical loader**

Create notebook-format 4.5 JSON with stable cell IDs and no outputs or execution
counts. Copy cell 0 from `notebooks/flenqa_full_run.ipynb` byte-for-byte.

The initialization cell must call:

```python
from jlens_reasoning.environments.colab import initialize_colab

context = initialize_colab(enable_wandb=False, require_cuda=True)
context
```

The import/settings cell imports `matplotlib.pyplot`, `pyarrow`, `torch`,
`transformers`, dataset loading, model constants, and the new public accuracy
API. It sets `OUTPUT_DIR = context.runs_dir / "flenqa-accuracy"`.

The dataset cell uses `normalize_rows(raw_rows, full=True)`, `deduplicate(rows)`,
and asserts `len(prompts) == 9_862` before displaying counts.

- [ ] **Step 4: Add deterministic model generation and the full run**

Load the local Qwen causal LM and tokenizer exactly like the existing FLenQA
drivers. Define:

```python
def generate_output(prompt: str, *, max_new_tokens: int) -> ModelOutput:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = encoded["input_ids"].to(context.device)
    with torch.inference_mode():
        generated = causal_lm.generate(
            input_ids=input_ids,
            attention_mask=encoded["attention_mask"].to(context.device),
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    generated_ids = generated[0, input_ids.shape[1] :].tolist()
    eos_ids = causal_lm.generation_config.eos_token_id
    eos_token_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or ())
    complete = bool(generated_ids and generated_ids[-1] in eos_token_ids)
    scored_ids = generated_ids[:-1] if complete else generated_ids
    return ModelOutput(
        text=tokenizer.decode(scored_ids, skip_special_tokens=True),
        token_ids=tuple(generated_ids),
        token_pieces=tuple(
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            for token_id in generated_ids
        ),
        generation_status=(
            GenerationStatus.COMPLETE if complete else GenerationStatus.TRUNCATED
        ),
        finish_reason="eos" if complete else "length",
    )
```

Invoke `run_accuracy` with all normalized rows and:

```python
config=AccuracyRunConfig(
    model_name=MODEL_NAME,
    tokenizer_name=MODEL_NAME,
    code_revision=PROJECT_COMMIT,
    expected_source_rows=12_000,
    expected_prompts=9_862,
)
```

- [ ] **Step 5: Add integrity summaries and plots**

Load results through `load_accuracy_results`. Render tables from the summary
dataclasses and plot accuracy against `[250, 500, 1000, 2000, 3000]` using
matplotlib. The first plot uses `summarize_paper_random`; the second uses
`summarize_unique_prompts`. Add per-task curves, verdict-frequency diagnostics,
and exact-token-length summary cells. Do not save notebook outputs.

- [ ] **Step 6: Run notebook and full notebook-contract tests**

Run:

```bash
uv run pytest tests/test_notebooks.py -q
uv run ruff format --check tests/test_notebooks.py
uv run ruff check tests/test_notebooks.py
```

Expected: all notebook tests pass, including canonical loader identity, no
outputs, no credentials, and the full accuracy workflow.

- [ ] **Step 7: Commit**

```bash
git add notebooks/flenqa_accuracy.ipynb tests/test_notebooks.py
git commit -m "feat: add FLenQA accuracy notebook"
```

### Task 6: Document and verify the complete feature

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the notebook and artifact contract**

Add a `FLenQA accuracy by prompt length` section after the benchmark-runner
section. State that the notebook runs 9,862 unique prompts by default, resumes
under `runs/flenqa-accuracy/`, uses greedy 64-token generation, reports the
paper's nominal length buckets and final-verdict extraction, and shows both
paper-weighted random-placement and unique-prompt curves.

- [ ] **Step 2: Run formatting and lint**

Run:

```bash
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
```

Expected: formatter makes no further changes on the check pass and lint reports
no errors.

- [ ] **Step 3: Run the full CPU suite and lock check**

Run:

```bash
uv run pytest
uv lock --check
```

Expected: all CPU tests pass and the committed lockfile is current. The
model-backed notebook is intentionally not executed in CI.

- [ ] **Step 4: Inspect the final diff and requirements**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Confirm every approved design requirement maps to implemented code or a passing
test, and confirm the unrelated `.claude/` directory remains untouched and
unstaged.

- [ ] **Step 5: Commit documentation and any mechanical formatting**

```bash
git add README.md
git commit -m "docs: document FLenQA accuracy evaluation"
```
