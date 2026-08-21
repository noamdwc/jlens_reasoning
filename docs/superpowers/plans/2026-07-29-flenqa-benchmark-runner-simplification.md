# FLenQA Benchmark Runner Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `experiments.flenqa_length_drift` with a small, tested FLenQA package that runs Jacobian Lens and Logit Lens at validated meaningful positions and saves three resumable Parquet tables.

**Architecture:** Move dataset and position-selection behavior into `jlens_reasoning.benchmarks.flenqa`, remove study-only scoring/preflight/summary behavior, and keep paired-lens execution separate from Arrow persistence. Reuse `jlens_reasoning.experiments_utils.storage.ShardWriter` for atomic table replacement, manifest-last completion, checksums, and incomplete-shard recovery.

**Tech Stack:** Python 3.11, PyTorch, PyArrow/Parquet, pytest, nbformat, Ruff.

---

## Final File Map

**Create**

- `src/jlens_reasoning/benchmarks/flenqa/__init__.py` — curated imports.
- `src/jlens_reasoning/benchmarks/flenqa/dataset.py` — source rows, typed provenance, prompts, normalization, and deduplication.
- `src/jlens_reasoning/benchmarks/flenqa/positions.py` — bridge extraction/gate, span resolution, padding-content eligibility, and labeled positions.
- `src/jlens_reasoning/benchmarks/flenqa/runner.py` — paired lens passes, tolerant comparison, top-k reduction, shards, and runs.
- `src/jlens_reasoning/benchmarks/flenqa/storage.py` — three Arrow schemas, record batches, run metadata, and manifest validation.
- `tests/benchmarks/flenqa/test_dataset.py`
- `tests/benchmarks/flenqa/test_positions.py`
- `tests/benchmarks/flenqa/test_runner.py`
- `tests/benchmarks/flenqa/test_storage.py`
- `notebooks/flenqa_smoke.ipynb`
- `notebooks/flenqa_full_run.ipynb`

**Modify**

- `tests/test_notebooks.py` — discover and validate the new notebook locations.
- `README.md` — document the benchmark runner and renamed notebooks.

**Delete after migration**

- `src/jlens_reasoning/benchmarks/flenqa.py`
- `src/jlens_reasoning/benchmarks/flenqa_prompts.py`
- `src/jlens_reasoning/benchmarks/flenqa_preparation.py`
- `src/jlens_reasoning/benchmarks/flenqa_conditions.py`
- all tracked Python files and tracked notebooks under `experiments/flenqa_length_drift/`
- `tests/benchmarks/test_flenqa.py`
- `tests/benchmarks/test_flenqa_prompts.py`
- `tests/benchmarks/test_flenqa_preparation.py`
- `tests/benchmarks/test_flenqa_conditions.py`
- `tests/experiments/flenqa_length_drift/`

Do not touch the untracked
`experiments/flenqa_length_drift/flenqa_smoke_output.ipynb`.

### Task 1: Create the Dataset Package and Typed Provenance

**Files:**

- Create: `src/jlens_reasoning/benchmarks/flenqa/__init__.py`
- Create: `src/jlens_reasoning/benchmarks/flenqa/dataset.py`
- Create: `tests/benchmarks/flenqa/__init__.py`
- Create: `tests/benchmarks/flenqa/test_dataset.py`
- Delete later: `src/jlens_reasoning/benchmarks/flenqa.py`
- Delete later: `src/jlens_reasoning/benchmarks/flenqa_prompts.py`

- [ ] **Step 1: Write the failing package/provenance tests**

Move the existing dataset and prompt tests into
`tests/benchmarks/flenqa/test_dataset.py`, change imports to
`jlens_reasoning.benchmarks.flenqa.dataset`, and add:

```python
from jlens_reasoning.benchmarks.flenqa.dataset import (
    SourceProvenance,
    deduplicate,
)


def test_deduplicate_preserves_complete_source_provenance() -> None:
    rows = (
        _row(
            source_row_id=7,
            ctx_size_declared=500,
            padding_type_declared="books",
            dispersion_declared="first",
        ),
        _row(
            source_row_id=3,
            ctx_size_declared=500,
            padding_type_declared="same",
            dispersion_declared="last",
        ),
    )

    prompt = deduplicate(rows)[0]

    assert prompt.provenance == (
        SourceProvenance(3, 500, "same", "last"),
        SourceProvenance(7, 500, "books", "first"),
    )
```

Keep the existing schema, normalization, count-invariant, prompt-text,
prompt-ID, and invariant-conflict tests in the same file.
Add a test that `normalize_rows(raw_rows, full=True)` applies the published
12,000-row marginals, while `full=False` permits an explicitly bounded smoke
subset.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
uv run pytest tests/benchmarks/flenqa/test_dataset.py -q
```

Expected: collection fails because `jlens_reasoning.benchmarks.flenqa.dataset`
does not exist.

- [ ] **Step 3: Move the existing implementation and introduce complete provenance**

Create the package directory, move the bodies of `flenqa.py` and
`flenqa_prompts.py` into `dataset.py`, and replace parallel provenance fields
with:

```python
@dataclass(frozen=True, slots=True)
class SourceProvenance:
    source_row_id: int
    ctx_size: int
    padding_type: str
    dispersion: str


@dataclass(frozen=True, slots=True)
class FlenqaPrompt:
    canonical_index: int
    prompt_id: str
    problem_id: int
    task: str
    text: str
    question: str
    key_texts: tuple[str, ...]
    rule: str | None
    label: bool
    mixin: str
    provenance: tuple[SourceProvenance, ...]
```

In `deduplicate`, retain the existing invariant checks and construct:

```python
provenance=tuple(
    sorted(
        (
            SourceProvenance(
                source_row_id=row.source_row_id,
                ctx_size=row.ctx_size_declared,
                padding_type=row.padding_type_declared,
                dispersion=row.dispersion_declared,
            )
            for row in source_rows
        ),
        key=lambda item: item.source_row_id,
    )
),
```

Change `normalize_rows` to accept a keyword-only `full: bool = False` argument
and replace its `verify_schema(rows)` call with
`verify_schema(rows, full=full)`. Its subsequent typed row construction remains
unchanged.

Export only the data types and public functions from `__init__.py`:

```python
from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaPrompt,
    FlenqaRow,
    SourceProvenance,
    deduplicate,
    normalize_rows,
    verify_count_invariants,
    verify_schema,
)

__all__ = [
    "FlenqaPrompt",
    "FlenqaRow",
    "SourceProvenance",
    "deduplicate",
    "normalize_rows",
    "verify_count_invariants",
    "verify_schema",
]
```

- [ ] **Step 4: Run dataset tests and verify GREEN**

Run:

```bash
uv run pytest tests/benchmarks/flenqa/test_dataset.py -q
```

Expected: all dataset tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jlens_reasoning/benchmarks/flenqa tests/benchmarks/flenqa
git commit -m "refactor: package FLenQA dataset utilities"
```

### Task 2: Consolidate Validated Meaningful Positions

**Files:**

- Create: `src/jlens_reasoning/benchmarks/flenqa/positions.py`
- Create: `tests/benchmarks/flenqa/test_positions.py`
- Source: `src/jlens_reasoning/benchmarks/flenqa_preparation.py`
- Source: `src/jlens_reasoning/benchmarks/flenqa_conditions.py`
- Source: `experiments/flenqa_length_drift/bridges.py`
- Source: `experiments/flenqa_length_drift/gate.py`
- Source: `experiments/flenqa_length_drift/anchors.py`

- [ ] **Step 1: Write failing tests for padding-content eligibility and duplicate labels**

Move the existing preparation, bridge, gate, and anchor tests into
`tests/benchmarks/flenqa/test_positions.py`. Retain their behavior tests, remove
tests for `derive_conditions` and `select_summary_positions`, and add:

```python
@pytest.mark.parametrize(
    ("token_id", "offset", "special_ids", "expected"),
    [
        (0, (0, 4), frozenset({0}), False),
        (10, (4, 6), frozenset(), False),
        (11, (6, 7), frozenset(), True),
        (12, (7, 11), frozenset(), True),
    ],
)
def test_padding_candidates_exclude_special_and_whitespace_only_tokens(
    token_id: int,
    offset: tuple[int, int],
    special_ids: frozenset[int],
    expected: bool,
) -> None:
    assert (
        _eligible_padding_position(
            token_id=token_id,
            offset=offset,
            text="word \n,tail",
            padding_spans=(CharSpan(0, 11),),
            special_ids=special_ids,
        )
        is expected
    )


def test_unique_positions_keeps_labels_but_deduplicates_execution_positions() -> None:
    positions = (
        LabeledPosition("question_end", 9),
        LabeledPosition("final_prompt", 9),
    )

    assert positions == (
        LabeledPosition("question_end", 9),
        LabeledPosition("final_prompt", 9),
    )
    assert unique_positions(positions) == (9,)


def test_padding_sample_depends_only_on_prompt_id_and_fixed_seed() -> None:
    candidates = tuple(range(20))

    first = sample_padding_positions(
        candidates, prompt_id="a" * 64, sample_seed=1729, count=4
    )
    second = sample_padding_positions(
        tuple(reversed(candidates)),
        prompt_id="a" * 64,
        sample_seed=1729,
        count=4,
    )

    assert first == second
    assert len(first) == 4
```

- [ ] **Step 2: Run the position tests and verify RED**

Run:

```bash
uv run pytest tests/benchmarks/flenqa/test_positions.py -q
```

Expected: collection fails because `flenqa.positions` does not exist.

- [ ] **Step 3: Consolidate only the retained position logic**

Move the tested tokenization/span-resolution implementation into
`positions.py`. Keep `extract_bridge`, `bridge_gate`,
`build_padding_content_positions`, and semantic-anchor selection. Delete
condition classification and the 48-position summary selector.

Use these public data contracts:

```python
PADDING_SAMPLE_COUNT = 4


@dataclass(frozen=True, slots=True, order=True)
class LabeledPosition:
    label: str
    position: int


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    prompt: FlenqaPrompt
    bridge: str | None
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    positions: tuple[LabeledPosition, ...]

    @property
    def unique_positions(self) -> tuple[int, ...]:
        return unique_positions(self.positions)


def unique_positions(
    positions: Sequence[LabeledPosition],
) -> tuple[int, ...]:
    return tuple(sorted({item.position for item in positions}))
```

The padding-content eligibility filter must be:

```python
def _eligible_padding_position(
    *,
    token_id: int,
    offset: tuple[int, int],
    text: str,
    padding_spans: Sequence[CharSpan],
    special_ids: frozenset[int],
) -> bool:
    start, end = offset
    if token_id in special_ids or end <= start:
        return False
    if not text[start:end].strip():
        return False
    return any(start < span.end and end > span.start for span in padding_spans)
```

This keeps punctuation because `",".strip()` is nonempty.

Derive the deterministic sample seed exactly once:

```python
def _prompt_sample_seed(prompt_id: str, sample_seed: int) -> int:
    return int(prompt_id[:16], 16) ^ sample_seed
```

Sort/deduplicate padding candidates before sampling, sample without
replacement, and sort the result before creating `sampled_padding` labels.
Preserve every semantic label even when positions collide.

`prepare_prompt` must accept `sample_seed` and `max_seq_len`, resolve all
required spans, construct labels, and raise on unresolved required spans or
out-of-range positions. `bridge_gate` keeps explicit expected counts for full
and smoke runs.

- [ ] **Step 4: Run position tests and verify GREEN**

Run:

```bash
uv run pytest tests/benchmarks/flenqa/test_positions.py -q
```

Expected: all position tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jlens_reasoning/benchmarks/flenqa/positions.py tests/benchmarks/flenqa/test_positions.py
git commit -m "refactor: consolidate FLenQA semantic positions"
```

### Task 3: Define the Three Typed Output Tables

**Files:**

- Create: `src/jlens_reasoning/benchmarks/flenqa/storage.py`
- Create: `tests/benchmarks/flenqa/test_storage.py`
- Source: `experiments/flenqa_length_drift/tables.py`
- Reuse: `src/jlens_reasoning/experiments_utils/storage.py`

- [ ] **Step 1: Write failing Arrow schema tests**

Create:

```python
import pyarrow as pa

from jlens_reasoning.benchmarks.flenqa.storage import (
    REQUIRED_TABLES,
    TABLE_SCHEMAS,
)


def test_only_three_shard_tables_are_required() -> None:
    assert REQUIRED_TABLES == ("prompts", "positions", "topk")
    assert set(TABLE_SCHEMAS) == set(REQUIRED_TABLES)


def test_prompt_provenance_is_a_typed_list_of_structs() -> None:
    provenance = TABLE_SCHEMAS["prompts"].field("provenance")

    assert pa.types.is_list(provenance.type)
    assert pa.types.is_struct(provenance.type.value_type)
    assert provenance.type.value_type.names == [
        "source_row_id",
        "ctx_size",
        "padding_type",
        "dispersion",
    ]
    assert provenance.type.value_type.field("source_row_id").type == pa.int32()
    assert provenance.type.value_type.field("ctx_size").type == pa.int32()
    assert provenance.type.value_type.field("padding_type").type == pa.string()
    assert provenance.type.value_type.field("dispersion").type == pa.string()
```

Also retain the exact-column and typed-record-batch tests from the old table
suite, updated for `positions`.

- [ ] **Step 2: Run the storage tests and verify RED**

Run:

```bash
uv run pytest tests/benchmarks/flenqa/test_storage.py -q
```

Expected: collection fails because `flenqa.storage` does not exist.

- [ ] **Step 3: Implement exact schemas and batch construction**

Define:

```python
PROVENANCE_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("source_row_id", pa.int32(), nullable=False),
            pa.field("ctx_size", pa.int32(), nullable=False),
            pa.field("padding_type", pa.string(), nullable=False),
            pa.field("dispersion", pa.string(), nullable=False),
        ]
    )
)

REQUIRED_TABLES = ("prompts", "positions", "topk")

TABLE_SCHEMAS = {
    "prompts": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("canonical_index", pa.int32(), nullable=False),
            pa.field("problem_id", pa.int32(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("label", pa.bool_(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("input_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("bridge", pa.string()),
            pa.field("max_abs_logit_diff", pa.float32(), nullable=False),
            pa.field("provenance", PROVENANCE_TYPE, nullable=False),
        ]
    ),
    "positions": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("position", pa.int32(), nullable=False),
            pa.field("label", pa.string(), nullable=False),
        ]
    ),
    "topk": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("lens_kind", pa.string(), nullable=False),
            pa.field("layer", pa.int16(), nullable=False),
            pa.field("position", pa.int32(), nullable=False),
            pa.field("rank", pa.int16(), nullable=False),
            pa.field("token_id", pa.int32(), nullable=False),
            pa.field("logit", pa.float32(), nullable=False),
        ]
    ),
}
```

Port `record_batch` and `empty_batch`, requiring column-oriented input and exact
schema names. Re-export or import the generic `ShardWriter`,
`is_shard_complete`, `read_shard_manifest`, and `validate_shard_manifest`
instead of duplicating their atomic I/O implementation.

- [ ] **Step 4: Run storage tests and verify GREEN**

Run:

```bash
uv run pytest tests/benchmarks/flenqa/test_storage.py -q
```

Expected: all schema and batch tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jlens_reasoning/benchmarks/flenqa/storage.py tests/benchmarks/flenqa/test_storage.py
git commit -m "refactor: define compact FLenQA output tables"
```

### Task 4: Implement Unique-Position Paired-Lens Execution

**Files:**

- Create: `src/jlens_reasoning/benchmarks/flenqa/runner.py`
- Create: `tests/benchmarks/flenqa/test_runner.py`
- Source: `experiments/flenqa_length_drift/readout.py`
- Source: `experiments/flenqa_length_drift/experiment.py`

- [ ] **Step 1: Write failing tests for unique execution and tolerant comparison**

Create focused fake-runner tests:

```python
def test_run_prompt_executes_duplicate_labels_once() -> None:
    prepared = _prepared(
        positions=(
            LabeledPosition("question_end", 7),
            LabeledPosition("final_prompt", 7),
            LabeledPosition("sampled_padding", 2),
        )
    )
    runners = _matching_runners(layers=(3, 9), vocab_size=5)

    batches = run_prompt(prepared, runners=runners, config=_config(top_k=2))

    assert runners.jacobian.requested_positions == [(2, 7)]
    assert runners.logit.requested_positions == [(2, 7)]
    assert batches["positions"].num_rows == 3
    assert batches["topk"].num_rows == 2 * 2 * 2 * 2


def test_run_prompt_accepts_allclose_logits_and_records_exact_max_diff() -> None:
    jacobian_logits = torch.tensor([[1.0, 2.0]])
    logit_logits = torch.tensor([[1.0, 2.0 + 5e-7]])
    runners = _runners_with_model_logits(jacobian_logits, logit_logits)

    batches = run_prompt(_prepared(), runners=runners, config=_config())

    expected = (
        jacobian_logits - logit_logits
    ).abs().max().item()
    assert batches["prompts"].to_pydict()[
        "max_abs_logit_diff"
    ] == pytest.approx([expected])


def test_run_prompt_rejects_logits_outside_tolerance() -> None:
    runners = _runners_with_model_logits(
        torch.tensor([[1.0]]), torch.tensor([[1.01]])
    )

    with pytest.raises(RuntimeError, match="allclose"):
        run_prompt(_prepared(), runners=runners, config=_config())


def test_run_prompt_preserves_returned_layer_keys() -> None:
    batches = run_prompt(
        _prepared(),
        runners=_matching_runners(layers=(4, 11), vocab_size=5),
        config=_config(top_k=1),
    )

    assert set(batches["topk"].to_pydict()["layer"]) == {4, 11}
```

Retain deterministic tie-breaking and tensor-row mismatch tests from the old
readout suite. Add a same-layer-keys failure test.

- [ ] **Step 2: Run the runner tests and verify RED**

Run:

```bash
uv run pytest tests/benchmarks/flenqa/test_runner.py -q
```

Expected: collection fails because `flenqa.runner` does not exist.

- [ ] **Step 3: Implement the minimal paired-lens API**

Define:

```python
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


@dataclass(frozen=True, slots=True)
class LensRunners:
    jacobian: LensRunner
    logit: LensRunner
```

Port `ApplyLensRunner`, passing `use_jacobian=True` and `False` through the two
instances. `run_prompt` must:

1. use `prepared.unique_positions`;
2. run both modes with those exact positions;
3. normalize and compare both input-ID sequences to `prepared.input_ids`;
4. require the same layer-key set;
5. validate one logits row per unique position for every layer;
6. require model-logit shapes to match;
7. compare model logits using:

```python
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
```

8. build one `positions` row per label;
9. build top-k rows for each lens kind, exact returned layer key, and unique
   position; and
10. assert:

```python
expected_topk_rows = (
    2
    * len(jacobian.logits_by_layer)
    * len(prepared.unique_positions)
    * min(config.top_k, vocabulary_size)
)
```

Port deterministic top-k tie-breaking unchanged. Do not port scoring,
bridge-target ranking, entropy, or summary reductions.

- [ ] **Step 4: Run runner tests and verify GREEN**

Run:

```bash
uv run pytest tests/benchmarks/flenqa/test_runner.py -q
```

Expected: all paired-lens tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jlens_reasoning/benchmarks/flenqa/runner.py tests/benchmarks/flenqa/test_runner.py
git commit -m "refactor: run paired lenses at semantic positions"
```

### Task 5: Add Atomic Shards, Resume, and Run Manifests

**Files:**

- Modify: `src/jlens_reasoning/benchmarks/flenqa/runner.py`
- Modify: `src/jlens_reasoning/benchmarks/flenqa/storage.py`
- Modify: `tests/benchmarks/flenqa/test_runner.py`
- Modify: `tests/benchmarks/flenqa/test_storage.py`

- [ ] **Step 1: Write failing interruption and manifest tests**

Adapt the old shard/run tests to the three-table output and add:

```python
def test_incomplete_shard_is_rebuilt_from_the_beginning(tmp_path: Path) -> None:
    partial = tmp_path / "topk" / "shard-00000.parquet"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")

    result = run_shard(
        _shard(0),
        _prepared_prompts(2),
        output_dir=tmp_path,
        runners=_matching_runners(),
        config=_config(),
    )

    assert result.prompt_ids == ("prompt-0", "prompt-1")
    assert is_shard_complete(
        tmp_path,
        shard_id=0,
        schemas=TABLE_SCHEMAS,
        required_tables=REQUIRED_TABLES,
    )


def test_run_manifest_summarizes_max_logit_difference(tmp_path: Path) -> None:
    manifest = run_benchmark(
        _rows(),
        output_dir=tmp_path,
        tokenizer=_tokenizer(),
        runners=_matching_runners(model_logit_diff=5e-7),
        config=_config(),
    )

    assert manifest.max_abs_logit_diff == pytest.approx(5e-7)
```

Retain tests that interrupted writes leave no completion manifest, completed
shards resume without rerunning, prompt membership changes are rejected, and
table checksum/schema changes invalidate completion.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run pytest \
  tests/benchmarks/flenqa/test_storage.py \
  tests/benchmarks/flenqa/test_runner.py -q
```

Expected: failures for missing shard orchestration and run manifest behavior.

- [ ] **Step 3: Implement shard and run orchestration**

Port `PromptShard`, `plan_shards`, `run_shard`, config hashing, and run-manifest
logic from the old driver. Make these simplifications:

- required shard tables are exactly `prompts`, `positions`, and `topk`;
- `run_shard` calls `run_prompt` once per prepared prompt and appends all three
  batches;
- generic `ShardWriter` removes known final/temp files when no valid completion
  manifest exists, writes temporary Parquet files, uses `os.replace`, and writes
  the completion manifest last;
- a completed shard is reused only after schema, checksum, row-count, and
  prompt-membership validation;
- `run-meta.json` records the canonical serialized `RunConfig`, config hash,
  requested layers, returned layers, and tolerances;
- resume rejects a different config hash before inspecting shards; and
- `run-manifest.json` records all shard IDs, prompt IDs, and the maximum of the
  per-prompt `max_abs_logit_diff` values.

Write both JSON files through `*.tmp` followed by `os.replace`. The run
manifest is written only after every shard validates.

Expose `run_benchmark(rows, *, output_dir, tokenizer, runners, config) ->
RunManifest`. It must first require `len(rows) ==
config.expected_source_rows`, then deduplicate, run `bridge_gate`, prepare positions with
`config.padding_sample_seed`, plan shards, run/resume them, validate all
manifests, summarize max logit difference, and atomically commit the run
manifest.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
uv run pytest \
  tests/benchmarks/flenqa/test_storage.py \
  tests/benchmarks/flenqa/test_runner.py -q
```

Expected: all storage and runner tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  src/jlens_reasoning/benchmarks/flenqa/runner.py \
  src/jlens_reasoning/benchmarks/flenqa/storage.py \
  tests/benchmarks/flenqa/test_runner.py \
  tests/benchmarks/flenqa/test_storage.py
git commit -m "feat: save resumable FLenQA lens shards"
```

### Task 6: Move and Simplify the Notebooks

**Files:**

- Create: `notebooks/flenqa_smoke.ipynb`
- Create: `notebooks/flenqa_full_run.ipynb`
- Modify: `tests/test_notebooks.py`
- Delete tracked: `experiments/flenqa_length_drift/flenqa_smoke.ipynb`
- Delete tracked: `experiments/flenqa_length_drift/flenqa_length_drift.ipynb`

- [ ] **Step 1: Update notebook tests first**

Replace FLenQA experiment discovery assertions with:

```python
FLENQA_NOTEBOOKS = [
    Path("notebooks/flenqa_smoke.ipynb"),
    Path("notebooks/flenqa_full_run.ipynb"),
]


def test_flenqa_notebooks_are_benchmark_drivers() -> None:
    forbidden = (
        "run_preflight(",
        "score_binary_answer(",
        "select_summary_positions(",
        "reduce_readout(",
        "ParquetWriter",
        "TABLE_SCHEMAS",
    )
    for path in FLENQA_NOTEBOOKS:
        source = "\n".join(cell.source for cell in load_notebook(path).cells)
        assert (
            "from jlens_reasoning.benchmarks.flenqa.runner import" in source
        )
        assert "run_benchmark(" in source
        assert not any(fragment in source for fragment in forbidden)
```

Add `FLENQA_NOTEBOOKS` to `NOTEBOOKS`, while `EXPERIMENT_NOTEBOOKS` contains
only `experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`.

- [ ] **Step 2: Run notebook tests and verify RED**

Run:

```bash
uv run pytest tests/test_notebooks.py -q
```

Expected: failures because the new notebook paths do not exist.

- [ ] **Step 3: Move and edit the tracked notebooks**

Move:

```bash
git mv experiments/flenqa_length_drift/flenqa_smoke.ipynb notebooks/flenqa_smoke.ipynb
git mv experiments/flenqa_length_drift/flenqa_length_drift.ipynb notebooks/flenqa_full_run.ipynb
```

Use `nbformat` to make only structured cell-source edits:

- import `ApplyLensRunner`, `LensRunners`, `RunConfig`, and `run_benchmark` from
  `jlens_reasoning.benchmarks.flenqa.runner`;
- import `normalize_rows` from `jlens_reasoning.benchmarks.flenqa.dataset`;
- remove bridge-gate and preflight cells because `run_benchmark` owns the gate;
- remove generation/scoring setup;
- call `run_benchmark`;
- keep the 80-row smoke limit with `expected_source_rows=80` and
  `expected_bridge_problems=2`;
- verify the published full schema before normalization and use
  `expected_source_rows=12_000` and `expected_bridge_problems=200` in
  `flenqa_full_run.ipynb`; and
- keep execution counts `None` and outputs empty.

Do not read, edit, move, or delete
`experiments/flenqa_length_drift/flenqa_smoke_output.ipynb`.

- [ ] **Step 4: Run notebook tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_notebooks.py -q
```

Expected: all notebook tests pass.

- [ ] **Step 5: Commit**

```bash
git add notebooks tests/test_notebooks.py
git add -u experiments/flenqa_length_drift
git commit -m "refactor: move FLenQA runners to notebooks"
```

### Task 7: Remove the Legacy Experiment and Update Live Documentation

**Files:**

- Delete: tracked Python files under `experiments/flenqa_length_drift/`
- Delete: `tests/experiments/flenqa_length_drift/`
- Delete: old flat FLenQA modules and tests after all imports move
- Modify: `README.md`
- Modify: any live runtime/test imports found by `rg`

- [ ] **Step 1: Add a failing no-legacy-import assertion**

Add to `tests/benchmarks/flenqa/test_dataset.py`:

```python
def test_legacy_flenqa_experiment_package_is_absent() -> None:
    assert not Path("experiments/flenqa_length_drift/__init__.py").exists()
```

- [ ] **Step 2: Run it and verify RED**

Run:

```bash
uv run pytest \
  tests/benchmarks/flenqa/test_dataset.py::test_legacy_flenqa_experiment_package_is_absent \
  -q
```

Expected: failure because the legacy package still exists.

- [ ] **Step 3: Update imports and remove tracked legacy files**

Use:

```bash
rg -n "experiments\\.flenqa_length_drift|benchmarks\\.flenqa_(prompts|preparation|conditions)" \
  src tests notebooks README.md
```

Update every live import to the new four-module package. Delete the tracked old
modules/tests with `git rm`, but exclude the untracked executed notebook. Remove
the now-empty tracked experiment directory without touching its untracked file.

Update README’s heading and instructions to “FLenQA benchmark runner,” point to
`notebooks/flenqa_smoke.ipynb` and `notebooks/flenqa_full_run.ipynb`, and
describe the three saved tables and paired lens modes.

Historical documents under `docs/superpowers/` remain unchanged.

- [ ] **Step 4: Verify no live legacy references remain**

Run:

```bash
rg -n "experiments\\.flenqa_length_drift|benchmarks\\.flenqa_(prompts|preparation|conditions)" \
  src tests notebooks README.md
```

Expected: no matches.

Run:

```bash
uv run pytest tests/benchmarks/flenqa tests/test_notebooks.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add src tests notebooks README.md
git add -u experiments
git commit -m "refactor: remove legacy FLenQA experiment"
```

### Task 8: Full Verification

**Files:**

- Modify only files required by failures directly caused by this migration.

- [ ] **Step 1: Run Ruff**

Run:

```bash
uv run ruff check .
```

Expected: exit 0 with no violations.

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run pytest tests/benchmarks/flenqa tests/test_notebooks.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run the full suite**

Run:

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Verify repository state and protected untracked files**

Run:

```bash
git status --short
git diff --check
```

Expected: only intentional migration changes plus the pre-existing untracked
`.claude/` directory and
`experiments/flenqa_length_drift/flenqa_smoke_output.ipynb`; no whitespace
errors.

- [ ] **Step 5: Commit any verification-only fixes**

If verification required an in-scope fix:

```bash
git add src tests notebooks README.md
git commit -m "fix: complete FLenQA benchmark migration"
```

If no fix was required, do not create an empty commit.
