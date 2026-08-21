# FLenQA Accuracy by Prompt Length Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the reusable FLenQA accuracy framework with one transparent Colab notebook that evaluates all 9,862 unique prompts and plots paper-compatible accuracy by length.

**Architecture:** Keep only the shared paper-compatible binary evaluator in package code. Put model inference, result assembly, one-file Parquet persistence, weighted aggregation, and plotting directly in the notebook; deliberately omit resume and shard infrastructure.

**Tech Stack:** Python 3.11, PyTorch, Transformers, PyArrow/Parquet, pandas via the experiment environment, matplotlib in Colab, nbformat, pytest, ruff

---

## File map

- Keep `src/jlens_reasoning/evaluation.py` and `evaluation_utils.py`: shared paper-compatible verdict evaluation.
- Delete `src/jlens_reasoning/benchmarks/flenqa/accuracy.py`.
- Delete `src/jlens_reasoning/benchmarks/flenqa/accuracy_storage.py`.
- Delete `src/jlens_reasoning/benchmarks/flenqa/accuracy_analysis.py`.
- Delete `tests/benchmarks/flenqa/test_accuracy.py`.
- Delete `tests/benchmarks/flenqa/test_accuracy_storage.py`.
- Delete `tests/benchmarks/flenqa/test_accuracy_analysis.py`.
- Modify `src/jlens_reasoning/benchmarks/flenqa/__init__.py`: remove deleted public exports.
- Modify `notebooks/flenqa_accuracy.ipynb`: own the complete one-shot run and analysis.
- Modify `tests/test_notebooks.py`: enforce the simplified visible workflow.
- Modify `README.md`: document one-file output and restart-on-interruption behavior.

### Task 1: Pin the simplified notebook contract and remove the framework

**Files:**
- Modify: `tests/test_notebooks.py`
- Modify: `src/jlens_reasoning/benchmarks/flenqa/__init__.py`
- Delete: `src/jlens_reasoning/benchmarks/flenqa/accuracy.py`
- Delete: `src/jlens_reasoning/benchmarks/flenqa/accuracy_storage.py`
- Delete: `src/jlens_reasoning/benchmarks/flenqa/accuracy_analysis.py`
- Delete: `tests/benchmarks/flenqa/test_accuracy.py`
- Delete: `tests/benchmarks/flenqa/test_accuracy_storage.py`
- Delete: `tests/benchmarks/flenqa/test_accuracy_analysis.py`

- [ ] **Step 1: Change the notebook contract before implementation**

Replace the accuracy-notebook assertions in `tests/test_notebooks.py` with:

```python
def test_flenqa_accuracy_notebook_has_visible_full_run_workflow() -> None:
    notebook = load_notebook(FLENQA_ACCURACY_NOTEBOOK)
    cells = notebook_cells_by_id(FLENQA_ACCURACY_NOTEBOOK)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "initialize_colab(enable_wandb=False, require_cuda=True)" in source
    assert "normalize_rows(raw_rows, full=True)" in source
    assert "len(prompts) == 9_862" in source
    assert "for prompt in tqdm(prompts" in cells["run-accuracy"]
    assert "evaluate_paper_binary" in cells["run-accuracy"]
    assert "MAX_SEQ_LEN = 4096" in source
    assert "MAX_NEW_TOKENS = 64" in source
    assert "do_sample=False" in cells["define-generation"]
    assert "paper_weight" in cells["run-accuracy"]
    assert "pa.Table.from_pylist" in cells["save-results"]
    assert "pq.write_table" in cells["save-results"]
    assert '"results.parquet"' in cells["save-results"]
    assert "weighted_correct" in cells["paper-curve"]
    assert ".groupby(" in cells["paper-curve"]
    assert ".groupby(" in cells["unique-curve"]
    assert "run_accuracy(" not in source
    assert "load_accuracy_results(" not in source
    assert "run-manifest" not in source
    assert "shard" not in source.casefold()
    assert "JacobianLens" not in source
```

- [ ] **Step 2: Run the notebook test and verify RED**

Run `uv run pytest tests/test_notebooks.py::test_flenqa_accuracy_notebook_has_visible_full_run_workflow -q`.

Expected: failure because the current notebook delegates to `run_accuracy` and aggregation helpers.

- [ ] **Step 3: Remove framework code and exports**

Delete the six accuracy module/test files listed above. In
`benchmarks/flenqa/__init__.py`, remove every import and `__all__` entry for:

```text
AccuracyRunConfig
AccuracyRunManifest
AccuracyPoint
TokenLengthPoint
VerdictCountPoint
FULL_UNIQUE_PROMPT_COUNTS
load_accuracy_results
run_accuracy
summarize_paper_random
summarize_token_lengths
summarize_unique_prompts
summarize_verdicts
```

Keep the existing dataset exports byte-for-byte.

- [ ] **Step 4: Verify package regressions**

Run:

```bash
uv run pytest tests/benchmarks/flenqa tests/test_imports.py tests/test_package_discovery.py -q
uv run ruff check src/jlens_reasoning/benchmarks/flenqa tests/benchmarks/flenqa
```

Expected: existing FLenQA/package tests pass; the new notebook contract still
fails until Task 2.

- [ ] **Step 5: Commit**

```bash
git add src/jlens_reasoning/benchmarks/flenqa tests/benchmarks/flenqa tests/test_notebooks.py
git commit -m "refactor: remove FLenQA accuracy framework"
```

### Task 2: Make the notebook own inference, persistence, and analysis

**Files:**
- Modify: `notebooks/flenqa_accuracy.ipynb`

- [ ] **Step 1: Replace package API imports with notebook dependencies**

Import `pyarrow as pa`, `pyarrow.parquet as pq`, `pandas as pd`, and `tqdm.auto.tqdm`.
Import only `deduplicate` and `normalize_rows` from the FLenQA package, plus
`evaluate_paper_binary`, `GenerationStatus`, and `ModelOutput` from shared
evaluation code. Define:

```python
OUTPUT_DIR = context.runs_dir / "flenqa-accuracy"
RESULT_PATH = OUTPUT_DIR / "results.parquet"
LENGTHS = (250, 500, 1000, 2000, 3000)
EXPECTED_UNIQUE_COUNTS = {250: 300, 500: 2_368, 1000: 2_394, 2000: 2_400, 3000: 2_400}
MAX_SEQ_LEN = 4096
MAX_NEW_TOKENS = 64
```

- [ ] **Step 2: Return input length with each deterministic generation**

Define `generate_output` visibly in the notebook:

```python
def generate_output(prompt: str) -> tuple[int, ModelOutput]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = encoded["input_ids"].to(context.device)
    n_input_tokens = int(input_ids.shape[1])
    if n_input_tokens > MAX_SEQ_LEN:
        raise ValueError(
            f"Prompt exceeds {MAX_SEQ_LEN} tokens: {n_input_tokens}"
        )
    with torch.inference_mode():
        generated = causal_lm.generate(
            input_ids=input_ids,
            attention_mask=encoded["attention_mask"].to(context.device),
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    generated_ids = generated[0, n_input_tokens:].tolist()
    eos_ids = causal_lm.generation_config.eos_token_id
    eos_token_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or ())
    complete = bool(generated_ids and generated_ids[-1] in eos_token_ids)
    scored_ids = generated_ids[:-1] if complete else generated_ids
    return n_input_tokens, ModelOutput(
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

- [ ] **Step 3: Add the visible one-shot evaluation loop**

Replace the runner cell with:

```python
records = []
for prompt in tqdm(prompts, desc="FLenQA accuracy prompts", unit="prompt"):
    ctx_sizes = {item.ctx_size for item in prompt.provenance}
    if len(ctx_sizes) != 1:
        raise ValueError(f"Prompt {prompt.prompt_id} mixes context sizes")
    n_input_tokens, output = generate_output(prompt.text)
    evaluation = evaluate_paper_binary(output, expected=prompt.label)
    records.append(
        {
            "prompt_id": prompt.prompt_id,
            "problem_id": prompt.problem_id,
            "task": prompt.task,
            "label": prompt.label,
            "text": prompt.text,
            "ctx_size": ctx_sizes.pop(),
            "n_input_tokens": n_input_tokens,
            "paper_weight": sum(
                item.dispersion == "random" for item in prompt.provenance
            ),
            "model_name": MODEL_NAME,
            "code_revision": PROJECT_COMMIT,
            "generated_token_ids": list(output.token_ids),
            "generated_token_pieces": list(output.token_pieces),
            "generated_text": output.text,
            "generation_status": output.generation_status.value,
            "finish_reason": output.finish_reason,
            "verdict": evaluation.verdict,
            "correct": evaluation.correct,
        }
    )

assert len(records) == 9_862
assert Counter(record["ctx_size"] for record in records) == EXPECTED_UNIQUE_COUNTS
```

- [ ] **Step 4: Write one typed Parquet file**

Define a notebook-local Arrow schema with the record fields above. Make only
`finish_reason` and `verdict` nullable. Then:

```python
results = pa.Table.from_pylist(records, schema=RESULT_SCHEMA)
assert results.num_rows == 9_862
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pq.write_table(results, RESULT_PATH, compression="zstd")
RESULT_PATH
```

- [ ] **Step 5: Aggregate directly with pandas**

Create `frame = results.to_pandas()`. For the paper curve:

```python
frame["weighted_correct"] = frame["correct"].astype(int) * frame["paper_weight"]
frame["weighted_missing"] = frame["verdict"].isna().astype(int) * frame["paper_weight"]
paper_summary = (
    frame.groupby("ctx_size", as_index=False)
    .agg(
        correct=("weighted_correct", "sum"),
        total=("paper_weight", "sum"),
        no_verdict=("weighted_missing", "sum"),
    )
    .sort_values("ctx_size")
)
paper_summary["accuracy"] = paper_summary["correct"] / paper_summary["total"]
assert paper_summary["total"].tolist() == [600] * 5
```

For the unique curve:

```python
unique_summary = (
    frame.groupby("ctx_size", as_index=False)
    .agg(
        correct=("correct", "sum"),
        total=("prompt_id", "size"),
        no_verdict=("verdict", lambda values: values.isna().sum()),
    )
    .sort_values("ctx_size")
)
unique_summary["accuracy"] = unique_summary["correct"] / unique_summary["total"]
assert dict(zip(unique_summary["ctx_size"], unique_summary["total"], strict=True)) == EXPECTED_UNIQUE_COUNTS
```

Plot both summaries with the existing paper-style axes. Use pandas groupby for
task curves, verdict counts, and exact token-length min/median/max diagnostics.

- [ ] **Step 6: Verify notebook structure and syntax**

Run:

```bash
uv run ruff format notebooks/flenqa_accuracy.ipynb tests/test_notebooks.py
uv run pytest tests/test_notebooks.py -q
uv run ruff check notebooks/flenqa_accuracy.ipynb tests/test_notebooks.py
jq empty notebooks/flenqa_accuracy.ipynb
```

Expected: all notebook tests pass; notebook JSON and every non-loader code cell
are syntactically valid; no outputs or execution counts are stored.

- [ ] **Step 7: Commit**

```bash
git add notebooks/flenqa_accuracy.ipynb tests/test_notebooks.py
git commit -m "refactor: simplify FLenQA accuracy notebook"
```

### Task 3: Document and verify the simplified feature

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the runtime contract**

Replace resumable-shard wording with: the notebook writes one
`runs/flenqa-accuracy/results.parquet` file after all prompts complete; an
interrupted run restarts from the beginning; the artifact retains model name,
code revision, raw generation, token metadata, exact input length,
paper-compatible weight, verdict, and correctness.

- [ ] **Step 2: Run final verification**

Run:

```bash
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv lock --check
git diff --check
git status --short
```

Expected: formatting and lint are clean, all CPU tests pass, the lockfile is
unchanged, and only intended simplified-feature files differ from the base.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md
git commit -m "docs: describe simplified FLenQA accuracy run"
```
