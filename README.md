# jlens-reasoning

Research tooling for applying
[Jacobian Lens](https://github.com/anthropics/jacobian-lens) to reasoning
benchmarks. FLenQA is the first benchmark artifact; the environment is designed
to support additional benchmarks without package-level changes.

## Supported paths

- **Mac:** lightweight development, tests, and small CPU/MPS experiments.
- **Colab:** interactive GPU experiments launched through the IDE's Colab
  integration. Colab is intentionally not part of scripts or CI.
- **GitHub Actions:** secret-free CPU and macOS compatibility tests.

Python 3.11 is the baseline. Project metadata also supports Python 3.12 and 3.13.

## Local setup

Install `uv`, then run:

```bash
uv sync --locked --extra experiment
uv run pytest
uv run ruff format --check .
uv run ruff check .
```

Local artifacts default to the ignored `artifacts/` directory. Override the
location when needed:

```bash
export JLENS_REAS_ARTIFACT_ROOT=/absolute/path/to/artifacts
```

W&B uses the existing terminal login for experiment code. Environment setup
does not require W&B locally.

## Artifact layout

The directory selected by `JLENS_REAS_ARTIFACT_ROOT` contains:

```text
datasets/
cache/huggingface/
lenses/
checkpoints/
runs/
```

FLenQA data belongs at `datasets/flenqa/`. Additional benchmarks use sibling
directories. Data and experiment outputs are never committed.

For Colab, the default artifact root is:

```text
/content/drive/MyDrive/jlens-reasoning
```

Manually sync input data from local storage to this dedicated Drive folder
before an experiment, then sync results back afterward.

## Colab setup

Add this exact name to Colab Secrets when a notebook uses W&B:

- `WANDB_API_KEY`: W&B API key.

Before opening a notebook, build and upload the current Colab bundle from the
repository root:

```bash
./scripts/upload_colab_wheel.sh
```

The script exports locked project runtime requirements and uploads them beside
the project wheel and commit marker under `data/jlens-reasoning/wheels` on the
configured Drive remote. It excludes notebook extras and Colab-owned packages
so Colab keeps its preconfigured kernel, CUDA stack, NumPy, `fsspec`, and Rich.

Open `notebooks/_template.ipynb` through the IDE's Colab integration and run the
loader cell. It mounts Drive, installs the locked requirements, and
force-installs the uploaded wheel. Run the uploader again whenever project code
or dependencies change.

Before the first model-backed experiment, run
`notebooks/01_download_assets.ipynb`.

```python
from jlens_reasoning.environments.colab import initialize_colab

context = initialize_colab(require_cuda=True)
```

W&B is enabled by default in Colab and every login failure raises an error.
Disable it only when the notebook intentionally does not track an experiment:

```python
context = initialize_colab(enable_wandb=False, require_cuda=True)
```

Initialization mounts Drive, validates artifact writability, validates W&B when
enabled, selects the device, and returns generic artifact paths. Experiment
notebooks load their model and lens from Drive without Hugging Face
authentication. W&B authentication does not create a run.

Run `notebooks/00_environment_check.ipynb` after changing environment code. It
does not download a model or benchmark.

## J-Lens read-and-change sanity experiment

`experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb` is the first model-backed experiment.
Open it through the IDE's Colab integration with a GPU runtime and run all cells.
It uses the released `Qwen/Qwen3.5-4B` Jacobian lens, disables W&B, and writes
results beneath:

```text
runs/jlens-readout-sanity/
└── result.json
```

The experiment checks whether the J-Lens surfaces the unspoken `spider`
intermediate and whether clamped coordinate swaps causally redirect next-token
answers. It runs the paper's `spider`→`ant` example and the same
`France`→`China` swap across capital, language, continent, and currency prompts
at both the standard (`alpha=1`) and double (`alpha=2`) strengths. The result
artifact reports exact per-swap ranks and applies an open-model capability gate;
it does not claim numerical replication of Claude 4.5.

The notebook prints a text-only report for every configured check. `PASS` or
`FAIL` always reflects the Qwen sanity threshold used by the run. Where the
paper provides a directly comparable target, the report also shows the paper
gap as diagnostic context; that gap does not change pass/fail.

All experiments that grade model responses are expected to follow the
[LLM answer-evaluation policy](docs/llm-answer-evaluation.md). The policy keeps
raw generation, visible output, gold-blind extraction, normalization, and
scoring separate, distinguishes paper-faithful metrics from semantic
correctness, and records adoption status for evaluators that still need to be
migrated.

## FLenQA benchmark runner

Run `notebooks/flenqa_smoke.ipynb` before
`notebooks/flenqa_full_run.ipynb`. Both are thin Colab drivers over
`jlens_reasoning.benchmarks.flenqa`: they validate the dataset and bridge
spans, select meaningful fact, bridge, question, final-prompt, and
padding-content positions, then save Jacobian Lens and Logit Lens top-k values
at those positions.

Each immutable shard contains typed `prompts`, `positions`, and `topk` Parquet
tables. Temporary or partial shard files are rebuilt on resume; a shard or run
is complete only when its validated completion manifest exists.

## FLenQA accuracy by prompt length

Open `notebooks/flenqa_accuracy.ipynb` in a Colab GPU runtime after uploading
the current wheel. The notebook evaluates all 9,862 unique final FLenQA prompts
by default, then writes one result table after the full run completes:

```text
runs/flenqa-accuracy/results.parquet
```

The deliberately simple notebook does not checkpoint or resume partial runs;
if generation is interrupted, rerun it from the beginning.

Generation uses the shared Hugging Face chat-inference module. The
paper-compatible curve runs Qwen in direct mode with its native chat template,
thinking explicitly disabled, deterministic decoding, and the paper wrapper's
400-token completion allowance. The saved table records the effective inference
mode and decoding settings alongside raw output, structured reasoning/answer
fields, exact wrapped input length, parsed verdict, and correctness.

Results produced by the earlier raw-prompt, 64-token notebook are not comparable
and should be regenerated rather than appended to the corrected result table.

For paper compatibility, the behavioral score uses the final standalone,
case-insensitive `True` or `False` in each response and reports the published
nominal length buckets of 250, 500, 1000, 2000, and 3000 tokens. The notebook
asserts the expected unique-prompt and paper-weighted counts before saving.

The headline curve restores the paper's random-placement source-row weighting.
A second curve weights each unique prompt once, avoiding duplicate input weight
at the shortest length and for incidental prompt collisions. The notebook also
shows task-level accuracy, verdict frequencies, and measured token-length
diagnostics.

## CI policy

CI installs the committed `uv.lock`, disables W&B, sets Hugging Face and
Transformers offline modes for test runtime, and uses a temporary artifact root.
CI tests imports and mocked setup behavior only; it never uses repository,
Hugging Face, W&B, or Google Drive credentials.
