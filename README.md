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

Add these exact names to Colab Secrets:

- `GITHUB_TOKEN_JLENS_REAS`: GitHub token with read access to this repository.
- `HF_READ_TOKEN`: Hugging Face read token.
- `WANDB_API_KEY`: W&B API key.

Open `notebooks/_template.ipynb` through the IDE's Colab integration. Set
`PROJECT_REF` to an explicit branch, tag, or full commit SHA, run the loader
cell, then initialize. The bootstrap preserves Colab's CUDA-enabled PyTorch
and preloaded NumPy, and installs the remaining project and experiment
dependencies from the committed lockfile.

```python
from jlens_reasoning.environments.colab import initialize_colab

context = initialize_colab(require_cuda=True)
```

W&B is enabled by default in Colab and every login failure raises an error.
Disable it only when the notebook intentionally does not track an experiment:

```python
context = initialize_colab(enable_wandb=False, require_cuda=True)
```

Initialization mounts Drive, validates artifact writability, authenticates
Hugging Face, validates W&B when enabled, selects the device, and returns
generic artifact paths. It authenticates W&B but does not create a run.

Run `notebooks/00_environment_check.ipynb` after changing environment code. It
does not download a model or benchmark.

## J-Lens read-and-change sanity experiment

`experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb` is the first model-backed experiment.
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
intermediate and whether clamped coordinate swaps causally redirect next-token
answers. It runs the paper's `spider`→`ant` example and the same
`France`→`China` swap across capital, language, continent, and currency prompts
at both the standard (`alpha=1`) and double (`alpha=2`) strengths. The result
artifact reports exact per-swap ranks and applies an open-model capability gate;
it does not claim numerical replication of Claude 4.5.

All experiments that grade model responses are expected to follow the
[LLM answer-evaluation policy](docs/llm-answer-evaluation.md). The policy keeps
raw generation, visible output, gold-blind extraction, normalization, and
scoring separate, distinguishes paper-faithful metrics from semantic
correctness, and records adoption status for evaluators that still need to be
migrated.

## CI policy

CI installs the committed `uv.lock`, disables W&B, sets Hugging Face and
Transformers offline modes for test runtime, and uses a temporary artifact root.
CI tests imports and mocked setup behavior only; it never uses repository,
Hugging Face, W&B, or Google Drive credentials.
