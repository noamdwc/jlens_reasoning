# Reproducing the experiments

The repository separates model-free development from model-backed research
runs:

- **Local Mac/Linux:** package development, CPU tests, and small CPU/MPS checks.
- **Colab GPU:** model-backed FLenQA and Jacobian Lens notebooks.
- **GitHub Actions:** secret-free, offline CPU compatibility checks.

The model-backed path requires external FLenQA data, model/lens assets, a
configured Colab runtime, and enough storage for generated artifacts. The
repository does not contain those assets or generated result tables.

## Local setup

Python 3.11 is the baseline; project metadata supports Python 3.11–3.13.
Install [`uv`](https://docs.astral.sh/uv/) and run from the repository root:

```bash
uv sync --locked --extra experiment
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv lock --check
```

Tests are CPU-only and model-free. They disable W&B and use offline Hugging
Face/Transformers settings where relevant. They do not require credentials.

Local artifacts default to the ignored `artifacts/` directory. Set
`JLENS_REAS_ARTIFACT_ROOT` to use another writable location:

```bash
export JLENS_REAS_ARTIFACT_ROOT=/absolute/path/to/artifacts
```

The artifact root contains:

```text
datasets/
cache/huggingface/
lenses/
checkpoints/
runs/
```

Data and generated outputs are never committed. The `artifacts/` directory is
also the default destination for executed notebook copies produced by the
Colab CLI.

## Colab bundle workflow

Colab notebooks install the exact project wheel and locked runtime requirements
from a Drive folder. From the repository root, configure an `rclone` remote and
run:

```bash
./scripts/upload_colab_wheel.sh
```

The uploader exports locked non-development requirements, builds the wheel,
and uploads them with `project-commit.txt` and `project-dirty.txt` under:

```text
<rclone-remote>:data/jlens-reasoning/wheels/
```

The canonical loader cell in each notebook mounts Drive, validates those
markers, installs the requirements, and force-installs the wheel. Re-run the
uploader after any project-code or dependency change. Otherwise Colab can run
stale code. The uploader refuses a dirty tree unless `--allow-dirty` is
explicitly provided.

The `scripts/experiment_colab_run.sh` helper chains the upload and notebook
execution for an experiment package:

```bash
./scripts/experiment_colab_run.sh --gpu L4 jlens_readout_sanity
```

The standalone runner is useful for shared notebooks:

```bash
./scripts/run_colab_notebook.sh --gpu L4 notebooks/flenqa_full_run.ipynb
```

Both scripts require the local `colab` CLI. The runner writes the executed
notebook copy under `artifacts/colab/` by default and fails if a cell reports an
error.

## Download model and lens assets

Run [`notebooks/01_download_assets.ipynb`](../notebooks/01_download_assets.ipynb)
once in Colab with `HF_TOKEN` stored in Colab Secrets. It downloads the pinned
Qwen3.5-4B model snapshot and the pinned Neuronpedia Jacobian Lens checkpoint
to the Drive asset root:

```text
/content/drive/MyDrive/data/jlens-reasoning/assets/
├── models/qwen3.5-4b/
└── lenses/qwen3.5-4b/Qwen3.5-4B_jacobian_lens_n1000.pt
```

The experiment notebooks load those assets locally from Drive and do not need
Hugging Face authentication after the download step. The current model and
lens revisions are declared in
[`experiments/jlens_readout_sanity/constants.py`](../experiments/jlens_readout_sanity/constants.py)
and the download notebook.

## Run order

1. **Environment check:** run
   [`notebooks/00_environment_check.ipynb`](../notebooks/00_environment_check.ipynb)
   after changing environment setup. It checks runtime and artifact paths but
   does not download a model or benchmark.
2. **Assets:** run
   [`notebooks/01_download_assets.ipynb`](../notebooks/01_download_assets.ipynb)
   when the pinned model/lens files are not already present.
3. **J-Lens sanity:** run
   [`experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`](../experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb)
   on a GPU runtime. It writes `runs/jlens-readout-sanity/result.json`.
4. **FLenQA benchmark:** run
   [`notebooks/flenqa_smoke.ipynb`](../notebooks/flenqa_smoke.ipynb)
   before [`notebooks/flenqa_full_run.ipynb`](../notebooks/flenqa_full_run.ipynb).
   The smoke run validates a bounded subset; the full run writes lens shards
   and raw model outputs.
5. **Accuracy:** run
   [`notebooks/flenqa_accuracy.ipynb`](../notebooks/flenqa_accuracy.ipynb)
   after the full run. It reads raw model outputs and writes the separate
   scored result table.
6. **Drift:** run
   [`experiments/flenqa_lens_drift/flenqa_lens_drift.ipynb`](../experiments/flenqa_lens_drift/flenqa_lens_drift.ipynb)
   after full-run and accuracy artifacts exist.
7. **Probes:** run
   [`notebooks/flenqa_probe_assets.ipynb`](../notebooks/flenqa_probe_assets.ipynb)
   followed by [`notebooks/flenqa_probe_eval.ipynb`](../notebooks/flenqa_probe_eval.ipynb)
   to create and evaluate the frozen per-layer representation probes.
8. **Intervention pilots:** run
   [`experiments/flenqa_lens_drift/flenqa_lens_intervention.ipynb`](../experiments/flenqa_lens_drift/flenqa_lens_intervention.ipynb)
   or [`flenqa_failure_concept_intervention.ipynb`](../experiments/flenqa_lens_drift/flenqa_failure_concept_intervention.ipynb)
   for the explicitly configured matched-prompt pilots.

Every notebook is committed without saved output. To make a public result
claim, preserve the executed output or a compact derived report with exact
model, lens, data, code, and decoding provenance; do not copy numbers from an
untracked session into the README.

## Artifact contracts

### J-Lens sanity

The sanity notebook writes:

```text
runs/jlens-readout-sanity/result.json
```

The result includes per-case baseline evaluation, J-Lens and Logit Lens ranks,
intervention conditions, control results, thresholds, and project/dependency
provenance. The configured pass/fail gates are defined in the experiment
package; a pass is not a claim of paper replication.

### FLenQA full run and accuracy

The full run writes three non-resumable Parquet table directories plus raw
model outputs:

```text
runs/flenqa-full-run/
├── prompts/shard-*.parquet
├── positions/shard-*.parquet
├── topk/shard-*.parquet
└── model_outputs.parquet
```

The runner refuses to append to non-empty table directories. If it is
interrupted, remove or move the incomplete run and restart into empty output
directories. The raw model-output table preserves generated text, token IDs
and pieces, answer fields, status fields, inference settings, measured wrapped
input length, nominal FLenQA length, prompt provenance, and code revision.

The accuracy notebook writes:

```text
runs/flenqa-accuracy/results.parquet
```

It applies the versioned FLenQA paper-compatible binary scorer after generation;
it does not regenerate model responses. The paper-weighted curve and
unique-prompt curve answer different aggregation questions and should remain
separate.

### Frozen probe assets

The probe workflow writes:

```text
checkpoints/flenqa-probe-assets/
└── problem_split.json
checkpoints/flenqa-probe-assets-chat-v2/
├── probes.pt
└── metadata.json
```

Generation, probe extraction, evaluation, and selected sensitivities use the
same direct chat template and final wrapped input token. Version 2 probes must
be retrained using the existing split; legacy raw-input probes are incompatible.
Saved answers must contain the actual input-token/mask hash from generation.
See the [probe × J-Lens migration instructions](../experiments/flenqa_probe_jlens/README.md)
for the rerun order and separate result directories.

The split is at the underlying-problem level. Probes are trained on 250/500
nominal-token rows from train problems, regularization is selected on
validation problems, and the held-out test problems are evaluated only by the
evaluation notebook. A held-out probe prediction demonstrates decodability
under the probe contract; it is not a causal intervention.

## Reproducibility boundary

The code, tests, notebook source, lockfile, and commit markers make the
software path inspectable and help prevent stale-code runs. Exact model-backed
reproduction additionally requires:

- the FLenQA dataset export in the configured Drive data directory. This
  repository validates the published count invariants but does not pin a
  dataset revision, so record the source revision or file hash with each run;
- the pinned Qwen3.5-4B and Neuronpedia lens assets;
- the Colab runtime/device and installed dependency versions;
- enough storage for Parquet top-k tables and model outputs; and
- the executed artifacts and their project commit marker.

GitHub Actions intentionally does not provide this environment. Its purpose is
to catch library, evaluation, notebook-structure, and packaging regressions
without credentials or networked model execution.
