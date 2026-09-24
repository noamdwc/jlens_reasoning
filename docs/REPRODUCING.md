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

## Colab workflow with `colab-utils`

Use `scripts/run_colab_notebook.sh` from this repository root. It calls the
sibling `../colab-utils/run.sh` runner. Install and authenticate the Google
Colab CLI.
The repository's [`.colab.env`](../.colab.env) selects the `side-projects` bucket and the
`jlens-reasoning` object prefix. Keep credentials in a private file outside
this repository:

```text
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
# HF_TOKEN=...  # required only for notebooks/01_download_assets.ipynb
```

The script defaults `R2_CREDENTIALS_FILE` to
`$HOME/.config/colab-utils/r2.env`. Pass it any notebook path:

```bash
./scripts/run_colab_notebook.sh notebooks/00_environment_check.ipynb
./scripts/run_colab_notebook.sh experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb
```

If your credential file is elsewhere, set `R2_CREDENTIALS_FILE` for that run:

```bash
R2_CREDENTIALS_FILE=/path/to/r2.env \
  ./scripts/run_colab_notebook.sh notebooks/00_environment_check.ipynb
```

The runner sends the current contents of files in Git's index, checks the
notebook's `check-environment` cell, and executes it in `/content/project`.
The notebooks export locked requirements with `uv`, install this project from
the sent source, and record a SHA-256 of their source bundle. The runner saves
executed notebooks under `artifacts/colab/` when
`NOTEBOOK_OUTPUT_DIR=artifacts/colab` is set in `.colab.env`.

Use the **same** `R2_DATA_PREFIX` and `R2_ARTIFACT_PREFIX`. Each notebook
downloads only the objects it needs from that prefix into
`/content/project/data`, removing the prefix from the local path. For example,
the key `jlens-reasoning/assets/models/qwen3.5-4b/` in the `side-projects`
bucket becomes
`/content/project/data/assets/models/qwen3.5-4b/` in Colab. New results go under
`/content/project/output`. After execution, `colab-utils` uploads only the
output files, preserving their paths under the same prefix:

```text
<R2_DATA_PREFIX>/
├── assets/models/qwen3.5-4b/
├── assets/lenses/qwen3.5-4b/
├── datasets/flenqa/
├── checkpoints/
└── runs/
```

Later notebooks read the previous run's files from `data/` and upload their
new files from `output/`. Running a notebook again replaces the R2 objects at
its output paths. Model backed notebooks download the pinned model assets, and
the drift notebook downloads the full top-k table; allow enough runtime disk
space for those inputs. Run dependent notebooks in the order below.
`EXPECT_GPU=true` requests a T4; set it to
`false` for the asset download notebook if desired. The runner does not run
multiple notebooks in one invocation.

Existing Drive assets and compatible `runs/` or `checkpoints/` can be copied
to these same R2 paths before running their consumers; they do not need to be
regenerated just because the Colab transport changed.

Keep the external credential file private. It is uploaded to the temporary
Colab VM. `colab-utils` stops the VM after the run;
its README describes recovery if upload fails. The local executed notebook
contains cell outputs and should remain under ignored `artifacts/` unless it
is intentionally released.

## Download model and lens assets

Run [`notebooks/01_download_assets.ipynb`](../notebooks/01_download_assets.ipynb)
once through `colab-utils`, with `HF_TOKEN` in the external credential file. It
downloads the pinned Qwen3.5-4B model snapshot and Neuronpedia Jacobian Lens
checkpoint into `output/assets/`, which the runner uploads to R2:

```text
<R2_DATA_PREFIX>/assets/
├── models/qwen3.5-4b/
└── lenses/qwen3.5-4b/Qwen3.5-4B_jacobian_lens_n1000.pt
```

Experiment notebooks download those assets from R2 and do not need Hugging Face
authentication after the asset download step. The current model and
lens revisions are declared in
[`experiments/jlens_readout_sanity/constants.py`](../experiments/jlens_readout_sanity/constants.py)
and the download notebook. Stage the external FLenQA `load_from_disk` export
under `<R2_DATA_PREFIX>/datasets/flenqa/` before benchmark runs.

## Run order

1. **Environment check:** run
   [`notebooks/00_environment_check.ipynb`](../notebooks/00_environment_check.ipynb)
   after changing environment setup. It checks runtime and R2 access but
   does not load a model or benchmark.
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
interrupted, `colab-utils` can still upload partial output shards. Remove or
move the incomplete `runs/flenqa-full-run/` objects in R2 before restarting;
a fresh VM has empty local output directories and cannot detect stale R2
shards. The raw model-output table preserves generated text, token IDs
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
├── problem_split.json
├── probes.pt
└── metadata.json
```

Generation, probe extraction, evaluation, and selected sensitivities use the
same direct chat template and final wrapped input token. Version 2 probes must
be retrained using the existing split; legacy raw-input probes are incompatible.
Saved answers must contain the actual input-token/mask hash from generation.
See the [probe × J-Lens migration instructions](../experiments/flenqa_probe_jlens/README.md)
for the rerun order. Each stage overwrites its outputs in the original directories;
the existing `problem_split.json` is preserved. No manual archiving is required.

The split is at the underlying-problem level. Probes are trained on 250/500
nominal-token rows from train problems, regularization is selected on
validation problems, and the held-out test problems are evaluated only by the
evaluation notebook. A held-out probe prediction demonstrates decodability
under the probe contract; it is not a causal intervention.

## Reproducibility boundary

The code, tests, notebook source, lockfile, and source bundle hashes make the
software path inspectable and help prevent stale-code runs. Exact model-backed
reproduction additionally requires:

- the FLenQA dataset export under the configured R2 data prefix. This
  repository validates the published count invariants but does not pin a
  dataset revision, so record the source revision or file hash with each run;
- the pinned Qwen3.5-4B and Neuronpedia lens assets;
- the Colab runtime/device and installed dependency versions;
- enough storage for Parquet top-k tables and model outputs; and
- the executed artifacts and their project source SHA-256.

GitHub Actions intentionally does not provide this environment. Its purpose is
to catch library, evaluation, notebook-structure, and packaging regressions
without credentials or networked model execution.
