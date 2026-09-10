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

### Unattended Colab CLI Drive access

The default unattended mode uses **user OAuth from the laptop's `jlens` rclone
remote**, including for personal My Drive. The runner extracts only `[jlens]`,
uploads it and the standalone bootstrap helper to `/content/jlens-credentials/`,
and skips `colab drivemount`. The notebook mounts that remote with rclone at
`/content/drive/MyDrive`, preserving the existing paths:

```text
<jlens remote root>/
├── jlens-reasoning/          # datasets, checkpoints, runs, cache
└── data/jlens-reasoning/     # wheels and model/lens assets
```

#### Mac setup and normal runs

The local `colab` CLI must already be authenticated. Install `rclone`, `uv`,
`jq`, and Python 3.11 or newer locally. If the existing `jlens` remote already
uploads wheels, reuse that configuration; no new Drive login is required per
run. For a new remote, run `rclone config` yourself once, choose Google Drive
with user OAuth, and name the remote `jlens`. See [rclone's Drive setup](https://rclone.org/drive/).

Config discovery order is `JLENS_RCLONE_CONFIG`, `RCLONE_CONFIG`, then
`$XDG_CONFIG_HOME/rclone/rclone.conf` (default `~/.config/rclone/rclone.conf`).
The source must be an unencrypted INI config with a Drive user-OAuth `jlens`
remote and a refresh token. Other remotes are not uploaded. Configs depending
on environment-only credentials or a service-account field are not accepted
as user OAuth. For an encrypted config, prepare a private, unencrypted config
containing only this remote locally and select it with `JLENS_RCLONE_CONFIG`.

```bash
# Optional: select a different local config; both uploader and runner use it.
export JLENS_RCLONE_CONFIG="$HOME/.config/rclone/rclone.conf"

# Verify the intended remote can see both project folders.
rclone lsd jlens: --config "$JLENS_RCLONE_CONFIG"

./scripts/upload_colab_wheel.sh
./scripts/run_colab_notebook.sh --gpu L4 notebooks/00_environment_check.ipynb
```

For a fresh setup, create the two project directories before running the
notebooks. The mount validates both paths and verifies a real API write and
readback before accepting the mount. Its temporary write probe is moved to
Drive trash after verification.

With no folder overrides, the saved remote root is used. Optionally set
`JLENS_DRIVE_ROOT_FOLDER_ID` to the **parent containing both project folders**;
this works for My Drive without a Shared Drive ID. The uploader and runner use
the same override without modifying the laptop config. Clear stale
`JLENS_DRIVE_SHARED_DRIVE_ID` / `JLENS_DRIVE_ROOT_FOLDER_ID` variables when
returning to the default My Drive root. The notebook runner uses the remote
named `jlens`; the uploader's standalone `--remote` option can target other
remotes. The chained `experiment_colab_run.sh` rejects remote names other than
`jlens` to prevent uploading a wheel to one remote and running from another.

#### Authentication precedence and security

`JLENS_DRIVE_AUTH` accepts:

| Value | Behavior |
| --- | --- |
| `auto` (default) | User `jlens` config, then SA key, then explicitly allowed interactive fallback |
| `rclone` | Require user `jlens` OAuth credentials |
| `service_account` | Require an SA key and Shared Drive configuration; ignore user config |
| `interactive` | Use interactive Drive authorization even if local credentials exist |

In auto mode, an existing default config without a `jlens` remote allows SA or
interactive fallback. An explicitly selected missing/invalid config, or an
invalid `jlens` entry, fails before VM allocation; auth failures never silently
fall back to browser prompts. `--allow-interactive-drivemount` (or
`JLENS_COLAB_ALLOW_INTERACTIVE_DRIVEMOUNT=1`) only permits fallback when no
credentials are available; it does not override available credentials.

**The Colab VM receives the selected remote's OAuth access and refresh tokens.**
Its effective permissions are those of that token: a full-Drive token allows
runtime code to access the user's entire Drive. Extracting one remote or setting
`root_folder_id` does not narrow OAuth permissions. Use a dedicated project
account or appropriately scoped authorization if stronger isolation is needed.
Do not run untrusted notebooks or dependencies with these credentials.

Never commit `rclone.conf`, SA keys, token exports, or `jlens.env`; never paste
tokens into notebook cells, command arguments, or logs. The runner uses a
private temporary directory (`0700`), credential files (`0600`), and a protected
VM credential directory. Staged local copies are removed on validation,
allocation, execution, and teardown failures as well as success. The original
laptop config is never rewritten. Tokens refreshed on the VM remain there;
they are not copied back over the laptop config. With `--keep`, or if teardown
cannot confirm uploads, credentials remain on the retained VM until it is
stopped. Revoked/expired refresh tokens require local reauthorization (for
example `rclone config reconnect jlens:`), then another run. Follow rclone's
[current OAuth client setup guidance](https://rclone.org/drive/#making-your-own-client-id)
if its shared default client is no longer supported.

`WANDB_API_KEY`, when set locally, is injected in `jlens.env` for unattended
runs because Colab Secrets are unavailable under `colab exec`. Browser runs
retain Secrets fallback. Use `enable_wandb=False` when tracking is unnecessary.
The separate initial asset-download notebook still uses browser Colab Secrets
for `HF_TOKEN`; existing experiments load the already-downloaded model assets.

#### Secondary service-account mode

SA mode remains available for writable **Workspace Shared Drives**. Google
[documents](https://developers.google.com/workspace/drive/api/guides/about-shareddrives)
that service accounts have no personal storage quota and cannot own files.
Sharing a regular My Drive folder may permit reads, but new checkpoint/run
writes can fail with `storageQuotaExceeded`. This runner does not use an SA
for writable My Drive or offer a read-only SA mode for these writing notebooks.
Personal Google accounts cannot create Shared Drives.

Store the key at `~/.config/jlens/drive-sa.json` or set `JLENS_DRIVE_SA_JSON`.
Grant the SA Content manager access to the intended Shared Drive and ensure
the laptop's rclone account can access it too. Use a project parent folder with
the layout above, then set:

```bash
export JLENS_DRIVE_AUTH=service_account
export JLENS_DRIVE_SHARED_DRIVE_ID="your-shared-drive-id"
export JLENS_DRIVE_ROOT_FOLDER_ID="your-project-folder-id"
./scripts/upload_colab_wheel.sh
./scripts/run_colab_notebook.sh notebooks/00_environment_check.ipynb
```

The IDs and required JSON key fields are validated locally before VM allocation.
Whether the key is valid and authorized is checked by the remote write probe. The same remote
write/readback and upload-draining checks apply to SA and user-OAuth mounts.

#### Upload completion and interactive fallback

At teardown, including notebook failures and `--keep` runs, the runner waits
up to ten minutes for rclone's queued and active uploads. Notebook writes must
be closed before the final cell finishes. If upload status is unavailable,
reports an error, or times out, the command fails and **keeps the VM** so cached
artifacts can be recovered. This may continue using Colab quota. Inspect the
kept session before stopping it:

```bash
printf '%s\n' 'import runpy' \
  'runpy.run_path("/content/jlens-credentials/colab_drive.py")["flush_colab_drive"]()' \
  | colab exec -s YOUR_SESSION --timeout 660
```

Stop it with `colab stop -s YOUR_SESSION` only after uploads succeed or after
recovering the needed files. Rclone status is exposed only on the VM's loopback
interface; inspect it with `rclone rc vfs/stats` there. The runner stops its VM
after a successful run unless `--keep` was requested.

For manual CLI authorization even when credentials are configured:

```bash
JLENS_DRIVE_AUTH=interactive ./scripts/run_colab_notebook.sh notebooks/00_environment_check.ipynb
```

For optional fallback only when no credentials are available:

```bash
./scripts/run_colab_notebook.sh --allow-interactive-drivemount notebooks/00_environment_check.ipynb
```

Agents must not select interactive mode for unattended work. Browser Colab
sessions without injected credentials continue using
`drive.mount("/content/drive")` and the existing My Drive layout.

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
