# Environment Setup Design

**Date:** 2026-07-14

## Purpose

Create a reproducible Python environment for research built on Jacobian Lens. The
same repository must support lightweight development on an Apple Silicon Mac,
interactive GPU experiments through the IDE's Colab extension, and automated
CPU tests in GitHub Actions.

FLenQA is the first benchmark, but it is not part of the environment API. It is
one dataset artifact beneath a generic datasets directory. Later benchmarks
must be addable without changing environment initialization.

## Runtime Contract

- **Mac:** dependency management, editing, linting, unit tests, and small-model
  smoke experiments on CPU or MPS.
- **Colab:** interactive GPU experiments. Colab is accessed through the IDE and
  is not available to command-line scripts or automated pipelines.
- **GitHub Actions:** deterministic CPU-only validation with tiny or synthetic
  fixtures. CI does not mount Drive, use research secrets, or download model
  weights.
- **Google Drive:** persistent datasets, model caches, fitted lenses,
  checkpoints, and experiment results. GitHub remains the source of truth for
  code and notebooks.
- **Weights & Biases:** experiment tracking. Local runs use the existing terminal
  login. Colab enables W&B by default and fails initialization if W&B
  authentication fails. CI disables W&B.

## Dependency Management

Use a single `pyproject.toml` and committed `uv.lock` managed by uv. Python 3.11
is the pinned local and CI baseline. Project metadata accepts Python versions
from 3.11 through 3.13 so the active Colab interpreter can be used when it falls
inside that tested range.

The dependency groups are:

- Core runtime: PyTorch, Transformers, Hugging Face Hub, and NumPy.
- Experiment extra: datasets, W&B, and notebook support.
- Development group: pytest, Ruff, and notebook-validation tools.

Jacobian Lens is an immutable Git dependency pinned to upstream commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. It is not copied into this
repository and is not installed from a moving branch. If local patches become
necessary later, replacing the dependency with a maintained fork is a separate
design decision.

Local and CI commands use `uv sync --locked`. Colab clones an explicit project
Git ref, installs uv, exports the locked dependency set, installs that set into
the active notebook interpreter, and installs this project without resolving a
second dependency graph.

## Repository Structure

```text
jlens-reasoning/
├── .github/workflows/ci.yml
├── docs/
├── notebooks/
│   ├── _template.ipynb
│   └── 00_environment_check.ipynb
├── scripts/
│   └── colab_bootstrap.py
├── src/jlens_reasoning/
│   ├── __init__.py
│   ├── config.py
│   ├── runtime.py
│   ├── tracking.py
│   └── environments/
│       ├── __init__.py
│       ├── common.py
│       └── colab.py
├── tests/
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

Responsibilities are deliberately narrow:

- `config.py` resolves generic artifact directories and validates that they are
  writable.
- `runtime.py` selects CPU, MPS, or CUDA and supports an explicit requirement
  for CUDA.
- `tracking.py` holds W&B authentication and status behavior but does not start
  runs implicitly.
- `environments/common.py` creates a platform-neutral runtime context.
- `environments/colab.py` performs Colab-only Drive and secret integration.
- `scripts/colab_bootstrap.py` is a dependency-free installer that runs before
  the package can be imported.

## Artifact Storage

`JLENS_REAS_ARTIFACT_ROOT` is the sole public environment variable that selects
persistent artifact storage.

- Mac default: the ignored repository directory `artifacts/`.
- Colab default: `/content/drive/MyDrive/jlens-reasoning` after Drive is mounted.
- CI: a pytest-provided temporary directory.

The generic layout beneath the root is:

```text
datasets/
cache/huggingface/
lenses/
checkpoints/
runs/
```

FLenQA data is stored at `datasets/flenqa/`. This path is an artifact convention,
not a property or special case in the environment modules. Future benchmarks
use sibling directories beneath `datasets/`.

The application does not implement local-to-Drive synchronization. The user
syncs input data to Drive before a Colab run and syncs results back afterward.

## Reusable Colab Bootstrap

A fresh Colab kernel cannot import project modules before obtaining the project,
so each notebook needs one minimal loader cell. The cell contains only the Git
ref and enough standard-library code to:

1. Read `GITHUB_TOKEN_JLENS_REAS` from Colab Secrets.
2. Fetch the canonical `scripts/colab_bootstrap.py` from the selected Git ref
   with the token in an authorization header, never in the URL.
3. Execute the fetched bootstrap script.

The script clones the selected ref with temporary credentials, installs the
locked environment, removes the temporary credential mechanism, and verifies
that the saved Git remote contains no token. New experiment notebooks are copied
from `notebooks/_template.ipynb` so the stable loader cell is not reconstructed
manually.

After installation, notebooks initialize the shared environment with:

```python
from jlens_reasoning.environments.colab import initialize_colab

context = initialize_colab()
```

`initialize_colab()` mounts Drive, configures generic artifact paths, configures
the Hugging Face cache, authenticates external services, detects CUDA, and
returns a structured runtime context. The context exposes:

- `device`
- `artifact_root`
- `datasets_dir`
- `huggingface_cache`
- `lenses_dir`
- `checkpoints_dir`
- `runs_dir`
- `wandb_enabled`

It does not expose benchmark-specific paths.

## Authentication and Secret Handling

Colab reads these exact secret names:

- `GITHUB_TOKEN_JLENS_REAS` for authenticated GitHub access during bootstrap.
- `HF_READ_TOKEN` for read access to Hugging Face resources.
- `WANDB_API_KEY` for W&B authentication.

Secrets must not appear in notebook output, exception messages, subprocess
command display, Git remote URLs, W&B configuration committed to the repository,
or `.env.example`. Authentication errors identify the missing or failing service
without including credential values.

Local development relies on the user's existing W&B terminal login. This
project does not copy that credential into its own configuration. GitHub Actions
uses none of the Colab secrets.

## W&B Policy

Colab calls `initialize_colab(enable_wandb=True)` by default.

- When enabled, `WANDB_API_KEY` must exist and W&B login must succeed. A missing
  key, rejected key, or network/login error raises a clear initialization error.
- Passing `enable_wandb=False` skips W&B authentication explicitly.
- Initialization authenticates and validates W&B but never starts a run.
- Experiment code starts and finishes runs explicitly through the tracking
  module.
- CI sets `WANDB_MODE=disabled` and tests that no network call is attempted.

## Failure Behavior

Initialization fails rather than continuing with a misleading partial setup
when any of the following occurs:

- Python is outside the supported range.
- The selected Git ref cannot be fetched or the locked installation fails.
- Google Drive cannot be mounted.
- `JLENS_REAS_ARTIFACT_ROOT` is missing when required or is not writable.
- A required GitHub or Hugging Face credential is absent or rejected.
- W&B is enabled in Colab and its key or login fails.
- An experiment requests CUDA but CUDA is unavailable.

Device detection itself can return CPU or MPS for local development. Colab
experiments that require a GPU opt into `require_cuda=True` so accidentally
starting a CPU runtime fails early.

## Automated Verification

GitHub Actions performs:

- `uv lock --check`.
- Ruff formatting and lint checks.
- CPU-only pytest tests on Ubuntu.
- Lightweight compatibility tests on a macOS runner.
- Package and Jacobian Lens import smoke tests without model downloads.
- Tests for artifact path resolution, directory creation, device selection,
  W&B policy, and secret redaction.
- Mocked tests of the Colab initializer and dependency-free bootstrap script.
- Notebook validation ensuring the template uses the canonical loader and that
  committed notebooks contain no outputs or credentials.

The test environment sets:

```text
WANDB_MODE=disabled
JLENS_REAS_ARTIFACT_ROOT=/tmp/jlens-reasoning-test-artifacts
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Unit tests override the fixed CI path with a unique pytest temporary directory
when isolation between test cases is required.

Dependency installation may access package indexes and the pinned public
Jacobian Lens repository. Test execution performs no model, dataset, Drive, W&B,
or Hugging Face network access.

## Manual Colab Acceptance Check

`notebooks/00_environment_check.ipynb` is the manual integration check for the
IDE-connected Colab runtime. It verifies:

- The requested project Git ref is installed.
- Drive is mounted and `JLENS_REAS_ARTIFACT_ROOT` is writable.
- Hugging Face authentication succeeds.
- W&B authentication succeeds when enabled and fails clearly when required
  configuration is missing.
- CUDA is visible when requested.
- A small artifact can be written beneath `runs/` on Drive.

It does not download a large model or run FLenQA. Those belong to later
benchmark-specific work.

## Success Criteria

The environment setup is complete when:

1. A new Mac checkout can install the locked development environment and pass
   local tests with documented commands.
2. GitHub Actions passes the Linux checks and the lightweight macOS compatibility
   job without research secrets.
3. A new experiment notebook can reuse the template loader, initialize Colab
   through the shared module, authenticate the configured services, access
   Drive artifacts, and detect CUDA.
4. W&B is enabled and fail-fast by default in Colab, can be explicitly disabled,
   and remains disabled in CI.
5. Adding a second benchmark requires only a new directory beneath `datasets/`
   and benchmark-specific code, with no environment-module changes.

## Source References

- [Jacobian Lens repository](https://github.com/anthropics/jacobian-lens)
- [Pinned Jacobian Lens commit](https://github.com/anthropics/jacobian-lens/commit/581d398613e5602a5af361e1c34d3a92ea82ba8e)
- [uv project and lockfile documentation](https://docs.astral.sh/uv/guides/projects/)
- [uv GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/)
- [FLenQA repository](https://github.com/alonj/Same-Task-More-Tokens)
