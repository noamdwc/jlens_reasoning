# Environment Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one reproducible Python environment for lightweight Mac development, interactive GPU-backed Colab experiments, and secret-free GitHub Actions tests around Jacobian Lens.

**Architecture:** A `uv`-managed `src` package owns artifact paths, device selection, authentication, and reusable Colab initialization. Colab notebooks retain only a small authenticated loader cell; all substantial setup lives in tested repository modules. GitHub Actions exercises the same lockfile and package on Ubuntu and macOS without downloading models, datasets, or contacting experiment services.

**Tech Stack:** Python 3.11 baseline, `uv`, PyTorch, Jacobian Lens pinned from Git, Hugging Face Hub, W&B, pytest, Ruff, Jupyter notebooks, Google Colab, GitHub Actions.

**Design reference:** `docs/superpowers/specs/2026-07-14-environment-setup-design.md`

---

## Task 1: Establish the locked Python project

**Files:**

- Create: `.python-version`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/jlens_reasoning/__init__.py`
- Create: `tests/test_imports.py`
- Create: `uv.lock`

- [ ] **Step 1: Write the failing package smoke test**

Create `tests/test_imports.py`:

```python
def test_project_and_jacobian_lens_import() -> None:
    import jlens
    import jlens_reasoning

    assert jlens_reasoning.__version__ == "0.1.0"
    assert jlens is not None
```

- [ ] **Step 2: Run the smoke test and confirm it fails**

Run:

```bash
uv run --no-project --with pytest pytest tests/test_imports.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `jlens` or `jlens_reasoning`.

- [ ] **Step 3: Add project metadata and dependency groups**

Create `.python-version`:

```text
3.11
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "jlens-reasoning"
version = "0.1.0"
description = "Research tooling for Jacobian Lens reasoning benchmarks"
readme = "README.md"
requires-python = ">=3.11,<3.14"
dependencies = [
  "huggingface-hub",
  "jlens",
  "numpy",
  "torch",
  "transformers>=5.5",
]

[project.optional-dependencies]
experiment = [
  "datasets",
  "ipykernel",
  "wandb",
]

[dependency-groups]
dev = [
  "nbformat",
  "pytest",
  "ruff",
]

[tool.uv.sources]
jlens = { git = "https://github.com/anthropics/jacobian-lens", rev = "581d398613e5602a5af361e1c34d3a92ea82ba8e" }
torch = [
  { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["B", "E", "F", "I", "UP", "W"]
ignore = ["E501"]
```

Create `src/jlens_reasoning/__init__.py`:

```python
"""Research tooling for Jacobian Lens reasoning benchmarks."""

__version__ = "0.1.0"
```

Create `.env.example`:

```dotenv
JLENS_REAS_ARTIFACT_ROOT=artifacts
# Configure Colab credentials in Colab Secrets, never in this file.
```

Create `.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
.env
artifacts/
wandb/
.DS_Store
```

Create a minimal `README.md` for this task:

```markdown
# jlens-reasoning

Research tooling for applying [Jacobian Lens](https://github.com/anthropics/jacobian-lens)
to reasoning benchmarks.

Environment setup and usage instructions are added in Task 8.
```

- [ ] **Step 4: Generate and install the cross-platform lockfile**

Run:

```bash
uv lock
uv sync --locked --extra experiment
```

Expected: `uv.lock` is created; the editable project and the exact Jacobian Lens Git revision install successfully.

- [ ] **Step 5: Re-run the smoke test**

Run:

```bash
uv run pytest tests/test_imports.py -v
```

Expected: PASS without downloading any model or benchmark data.

- [ ] **Step 6: Commit the project skeleton**

```bash
git add .python-version .gitignore .env.example pyproject.toml README.md src/jlens_reasoning/__init__.py tests/test_imports.py uv.lock
git commit -m "build: establish locked Python environment"
```

## Task 2: Add benchmark-agnostic artifact configuration

**Files:**

- Create: `src/jlens_reasoning/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write artifact-path tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from jlens_reasoning.config import (
    ARTIFACT_ROOT_ENV,
    ArtifactPaths,
    create_artifact_paths,
)


def test_explicit_artifact_root_creates_generic_directories(tmp_path: Path) -> None:
    root = tmp_path / "research-artifacts"

    paths = create_artifact_paths(root)

    assert paths == ArtifactPaths(
        root=root,
        datasets=root / "datasets",
        huggingface_cache=root / "cache" / "huggingface",
        lenses=root / "lenses",
        checkpoints=root / "checkpoints",
        runs=root / "runs",
    )
    assert all(path.is_dir() for path in paths.directories)


def test_artifact_root_comes_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "from-environment"
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(root))

    assert create_artifact_paths().root == root


def test_default_artifact_root_is_local_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ARTIFACT_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    assert create_artifact_paths().root == tmp_path / "artifacts"


def test_invalid_artifact_root_raises_redacted_error(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Artifact root is not writable"):
        create_artifact_paths(blocked)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
uv run pytest tests/test_config.py -v
```

Expected: FAIL because `jlens_reasoning.config` does not exist.

- [ ] **Step 3: Implement artifact-path resolution and writability validation**

Create `src/jlens_reasoning/config.py`:

```python
"""Artifact storage configuration shared by Mac, Colab, and CI."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

ARTIFACT_ROOT_ENV = "JLENS_REAS_ARTIFACT_ROOT"


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Benchmark-agnostic locations for datasets and experiment artifacts."""

    root: Path
    datasets: Path
    huggingface_cache: Path
    lenses: Path
    checkpoints: Path
    runs: Path

    @property
    def directories(self) -> tuple[Path, ...]:
        return (
            self.datasets,
            self.huggingface_cache,
            self.lenses,
            self.checkpoints,
            self.runs,
        )


def create_artifact_paths(root: str | Path | None = None) -> ArtifactPaths:
    """Resolve, create, and validate the configured artifact tree."""

    configured_root = root or os.environ.get(ARTIFACT_ROOT_ENV)
    artifact_root = (
        Path(configured_root).expanduser()
        if configured_root
        else Path.cwd() / "artifacts"
    )
    artifact_root = artifact_root.resolve()

    paths = ArtifactPaths(
        root=artifact_root,
        datasets=artifact_root / "datasets",
        huggingface_cache=artifact_root / "cache" / "huggingface",
        lenses=artifact_root / "lenses",
        checkpoints=artifact_root / "checkpoints",
        runs=artifact_root / "runs",
    )

    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        for directory in paths.directories:
            directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=artifact_root, prefix=".write-test-", delete=True
        ):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Artifact root is not writable: {artifact_root}"
        ) from exc

    return paths
```

The API deliberately exposes `datasets`, not `flenqa`. FLenQA will occupy
`paths.datasets / "flenqa"` as an artifact, and later benchmarks can be siblings.

- [ ] **Step 4: Run the artifact tests**

Run:

```bash
uv run pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit artifact configuration**

```bash
git add src/jlens_reasoning/config.py tests/test_config.py
git commit -m "feat: add generic artifact configuration"
```

## Task 3: Add device selection and the shared runtime context

**Files:**

- Create: `src/jlens_reasoning/runtime.py`
- Create: `src/jlens_reasoning/environments/__init__.py`
- Create: `src/jlens_reasoning/environments/common.py`
- Create: `tests/test_runtime.py`

- [ ] **Step 1: Write device and context tests**

Create `tests/test_runtime.py`:

```python
from pathlib import Path

import pytest
import torch

from jlens_reasoning.config import create_artifact_paths
from jlens_reasoning.environments.common import create_runtime_context
from jlens_reasoning.runtime import select_device


def test_cuda_is_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert select_device().type == "cuda"


def test_mps_is_used_for_lightweight_mac_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert select_device().type == "mps"


def test_cpu_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert select_device().type == "cpu"


def test_required_cuda_fails_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was required"):
        select_device(require_cuda=True)


def test_runtime_context_exposes_notebook_facing_fields(tmp_path: Path) -> None:
    paths = create_artifact_paths(tmp_path)
    context = create_runtime_context(
        paths=paths,
        device=torch.device("cpu"),
        wandb_enabled=False,
    )

    assert context.artifact_root == tmp_path
    assert context.datasets_dir == tmp_path / "datasets"
    assert context.huggingface_cache == tmp_path / "cache" / "huggingface"
    assert context.lenses_dir == tmp_path / "lenses"
    assert context.checkpoints_dir == tmp_path / "checkpoints"
    assert context.runs_dir == tmp_path / "runs"
    assert context.device.type == "cpu"
    assert context.wandb_enabled is False
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
uv run pytest tests/test_runtime.py -v
```

Expected: FAIL because the runtime modules do not exist.

- [ ] **Step 3: Implement device selection**

Create `src/jlens_reasoning/runtime.py`:

```python
"""Compute-device selection."""

import torch


def select_device(*, require_cuda: bool = False) -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if require_cuda:
        raise RuntimeError("CUDA was required but is not available")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
```

- [ ] **Step 4: Implement the reusable runtime context**

Create `src/jlens_reasoning/environments/common.py`:

```python
"""Shared environment context returned to notebooks and scripts."""

from dataclasses import dataclass
from pathlib import Path

import torch

from jlens_reasoning.config import ArtifactPaths


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    device: torch.device
    artifact_root: Path
    datasets_dir: Path
    huggingface_cache: Path
    lenses_dir: Path
    checkpoints_dir: Path
    runs_dir: Path
    wandb_enabled: bool


def create_runtime_context(
    *,
    paths: ArtifactPaths,
    device: torch.device,
    wandb_enabled: bool,
) -> RuntimeContext:
    return RuntimeContext(
        device=device,
        artifact_root=paths.root,
        datasets_dir=paths.datasets,
        huggingface_cache=paths.huggingface_cache,
        lenses_dir=paths.lenses,
        checkpoints_dir=paths.checkpoints,
        runs_dir=paths.runs,
        wandb_enabled=wandb_enabled,
    )
```

Create `src/jlens_reasoning/environments/__init__.py`:

```python
"""Runtime setup for supported execution environments."""

from jlens_reasoning.environments.common import RuntimeContext

__all__ = ["RuntimeContext"]
```

- [ ] **Step 5: Run the runtime tests**

Run:

```bash
uv run pytest tests/test_runtime.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit runtime selection**

```bash
git add src/jlens_reasoning/runtime.py src/jlens_reasoning/environments tests/test_runtime.py
git commit -m "feat: add shared runtime context"
```

## Task 4: Add strict, optional W&B authentication

**Files:**

- Create: `src/jlens_reasoning/tracking.py`
- Create: `tests/test_tracking.py`

- [ ] **Step 1: Write W&B behavior and redaction tests**

Create `tests/test_tracking.py`:

```python
from collections.abc import Callable

import pytest

from jlens_reasoning.tracking import authenticate_wandb


def test_disabled_wandb_does_not_call_login() -> None:
    def unexpected_login(**_: object) -> bool:
        raise AssertionError("login must not run")

    assert (
        authenticate_wandb(
            api_key=None,
            enabled=False,
            login=unexpected_login,
        )
        is False
    )


def test_enabled_wandb_requires_a_key() -> None:
    with pytest.raises(RuntimeError, match="WANDB_API_KEY is missing"):
        authenticate_wandb(api_key=None, enabled=True)


def test_enabled_wandb_verifies_server_authentication() -> None:
    calls: list[dict[str, object]] = []

    def successful_login(**kwargs: object) -> bool:
        calls.append(kwargs)
        return True

    assert authenticate_wandb(
        api_key="secret-value",
        enabled=True,
        login=successful_login,
    )
    assert calls == [
        {
            "key": "secret-value",
            "relogin": True,
            "verify": True,
        }
    ]


@pytest.mark.parametrize(
    "login",
    [
        lambda **_: False,
        lambda **_: (_ for _ in ()).throw(ConnectionError("secret-value")),
    ],
)
def test_enabled_wandb_failure_is_fatal_and_redacted(
    login: Callable[..., bool],
) -> None:
    with pytest.raises(RuntimeError, match="W&B authentication failed") as error:
        authenticate_wandb(
            api_key="secret-value",
            enabled=True,
            login=login,
        )

    assert "secret-value" not in str(error.value)
    assert error.value.__cause__ is None
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
uv run pytest tests/test_tracking.py -v
```

Expected: FAIL because `jlens_reasoning.tracking` does not exist.

- [ ] **Step 3: Implement fail-fast W&B authentication**

Create `src/jlens_reasoning/tracking.py`:

```python
"""Experiment-tracking authentication."""

from __future__ import annotations

from collections.abc import Callable


def authenticate_wandb(
    *,
    api_key: str | None,
    enabled: bool = True,
    login: Callable[..., bool] | None = None,
) -> bool:
    """Authenticate W&B when enabled and fail on every authentication error."""

    if not enabled:
        return False

    if not api_key:
        raise RuntimeError("W&B is enabled but WANDB_API_KEY is missing")

    if login is None:
        import wandb

        login = wandb.login

    try:
        authenticated = login(
            key=api_key,
            relogin=True,
            verify=True,
        )
    except Exception:
        raise RuntimeError("W&B authentication failed") from None

    if not authenticated:
        raise RuntimeError("W&B authentication failed")

    return True
```

`verify=True` is essential: Colab setup must validate the key against the W&B
server, not merely persist it locally. This function authenticates only; it does
not call `wandb.init()` or create a run.

- [ ] **Step 4: Run the W&B tests**

Run:

```bash
uv run pytest tests/test_tracking.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit tracking authentication**

```bash
git add src/jlens_reasoning/tracking.py tests/test_tracking.py
git commit -m "feat: add strict optional wandb authentication"
```

## Task 5: Build reusable Colab initialization

**Files:**

- Create: `src/jlens_reasoning/environments/colab.py`
- Create: `tests/test_colab_environment.py`

- [ ] **Step 1: Write mocked Colab initialization tests**

Create `tests/test_colab_environment.py`:

```python
from pathlib import Path

import pytest
import torch

from jlens_reasoning.environments.colab import initialize_colab


def test_colab_initialization_mounts_drive_and_authenticates(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    secrets = {
        "HF_READ_TOKEN": "hf-secret",
        "WANDB_API_KEY": "wandb-secret",
    }

    context = initialize_colab(
        artifact_root=tmp_path,
        secret_getter=secrets.__getitem__,
        drive_mounter=lambda: events.append("drive-mounted"),
        hf_authenticator=lambda **kwargs: events.append(("hf", kwargs)),
        wandb_authenticator=lambda **kwargs: events.append(("wandb", kwargs))
        or True,
        device_selector=lambda **_: torch.device("cuda"),
    )

    assert events == [
        "drive-mounted",
        (
            "hf",
            {
                "token": "hf-secret",
                "add_to_git_credential": False,
                "skip_if_logged_in": False,
            },
        ),
        (
            "wandb",
            {
                "api_key": "wandb-secret",
                "enabled": True,
            },
        ),
    ]
    assert context.device.type == "cuda"
    assert context.artifact_root == tmp_path
    assert context.wandb_enabled is True


def test_wandb_is_enabled_by_default_and_failure_is_fatal(tmp_path: Path) -> None:
    secrets = {
        "HF_READ_TOKEN": "hf-secret",
        "WANDB_API_KEY": "bad-wandb-secret",
    }

    with pytest.raises(RuntimeError, match="W&B authentication failed") as error:
        initialize_colab(
            artifact_root=tmp_path,
            secret_getter=secrets.__getitem__,
            drive_mounter=lambda: None,
            hf_authenticator=lambda **_: None,
            wandb_authenticator=lambda **_: (_ for _ in ()).throw(
                RuntimeError("W&B authentication failed")
            ),
            device_selector=lambda **_: torch.device("cuda"),
        )

    assert "bad-wandb-secret" not in str(error.value)
    assert error.value.__cause__ is None


def test_wandb_can_be_explicitly_disabled(tmp_path: Path) -> None:
    requested: list[str] = []
    secrets = {"HF_READ_TOKEN": "hf-secret"}

    def get_secret(name: str) -> str:
        requested.append(name)
        return secrets[name]

    context = initialize_colab(
        enable_wandb=False,
        artifact_root=tmp_path,
        secret_getter=get_secret,
        drive_mounter=lambda: None,
        hf_authenticator=lambda **_: None,
        wandb_authenticator=lambda **_: (_ for _ in ()).throw(
            AssertionError("W&B authentication must be skipped")
        ),
        device_selector=lambda **_: torch.device("cuda"),
    )

    assert requested == ["HF_READ_TOKEN"]
    assert context.wandb_enabled is False


def test_required_secret_error_does_not_include_secret_value(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="HF_READ_TOKEN is unavailable") as error:
        initialize_colab(
            artifact_root=tmp_path,
            secret_getter=lambda _: (_ for _ in ()).throw(KeyError("private")),
            drive_mounter=lambda: None,
            device_selector=lambda **_: torch.device("cuda"),
        )

    assert "private" not in str(error.value)
    assert error.value.__cause__ is None


def test_drive_mount_failure_is_fatal_and_redacted(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Google Drive mount failed") as error:
        initialize_colab(
            artifact_root=tmp_path,
            secret_getter=lambda _: "unused",
            drive_mounter=lambda: (_ for _ in ()).throw(
                RuntimeError("sensitive mount detail")
            ),
            device_selector=lambda **_: torch.device("cuda"),
        )

    assert "sensitive mount detail" not in str(error.value)
    assert error.value.__cause__ is None


def test_huggingface_failure_is_fatal_and_redacted(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError, match="Hugging Face authentication failed"
    ) as error:
        initialize_colab(
            artifact_root=tmp_path,
            secret_getter=lambda _: "hf-sensitive-value",
            drive_mounter=lambda: None,
            hf_authenticator=lambda **_: (_ for _ in ()).throw(
                RuntimeError("hf-sensitive-value")
            ),
            device_selector=lambda **_: torch.device("cuda"),
        )

    assert "hf-sensitive-value" not in str(error.value)
    assert error.value.__cause__ is None
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
uv run pytest tests/test_colab_environment.py -v
```

Expected: FAIL because `jlens_reasoning.environments.colab` does not exist.

- [ ] **Step 3: Implement the Colab environment module**

Create `src/jlens_reasoning/environments/colab.py`:

```python
"""Reusable initialization for Google Colab notebooks."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from jlens_reasoning.config import ARTIFACT_ROOT_ENV, create_artifact_paths
from jlens_reasoning.environments.common import (
    RuntimeContext,
    create_runtime_context,
)
from jlens_reasoning.runtime import select_device
from jlens_reasoning.tracking import authenticate_wandb

DEFAULT_COLAB_ARTIFACT_ROOT = Path(
    "/content/drive/MyDrive/jlens-reasoning"
)


def _get_colab_secret(name: str) -> str:
    from google.colab import userdata

    return userdata.get(name)


def _mount_google_drive() -> None:
    from google.colab import drive

    drive.mount("/content/drive")


def _authenticate_huggingface(**kwargs: Any) -> None:
    from huggingface_hub import login

    login(**kwargs)


def _required_secret(
    name: str,
    secret_getter: Callable[[str], str],
) -> str:
    try:
        value = secret_getter(name)
    except Exception:
        raise RuntimeError(
            f"Required Colab secret {name} is unavailable"
        ) from None

    if not value:
        raise RuntimeError(f"Required Colab secret {name} is unavailable")

    return value


def initialize_colab(
    *,
    enable_wandb: bool = True,
    require_cuda: bool = False,
    artifact_root: str | Path = DEFAULT_COLAB_ARTIFACT_ROOT,
    secret_getter: Callable[[str], str] | None = None,
    drive_mounter: Callable[[], None] | None = None,
    hf_authenticator: Callable[..., None] | None = None,
    wandb_authenticator: Callable[..., bool] = authenticate_wandb,
    device_selector: Callable[..., torch.device] = select_device,
) -> RuntimeContext:
    """Mount Drive, authenticate services, and return notebook runtime paths."""

    secret_getter = secret_getter or _get_colab_secret
    drive_mounter = drive_mounter or _mount_google_drive
    hf_authenticator = hf_authenticator or _authenticate_huggingface

    try:
        drive_mounter()
    except Exception:
        raise RuntimeError("Google Drive mount failed") from None

    os.environ[ARTIFACT_ROOT_ENV] = str(artifact_root)
    paths = create_artifact_paths(artifact_root)
    os.environ["HF_HOME"] = str(paths.huggingface_cache)

    hf_token = _required_secret("HF_READ_TOKEN", secret_getter)
    try:
        hf_authenticator(
            token=hf_token,
            add_to_git_credential=False,
            skip_if_logged_in=False,
        )
    except Exception:
        raise RuntimeError("Hugging Face authentication failed") from None

    wandb_enabled = False
    if enable_wandb:
        wandb_key = _required_secret("WANDB_API_KEY", secret_getter)
        try:
            wandb_enabled = wandb_authenticator(
                api_key=wandb_key,
                enabled=True,
            )
        except Exception:
            raise RuntimeError("W&B authentication failed") from None

    device = device_selector(require_cuda=require_cuda)
    return create_runtime_context(
        paths=paths,
        device=device,
        wandb_enabled=wandb_enabled,
    )
```

The default keeps W&B enabled. Only `initialize_colab(enable_wandb=False)`
skips W&B authentication. `require_cuda=True` is reserved for experiment
notebooks that cannot produce meaningful results without a GPU.

- [ ] **Step 4: Run the Colab environment tests**

Run:

```bash
uv run pytest tests/test_colab_environment.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Colab initialization**

```bash
git add src/jlens_reasoning/environments/colab.py tests/test_colab_environment.py
git commit -m "feat: add reusable Colab initialization"
```

## Task 6: Add the authenticated, ref-pinned Colab bootstrap

**Files:**

- Create: `scripts/colab_bootstrap.py`
- Create: `tests/test_colab_bootstrap.py`

- [ ] **Step 1: Write bootstrap command and redaction tests**

Create `tests/test_colab_bootstrap.py`:

```python
import subprocess
from pathlib import Path
from typing import Any

from scripts.colab_bootstrap import (
    clone_repository,
    install_locked_environment,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        self.calls.append((command, kwargs))
        stdout = (
            "https://github.com/noamdwc/jlens-reasoning.git\n"
            if command[-2:] == ["get-url", "origin"]
            else ""
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_clone_uses_header_auth_without_putting_token_in_commands(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner()
    token = "github-secret-token"
    project_dir = tmp_path / "checkout"

    clone_repository(
        project_ref="experiment-branch",
        github_token=token,
        project_dir=project_dir,
        runner=runner,
    )

    rendered_commands = "\n".join(" ".join(call[0]) for call in runner.calls)
    assert token not in rendered_commands
    assert "experiment-branch" in rendered_commands

    fetch_call = next(call for call in runner.calls if "fetch" in call[0])
    fetch_environment = fetch_call[1]["env"]
    assert fetch_environment["GIT_CONFIG_COUNT"] == "1"
    assert fetch_environment["GIT_CONFIG_KEY_0"].endswith(".extraheader")
    assert fetch_environment["GIT_CONFIG_VALUE_0"].startswith(
        "AUTHORIZATION: basic "
    )


def test_locked_install_exports_experiment_dependencies(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner()

    install_locked_environment(
        project_dir=tmp_path,
        runner=runner,
        uv_bin="uv",
    )

    commands = [call[0] for call in runner.calls]
    assert commands[0][-2:] == ["install", "uv==0.11.28"]
    assert commands[1] == [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--extra",
        "experiment",
        "--prune",
        "torch",
        "--no-emit-project",
        "--format",
        "requirements.txt",
        "--output-file",
        "/tmp/jlens-requirements.txt",
        "--project",
        str(tmp_path),
    ]
    assert commands[2] == [
        "uv",
        "pip",
        "install",
        "--system",
        "--requirement",
        "/tmp/jlens-requirements.txt",
    ]
    assert commands[3] == [
        "uv",
        "pip",
        "install",
        "--system",
        "--no-deps",
        "--editable",
        str(tmp_path),
    ]
```

- [ ] **Step 2: Run the bootstrap tests and confirm they fail**

Run:

```bash
uv run pytest tests/test_colab_bootstrap.py -v
```

Expected: FAIL because `scripts.colab_bootstrap` does not exist.

- [ ] **Step 3: Implement the dependency-free bootstrap script**

Create `scripts/colab_bootstrap.py`:

```python
"""Clone an explicit project ref and install its locked Colab environment."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPOSITORY_URL = "https://github.com/noamdwc/jlens-reasoning.git"
DEFAULT_PROJECT_DIR = Path("/content/jlens-reasoning")
UV_VERSION = "0.11.28"
Runner = Callable[..., subprocess.CompletedProcess[str]]


def _git_auth_environment(token: str) -> dict[str, str]:
    encoded = base64.b64encode(
        f"x-access-token:{token}".encode()
    ).decode()
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {encoded}",
        }
    )
    return environment


def _run(
    runner: Runner,
    command: list[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            command,
            check=True,
            text=True,
            capture_output=True,
            **kwargs,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Colab bootstrap command failed") from exc


def clone_repository(
    *,
    project_ref: str,
    github_token: str,
    project_dir: Path = DEFAULT_PROJECT_DIR,
    runner: Runner = subprocess.run,
) -> Path:
    """Clone exactly one branch, tag, or commit without exposing the token."""

    if not project_ref:
        raise RuntimeError("A non-empty project ref is required")
    if not github_token:
        raise RuntimeError("GITHUB_TOKEN_JLENS_REAS is unavailable")

    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)

    _run(runner, ["git", "init", str(project_dir)])
    _run(
        runner,
        ["git", "-C", str(project_dir), "remote", "add", "origin", REPOSITORY_URL],
    )
    _run(
        runner,
        [
            "git",
            "-C",
            str(project_dir),
            "fetch",
            "--depth",
            "1",
            "origin",
            project_ref,
        ],
        env=_git_auth_environment(github_token),
    )
    _run(
        runner,
        ["git", "-C", str(project_dir), "checkout", "--detach", "FETCH_HEAD"],
    )

    remote = _run(
        runner,
        ["git", "-C", str(project_dir), "remote", "get-url", "origin"],
    ).stdout.strip()
    if github_token in remote or remote != REPOSITORY_URL:
        raise RuntimeError("Repository remote URL failed security validation")

    return project_dir


def install_locked_environment(
    *,
    project_dir: Path,
    runner: Runner = subprocess.run,
    uv_bin: str | None = None,
) -> None:
    """Install the lockfile into Colab's active Python environment."""

    _run(
        runner,
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"uv=={UV_VERSION}",
        ],
    )
    uv_bin = uv_bin or shutil.which("uv") or "uv"
    requirements = Path("/tmp/jlens-requirements.txt")

    _run(
        runner,
        [
            uv_bin,
            "export",
            "--frozen",
            "--no-dev",
            "--extra",
            "experiment",
            "--prune",
            "torch",
            "--no-emit-project",
            "--format",
            "requirements.txt",
            "--output-file",
            str(requirements),
            "--project",
            str(project_dir),
        ],
    )
    _run(
        runner,
        [
            uv_bin,
            "pip",
            "install",
            "--system",
            "--requirement",
            str(requirements),
        ],
    )
    _run(
        runner,
        [
            uv_bin,
            "pip",
            "install",
            "--system",
            "--no-deps",
            "--editable",
            str(project_dir),
        ],
    )


def bootstrap(
    *,
    project_ref: str,
    github_token: str,
    project_dir: Path = DEFAULT_PROJECT_DIR,
) -> Path:
    """Clone the requested source revision and install its locked environment."""

    checkout = clone_repository(
        project_ref=project_ref,
        github_token=github_token,
        project_dir=project_dir,
    )
    install_locked_environment(project_dir=checkout)
    return checkout
```

`--prune torch` is intentional. Linux CI resolves PyTorch from the locked CPU
index, while Colab retains the CUDA-enabled PyTorch supplied by its managed
runtime. The bootstrap installs every other project and experiment dependency
from `uv.lock`, then the editable package without dependency re-resolution.
`initialize_colab(require_cuda=True)` verifies that the retained runtime is
actually CUDA-capable before an experiment begins.

Do not print subprocess commands, environments, exception stderr, or tokens.
The Git remote remains the plain repository URL. Authentication exists only in
the `git fetch` subprocess environment.

- [ ] **Step 4: Run the bootstrap tests**

Run:

```bash
uv run pytest tests/test_colab_bootstrap.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the bootstrap**

```bash
git add scripts/colab_bootstrap.py tests/test_colab_bootstrap.py
git commit -m "feat: add secure Colab bootstrap"
```

## Task 7: Add reusable notebooks and notebook policy tests

**Files:**

- Create: `notebooks/_template.ipynb`
- Create: `notebooks/00_environment_check.ipynb`
- Create: `tests/test_notebooks.py`

- [ ] **Step 1: Write notebook policy tests before creating notebooks**

Create `tests/test_notebooks.py`:

```python
from pathlib import Path

import nbformat

NOTEBOOKS = [
    Path("notebooks/_template.ipynb"),
    Path("notebooks/00_environment_check.ipynb"),
]


def load_notebook(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(path, as_version=4)


def test_notebooks_have_no_saved_outputs_or_execution_counts() -> None:
    for path in NOTEBOOKS:
        notebook = load_notebook(path)
        for cell in notebook.cells:
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []


def test_notebooks_share_one_canonical_loader_cell() -> None:
    loader_cells = [load_notebook(path).cells[0].source for path in NOTEBOOKS]

    assert loader_cells[0] == loader_cells[1]
    assert "GITHUB_TOKEN_JLENS_REAS" in loader_cells[0]
    assert "scripts/colab_bootstrap.py" in loader_cells[0]
    assert "PROJECT_REF" in loader_cells[0]


def test_notebooks_do_not_contain_credentials() -> None:
    forbidden_fragments = ("github_pat_", "ghp_", "hf_", "wandb-secret")

    for path in NOTEBOOKS:
        source = path.read_text(encoding="utf-8")
        assert not any(fragment in source for fragment in forbidden_fragments)


def test_notebooks_use_the_colab_environment_module() -> None:
    for path in NOTEBOOKS:
        notebook = load_notebook(path)
        source = "\n".join(cell.source for cell in notebook.cells)

        assert (
            "from jlens_reasoning.environments.colab import initialize_colab"
            in source
        )
        assert "context = initialize_colab(" in source
```

- [ ] **Step 2: Run the policy tests and confirm they fail**

Run:

```bash
uv run pytest tests/test_notebooks.py -v
```

Expected: FAIL because both notebooks are absent.

- [ ] **Step 3: Create the canonical loader cell in both notebooks**

Set the first code cell of both notebooks to this exact source:

```python
PROJECT_REF = "main"

def _install_project(project_ref: str):
    import urllib.parse
    import urllib.request

    from google.colab import userdata

    github_token = userdata.get("GITHUB_TOKEN_JLENS_REAS")
    if not github_token:
        raise RuntimeError(
            "Required Colab secret GITHUB_TOKEN_JLENS_REAS is unavailable"
        )

    query = urllib.parse.urlencode({"ref": project_ref})
    bootstrap_url = (
        "https://api.github.com/repos/noamdwc/jlens-reasoning/"
        "contents/scripts/colab_bootstrap.py?"
        + query
    )
    request = urllib.request.Request(
        bootstrap_url,
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            bootstrap_source = response.read().decode("utf-8")
    except Exception:
        raise RuntimeError("Unable to load the Colab bootstrap") from None

    namespace = {}
    exec(
        compile(
            bootstrap_source,
            "scripts/colab_bootstrap.py",
            "exec",
        ),
        namespace,
    )
    return namespace["bootstrap"](
        project_ref=project_ref,
        github_token=github_token,
    )

PROJECT_DIR = _install_project(PROJECT_REF)
del _install_project
```

This is the intentional minimum notebook duplication. The cell fetches the
bootstrap at the same explicit ref it later checks out, so a branch, tag, or full
commit SHA is reproducible.

- [ ] **Step 4: Complete the template notebook**

Use standard Python 3 notebook metadata and add this second code cell:

```python
from jlens_reasoning.environments.colab import initialize_colab

context = initialize_colab(require_cuda=True)
context
```

Keep all execution counts `null` and outputs empty. New experiment notebooks
must start by copying this template, then changing only `PROJECT_REF` when a
different source revision is required.

- [ ] **Step 5: Complete the environment-check notebook**

Use the same loader cell and add this second code cell:

```python
from jlens_reasoning.environments.colab import initialize_colab

context = initialize_colab(require_cuda=True)
context
```

Add a third code cell:

```python
import subprocess

commit = subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

probe = context.runs_dir / "environment-check.txt"
probe.write_text("colab environment check\n", encoding="utf-8")

print(f"Commit: {commit}")
print(f"Device: {context.device}")
print(f"Artifact root: {context.artifact_root}")
print(f"Drive write probe: {probe}")
print(f"W&B enabled: {context.wandb_enabled}")
```

Do not load FLenQA or a large model. This notebook verifies the Git ref, Drive
writability, HF login, W&B login, CUDA selection, and a small persisted result.

- [ ] **Step 6: Run notebook policy tests**

Run:

```bash
uv run pytest tests/test_notebooks.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit notebooks and policy tests**

```bash
git add notebooks/_template.ipynb notebooks/00_environment_check.ipynb tests/test_notebooks.py
git commit -m "feat: add reusable Colab notebooks"
```

## Task 8: Add CI and complete developer documentation

**Files:**

- Create: `.github/workflows/ci.yml`
- Modify: `README.md`

- [ ] **Step 1: Create the Ubuntu and macOS CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

env:
  WANDB_MODE: disabled
  HF_HUB_OFFLINE: "1"
  TRANSFORMERS_OFFLINE: "1"

jobs:
  test:
    name: Python ${{ matrix.python-version }} on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            python-version: "3.11"
          - os: ubuntu-latest
            python-version: "3.12"
          - os: macos-latest
            python-version: "3.11"

    env:
      JLENS_REAS_ARTIFACT_ROOT: ${{ runner.temp }}/jlens-reasoning-test-artifacts

    steps:
      - uses: actions/checkout@v5

      - name: Install uv and Python
        uses: astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b
        with:
          version: "0.11.28"
          python-version: ${{ matrix.python-version }}
          enable-cache: true

      - name: Verify lockfile
        run: uv lock --check

      - name: Install locked environment
        run: uv sync --locked --extra experiment

      - name: Check formatting
        run: uv run ruff format --check .

      - name: Lint
        run: uv run ruff check .

      - name: Run tests
        run: uv run pytest -v
```

The offline variables constrain test runtime behavior, not dependency
installation. Tests must remain mocked and must not download models, datasets,
lenses, or checkpoints.

- [ ] **Step 2: Replace the README with complete operating instructions**

Replace `README.md` with:

````markdown
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
and installs all other project and experiment dependencies from the committed
lockfile.

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

## CI policy

CI installs the committed `uv.lock`, disables W&B, sets Hugging Face and
Transformers offline modes for test runtime, and uses a temporary artifact root.
CI tests imports and mocked setup behavior only; it never uses repository,
Hugging Face, W&B, or Google Drive credentials.
````

- [ ] **Step 3: Run formatting, lint, and the complete test suite**

Run:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest -v
```

Expected: all checks PASS. If Ruff changes files, inspect the diff and repeat
the three commands.

- [ ] **Step 4: Validate the CI workflow locally as static configuration**

Run:

```bash
rg -n "WANDB_MODE|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE|JLENS_REAS_ARTIFACT_ROOT" .github/workflows/ci.yml
rg -n "GITHUB_TOKEN_JLENS_REAS|HF_READ_TOKEN|WANDB_API_KEY" README.md notebooks
```

Expected: CI contains only non-secret runtime settings. Exact credential names
appear only in documentation and Colab code, with no credential values.

- [ ] **Step 5: Commit CI and documentation**

```bash
git add .github/workflows/ci.yml README.md
git commit -m "ci: test locked environment on Linux and macOS"
```

## Task 9: Perform release-level verification

**Files:**

- Verify all files created in Tasks 1–8
- Modify only files required by failures discovered during verification

- [ ] **Step 1: Verify the lockfile is current**

Run:

```bash
uv lock --check
uv sync --locked --extra experiment
```

Expected: both commands exit zero and do not modify `uv.lock`.

- [ ] **Step 2: Verify source quality and tests from the locked environment**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -v
```

Expected: all commands exit zero.

- [ ] **Step 3: Verify imports without model downloads**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -c "import jlens; import jlens_reasoning; print(jlens_reasoning.__version__)"
```

Expected output:

```text
0.1.0
```

No model, dataset, lens, or checkpoint is downloaded.

- [ ] **Step 4: Inspect repository hygiene**

Run:

```bash
git status --short
git diff --check
rg -n "github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}" --glob '!uv.lock' .
```

Expected: no uncommitted generated artifacts, no whitespace errors, and no
credential values. References to secret variable names are acceptable.

- [ ] **Step 5: Run the manual Colab acceptance check**

Through the IDE's Colab integration:

1. Open `notebooks/00_environment_check.ipynb`.
2. Set `PROJECT_REF` to the commit being accepted.
3. Confirm a GPU runtime is selected.
4. Run all cells.
5. Confirm the printed commit matches `PROJECT_REF`.
6. Confirm the device is CUDA.
7. Confirm the Drive probe exists under
   `/content/drive/MyDrive/jlens-reasoning/runs/`.
8. Confirm Hugging Face and default-enabled W&B authentication succeed.
9. Re-run initialization once with `enable_wandb=False` and confirm W&B is
   skipped.
10. Confirm notebook outputs are cleared before committing the notebook.

This is the only manual environment gate. It intentionally does not exercise
FLenQA or a large model.

- [ ] **Step 6: Commit any verification-only corrections**

If verification required a correction:

```bash
git add -u
git commit -m "fix: correct environment verification issues"
```

If no correction was required, do not create an empty commit.
