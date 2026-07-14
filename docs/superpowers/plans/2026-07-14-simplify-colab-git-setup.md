# Simplify Colab Git Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the GitHub Contents API bootstrap with an authenticated private `git clone`, explicit `git checkout`, and an installer-only bootstrap script.

**Architecture:** The canonical notebook loader reads `GITHUB_TOKEN_JLENS_REAS`, passes it only through a process-local Git authorization environment, and runs normal clone and checkout commands. After checkout, the notebook executes the repository-local bootstrap script, whose only responsibility is installing the locked Colab environment while preserving Colab's CUDA-enabled PyTorch.

**Tech Stack:** Python 3.11, Git, Google Colab Secrets, uv, nbformat, pytest, Ruff.

**Design reference:** `docs/superpowers/specs/2026-07-14-environment-setup-design.md`

---

### Task 1: Make the Colab bootstrap installer-only

**Files:**

- Modify: `scripts/colab_bootstrap.py`
- Modify: `tests/test_colab_bootstrap.py`

- [ ] **Step 1: Replace the bootstrap tests with installer-only expectations**

Replace `tests/test_colab_bootstrap.py` with:

```python
import subprocess
from pathlib import Path
from typing import Any

import pytest

import scripts.colab_bootstrap as colab_bootstrap
from scripts.colab_bootstrap import (
    PROJECT_ROOT,
    install_locked_environment,
    main,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess:
        self.calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_locked_install_exports_experiment_dependencies(tmp_path: Path) -> None:
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


def test_main_installs_the_cloned_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[Path] = []

    def record_install(*, project_dir: Path) -> None:
        installed.append(project_dir)

    monkeypatch.setattr(
        colab_bootstrap,
        "install_locked_environment",
        record_install,
    )

    main()

    assert installed == [PROJECT_ROOT]
```

This removes the clone/authentication test because Git ownership moves to the
notebook loader. It adds a CLI-entry test proving that executing the script
installs the repository containing the script.

- [ ] **Step 2: Run the bootstrap tests and verify the new contract fails**

Run:

```bash
uv run pytest tests/test_colab_bootstrap.py -v
```

Expected: collection FAIL because `PROJECT_ROOT` and `main` do not yet exist.

- [ ] **Step 3: Replace the bootstrap with an installer-only script**

Replace `scripts/colab_bootstrap.py` with:

```python
"""Install the locked environment into an active Colab interpreter."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UV_VERSION = "0.11.28"
Runner = Callable[..., subprocess.CompletedProcess[str]]


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


def install_locked_environment(
    *,
    project_dir: Path,
    runner: Runner = subprocess.run,
    uv_bin: str | None = None,
) -> None:
    """Install all locked dependencies except Colab's managed PyTorch."""

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


def main() -> None:
    """Install the repository containing this script."""

    install_locked_environment(project_dir=PROJECT_ROOT)


if __name__ == "__main__":
    main()
```

This deletes `REPOSITORY_URL`, `DEFAULT_PROJECT_DIR`,
`_git_auth_environment()`, `clone_repository()`, and `bootstrap()`.

- [ ] **Step 4: Run the bootstrap tests**

Run:

```bash
uv run pytest tests/test_colab_bootstrap.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit the installer-only bootstrap**

```bash
git add scripts/colab_bootstrap.py tests/test_colab_bootstrap.py
git commit -m "refactor: make Colab bootstrap installer-only"
```

### Task 2: Replace the notebook API loader with clone and checkout

**Files:**

- Modify: `tests/test_notebooks.py`
- Modify: `notebooks/_template.ipynb`
- Modify: `notebooks/00_environment_check.ipynb`

- [ ] **Step 1: Strengthen notebook policy tests for the simple Git flow**

Replace `test_notebooks_share_one_canonical_loader_cell` in
`tests/test_notebooks.py` with:

```python
def test_notebooks_share_one_canonical_loader_cell() -> None:
    loader_cells = [load_notebook(path).cells[0].source for path in NOTEBOOKS]

    assert loader_cells[0] == loader_cells[1]
    assert "GITHUB_TOKEN_JLENS_REAS" in loader_cells[0]
    assert '["git", "clone"' in loader_cells[0]
    assert '"checkout",' in loader_cells[0]
    assert "scripts/colab_bootstrap.py" in loader_cells[0]
    assert "PROJECT_REF" in loader_cells[0]
    assert "api.github.com" not in loader_cells[0]
    assert "urllib.request" not in loader_cells[0]
```

Add this test to the end of `tests/test_notebooks.py`:

```python
def test_notebook_code_cells_compile() -> None:
    for path in NOTEBOOKS:
        notebook = load_notebook(path)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")
```

The existing output, credential, and shared-initializer tests remain unchanged.

- [ ] **Step 2: Run notebook tests and verify the old API loader fails**

Run:

```bash
uv run pytest tests/test_notebooks.py -v
```

Expected: `test_notebooks_share_one_canonical_loader_cell` FAIL because the old
cell does not contain the `git clone` command and still contains
`api.github.com`.

- [ ] **Step 3: Replace the canonical loader cell in both notebooks**

Set the first cell of `notebooks/_template.ipynb` and
`notebooks/00_environment_check.ipynb` to this exact source:

```python
PROJECT_REF = "main"

import base64
import os
import subprocess
import sys
from pathlib import Path

from google.colab import userdata

REPOSITORY_URL = "https://github.com/noamdwc/jlens_reasoning.git"
PROJECT_DIR = Path("/content/jlens-reasoning")

github_token = userdata.get("GITHUB_TOKEN_JLENS_REAS")
if not github_token:
    raise RuntimeError(
        "Required Colab secret GITHUB_TOKEN_JLENS_REAS is unavailable"
    )

authorization = base64.b64encode(
    f"x-access-token:{github_token}".encode()
).decode()
git_environment = os.environ.copy()
git_environment.update(
    {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {authorization}",
    }
)

try:
    subprocess.run(
        ["git", "clone", REPOSITORY_URL, str(PROJECT_DIR)],
        check=True,
        env=git_environment,
    )
except Exception:
    raise RuntimeError("Unable to clone the private repository") from None
finally:
    del github_token
    del authorization
    del git_environment

try:
    subprocess.run(
        ["git", "-C", str(PROJECT_DIR), "checkout", PROJECT_REF],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "colab_bootstrap.py"),
        ],
        check=True,
    )
except Exception:
    raise RuntimeError(
        "Unable to check out or install the selected project ref"
    ) from None
```

The token exists only in the clone subprocess environment. The command list and
saved remote contain the plain repository URL. The cell assumes
`/content/jlens-reasoning` does not exist, which is the documented fresh-runtime
contract.

Do not change the second initializer cell or the environment-check probe cell.

- [ ] **Step 4: Run the notebook policy tests**

Run:

```bash
uv run pytest tests/test_notebooks.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit the simple notebook loader**

```bash
git add tests/test_notebooks.py notebooks/_template.ipynb notebooks/00_environment_check.ipynb
git commit -m "refactor: simplify Colab Git setup"
```

### Task 3: Update user documentation and verify the branch

**Files:**

- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-14-environment-setup.md`

- [ ] **Step 1: Clarify the clone flow in the README**

In `README.md`, replace:

```markdown
Open `notebooks/_template.ipynb` through the IDE's Colab integration. Set
`PROJECT_REF` to an explicit branch, tag, or full commit SHA, run the loader
cell, then initialize. The bootstrap preserves Colab's CUDA-enabled PyTorch
and installs all other project and experiment dependencies from the committed
lockfile.
```

with:

```markdown
Open `notebooks/_template.ipynb` through the IDE's Colab integration. Set
`PROJECT_REF` to an explicit branch, tag, or full commit SHA, then run the
loader cell. It uses `GITHUB_TOKEN_JLENS_REAS` only for a normal private
`git clone`, checks out the requested ref, and runs the repository's installer.
The installer preserves Colab's CUDA-enabled PyTorch and installs all other
project and experiment dependencies from the committed lockfile.
```

- [ ] **Step 2: Mark the original implementation-plan sections as superseded**

Immediately below the header in
`docs/superpowers/plans/2026-07-14-environment-setup.md`, add:

```markdown
> **Colab Git setup update:** Tasks 6 and 7 are superseded by
> `docs/superpowers/plans/2026-07-14-simplify-colab-git-setup.md`. The current
> design uses a private `git clone`, explicit `git checkout`, and an
> installer-only bootstrap instead of the GitHub Contents API.
```

This keeps the original execution record while directing readers to the active
design.

- [ ] **Step 3: Format and lint**

Run:

```bash
uv run ruff format .
uv run ruff check .
```

Expected: formatting completes and lint reports `All checks passed!`.

- [ ] **Step 4: Run the complete test suite**

Run:

```bash
uv run pytest -v
```

Expected: 28 tests PASS.

- [ ] **Step 5: Verify lockfile, imports, notebooks, and secret hygiene**

Run:

```bash
uv lock --check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -c "import jlens; import jlens_reasoning; print(jlens_reasoning.__version__)"
rg -n "api.github.com|urllib.request" README.md notebooks scripts
rg -n "github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}" --glob '!uv.lock' .
git diff --check
```

Expected:

- `uv lock --check` exits zero.
- The import command prints `0.1.0`.
- Both `rg` searches return no matches.
- `git diff --check` exits zero.

- [ ] **Step 6: Commit documentation and any formatting changes**

```bash
git add README.md docs/superpowers/plans/2026-07-14-environment-setup.md scripts tests notebooks
git commit -m "docs: document simple Colab clone flow"
```

- [ ] **Step 7: Push the updated branch**

Run:

```bash
git push origin codex/environment-setup
```

Expected: the remote branch advances to the final documentation commit. Do not
create or update a PR automatically; the user will run the prepared `gh pr
create` command after reviewing this revision.
