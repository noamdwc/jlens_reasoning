"""Bootstrap Google Drive paths on Colab without interactive OAuth.

This module must stay free of ``jlens_reasoning`` imports so
``scripts/run_colab_notebook.sh`` can upload it to the VM and execute it with
``runpy`` before the project wheel is installed.

Unattended CLI runs upload a service-account JSON to
``/content/jlens-credentials/drive-sa.json``. Interactive browser sessions keep
using ``google.colab.drive.mount``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

SA_JSON_VM_PATH = Path("/content/jlens-credentials/drive-sa.json")
CREDENTIALS_DIR = SA_JSON_VM_PATH.parent
WANDB_ENV_VM_PATH = CREDENTIALS_DIR / "wandb.env"
JLENS_ENV_VM_PATH = CREDENTIALS_DIR / "jlens.env"
DRIVE_HELPER_VM_PATH = CREDENTIALS_DIR / "colab_drive.py"

MYDRIVE_ROOT = Path("/content/drive/MyDrive")
ARTIFACTS_PATH = MYDRIVE_ROOT / "jlens-reasoning"
DATA_PATH = MYDRIVE_ROOT / "data" / "jlens-reasoning"

DEFAULT_ROOT_FOLDER_NAME = "jlens-colab-root"
ROOT_FOLDER_ID_ENV = "JLENS_DRIVE_ROOT_FOLDER_ID"
ROOT_FOLDER_NAME_ENV = "JLENS_DRIVE_ROOT_FOLDER_NAME"

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def drive_layout_ready(mydrive_root: Path = MYDRIVE_ROOT) -> bool:
    """Return True when the notebook-expected Drive paths already exist."""

    return (mydrive_root / "data" / "jlens-reasoning").is_dir() and (
        mydrive_root / "jlens-reasoning"
    ).is_dir()


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file into a dict."""

    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def apply_env_files(
    *,
    paths: Sequence[Path] = (JLENS_ENV_VM_PATH, WANDB_ENV_VM_PATH),
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    """Load uploaded env files into ``os.environ`` without overwriting."""

    target = os.environ if environ is None else environ
    loaded: dict[str, str] = {}
    for path in paths:
        for key, value in load_env_file(path).items():
            if key not in target or not target[key]:
                target[key] = value
                loaded[key] = value
    return loaded


def resolve_wandb_api_key(
    *,
    environ: Mapping[str, str] | None = None,
    secret_getter: Callable[[str], str] | None = None,
) -> str | None:
    """Resolve WANDB_API_KEY from env, uploaded files, then Colab Secrets."""

    env = dict(os.environ if environ is None else environ)
    apply_env_files(environ=env)
    value = env.get("WANDB_API_KEY", "").strip()
    if value:
        return value

    if secret_getter is not None:
        try:
            secret = secret_getter("WANDB_API_KEY")
        except Exception:
            return None
        if secret:
            return secret
    return None


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
    )


def _require_success(
    result: subprocess.CompletedProcess[str],
    *,
    action: str,
) -> None:
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "").strip()
    message = f"{action} failed"
    if detail:
        message = f"{message}: {detail.splitlines()[-1]}"
    raise RuntimeError(message)


def _ensure_rclone(runner: CommandRunner) -> str:
    existing = shutil.which("rclone")
    if existing:
        return existing

    install = runner(
        [
            "bash",
            "-lc",
            "curl -fsSL https://rclone.org/install.sh | bash",
        ]
    )
    _require_success(install, action="rclone install")
    installed = shutil.which("rclone")
    if not installed:
        raise RuntimeError("rclone install completed but rclone was not found on PATH")
    return installed


def _drive_remote(
    sa_json: Path,
    *,
    root_folder_id: str | None = None,
    shared_with_me: bool = False,
) -> str:
    parts = [
        "drive",
        f"service_account_file={sa_json}",
        "scope=drive",
    ]
    if root_folder_id:
        parts.append(f"root_folder_id={root_folder_id}")
    if shared_with_me:
        parts.append("shared_with_me=true")
    return ":" + ",".join(parts)


def _rclone_lsjson(
    rclone: str,
    sa_json: Path,
    *,
    shared_with_me: bool,
    runner: CommandRunner,
) -> list[dict[str, object]]:
    remote = _drive_remote(sa_json, shared_with_me=shared_with_me)
    result = runner([rclone, "lsjson", f"{remote}:"])
    _require_success(result, action="rclone lsjson")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("rclone lsjson returned a non-list payload")
    return payload


def resolve_root_folder_id(
    sa_json: Path,
    *,
    rclone: str,
    runner: CommandRunner,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the shared Drive folder that should appear as MyDrive."""

    env = dict(os.environ if environ is None else environ)
    apply_env_files(environ=env)

    configured = env.get(ROOT_FOLDER_ID_ENV, "").strip()
    if configured:
        return configured

    folder_name = env.get(ROOT_FOLDER_NAME_ENV, "").strip() or DEFAULT_ROOT_FOLDER_NAME
    candidates_by_id: dict[str, dict[str, object]] = {}
    for shared_with_me in (True, False):
        try:
            entries = _rclone_lsjson(
                rclone,
                sa_json,
                shared_with_me=shared_with_me,
                runner=runner,
            )
        except RuntimeError:
            continue
        for entry in entries:
            if entry.get("Name") != folder_name:
                continue
            mime = str(entry.get("MimeType", ""))
            is_dir = bool(entry.get("IsDir")) or mime.endswith("folder")
            folder_id = entry.get("ID")
            if is_dir and folder_id:
                candidates_by_id[str(folder_id)] = entry

    candidates = list(candidates_by_id.values())
    if not candidates:
        raise RuntimeError(
            "Could not find shared Drive folder "
            f"{folder_name!r} for the service account; share that folder "
            f"(containing jlens-reasoning/ and data/jlens-reasoning/) with the "
            f"SA and/or set {ROOT_FOLDER_ID_ENV}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple Drive folders named {folder_name!r} are visible to the "
            f"service account; set {ROOT_FOLDER_ID_ENV} to disambiguate"
        )
    folder_id = str(candidates[0]["ID"])
    return folder_id


def _wait_for_layout(
    mydrive_root: Path,
    *,
    timeout_seconds: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if drive_layout_ready(mydrive_root):
            return
        sleep(0.5)
    raise RuntimeError(
        "Service-account Drive mount did not expose "
        f"{mydrive_root / 'jlens-reasoning'} and "
        f"{mydrive_root / 'data' / 'jlens-reasoning'} in time"
    )


def mount_drive_with_service_account(
    sa_json: Path,
    *,
    mydrive_root: Path = MYDRIVE_ROOT,
    runner: CommandRunner = _default_runner,
    environ: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Mount the shared SA Drive root at ``/content/drive/MyDrive`` via rclone."""

    if not sa_json.is_file():
        raise RuntimeError(f"Service-account JSON not found: {sa_json}")

    env = dict(os.environ if environ is None else environ)
    apply_env_files(environ=env)
    rclone = _ensure_rclone(runner)
    root_folder_id = resolve_root_folder_id(
        sa_json,
        rclone=rclone,
        runner=runner,
        environ=env,
    )

    mydrive_root.mkdir(parents=True, exist_ok=True)
    if any(mydrive_root.iterdir()):
        # Already populated (prior mount or copy); require expected layout.
        if drive_layout_ready(mydrive_root):
            return
        raise RuntimeError(
            f"{mydrive_root} is not empty and does not contain the expected "
            "jlens-reasoning layout"
        )

    remote = _drive_remote(sa_json, root_folder_id=root_folder_id)
    result = runner(
        [
            rclone,
            "mount",
            f"{remote}:",
            str(mydrive_root),
            "--daemon",
            "--allow-other",
            "--vfs-cache-mode",
            "full",
            "--dir-cache-time",
            "30s",
        ]
    )
    _require_success(result, action="rclone mount")
    _wait_for_layout(mydrive_root, sleep=sleep)


def _mount_google_drive_interactive() -> None:
    from google.colab import drive

    drive.mount("/content/drive")


def ensure_colab_drive(
    *,
    sa_json: Path = SA_JSON_VM_PATH,
    mydrive_root: Path = MYDRIVE_ROOT,
    interactive_mounter: Callable[[], None] | None = None,
    sa_mounter: Callable[..., None] | None = None,
    runner: CommandRunner = _default_runner,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Ensure notebook Drive paths exist; return ``service_account`` or ``interactive``."""

    env = dict(os.environ if environ is None else environ)
    apply_env_files(environ=env)
    if drive_layout_ready(mydrive_root):
        return "service_account" if sa_json.is_file() else "interactive"

    if sa_json.is_file():
        mounter = sa_mounter or mount_drive_with_service_account
        try:
            mounter(
                sa_json,
                mydrive_root=mydrive_root,
                runner=runner,
                environ=env,
            )
        except Exception:
            raise RuntimeError("Google Drive service-account mount failed") from None
        if not drive_layout_ready(mydrive_root):
            raise RuntimeError("Google Drive service-account mount failed")
        return "service_account"

    mounter = interactive_mounter or _mount_google_drive_interactive
    try:
        mounter()
    except Exception:
        raise RuntimeError("Google Drive mount failed") from None
    return "interactive"


def main() -> None:
    mode = ensure_colab_drive()
    print(f"Colab Drive ready via {mode}")


if __name__ == "__main__":
    main()
