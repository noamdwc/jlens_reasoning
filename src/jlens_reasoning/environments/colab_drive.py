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
import tempfile
import time
import uuid
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

ROOT_FOLDER_ID_ENV = "JLENS_DRIVE_ROOT_FOLDER_ID"
SHARED_DRIVE_ID_ENV = "JLENS_DRIVE_SHARED_DRIVE_ID"
RCLONE_RC_ADDR = "127.0.0.1:5572"
MOUNT_MARKER_VM_PATH = CREDENTIALS_DIR / "rclone-mounted"

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


def _drive_remote(sa_json: Path, *, root_folder_id: str, shared_drive_id: str) -> str:
    return (
        f":drive,service_account_file={sa_json},scope=drive,"
        f"root_folder_id={root_folder_id},team_drive={shared_drive_id}"
    )


def _required_drive_id(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in value
    ):
        raise RuntimeError(
            f"Set {name} to a valid Drive ID; service accounts require a Workspace Shared Drive"
        )
    return value


def _verify_remote_write(rclone: str, remote: str, runner: CommandRunner) -> None:
    """Verify persistence through the API, bypassing the mount's local cache."""
    probe_name = f".jlens-write-probe-{uuid.uuid4().hex}"
    destination = f"{remote}:{probe_name}"
    with tempfile.TemporaryDirectory() as directory:
        probe = Path(directory) / probe_name
        probe.write_text(probe_name, encoding="utf-8")
        _require_success(
            runner([rclone, "copyto", str(probe), destination]),
            action="Drive write probe (use a writable Workspace Shared Drive)",
        )
        try:
            result = runner([rclone, "cat", destination])
            _require_success(result, action="Drive write probe readback")
            if result.stdout != probe_name:
                raise RuntimeError("Drive write probe readback did not match")
        finally:
            _require_success(
                runner([rclone, "deletefile", destination, "--drive-use-trash=true"]),
                action="Drive write probe cleanup",
            )


def wait_for_drive_uploads(
    *,
    runner: CommandRunner = _default_runner,
    timeout_seconds: float = 600.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait for the daemon's queued and active uploads, failing closed."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        result = runner(
            ["rclone", "rc", "--url", f"http://{RCLONE_RC_ADDR}", "vfs/stats"]
        )
        _require_success(result, action="Drive upload status")
        try:
            stats = json.loads(result.stdout)
            cache = stats["diskCache"]
            counts = [
                cache[name]
                for name in ("uploadsQueued", "uploadsInProgress", "erroredFiles")
            ]
            out_of_space = cache["outOfSpace"]
            if (
                any(type(value) is not int or value < 0 for value in counts)
                or type(out_of_space) is not bool
            ):
                raise ValueError("invalid counters")
        except (ValueError, KeyError, TypeError):
            raise RuntimeError(
                "Invalid Drive upload status; cannot confirm persistence"
            ) from None
        if counts[2] or out_of_space:
            raise RuntimeError(
                "Drive uploads failed; preserve the VM to recover cached files"
            )
        if not any(counts):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for Drive uploads; preserve the VM")
        sleep(1.0)


def flush_colab_drive(*, marker: Path = MOUNT_MARKER_VM_PATH) -> None:
    # Bootstrap failures before mounting have no daemon or buffered artifacts.
    if marker.is_file():
        wait_for_drive_uploads()


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
    marker: Path = MOUNT_MARKER_VM_PATH,
) -> None:
    """Mount the shared SA Drive root at ``/content/drive/MyDrive`` via rclone."""

    if not sa_json.is_file():
        raise RuntimeError(f"Service-account JSON not found: {sa_json}")

    env = dict(os.environ if environ is None else environ)
    apply_env_files(environ=env)
    shared_drive_id = _required_drive_id(env, SHARED_DRIVE_ID_ENV)
    root_folder_id = _required_drive_id(env, ROOT_FOLDER_ID_ENV)
    rclone = _ensure_rclone(runner)

    mydrive_root.mkdir(parents=True, exist_ok=True)
    if any(mydrive_root.iterdir()):
        # Already populated (prior mount or copy); require expected layout.
        if drive_layout_ready(mydrive_root):
            return
        raise RuntimeError(
            f"{mydrive_root} is not empty and does not contain the expected "
            "jlens-reasoning layout"
        )

    remote = _drive_remote(
        sa_json, root_folder_id=root_folder_id, shared_drive_id=shared_drive_id
    )
    _verify_remote_write(rclone, remote, runner)
    result = runner(
        [
            rclone,
            "mount",
            f"{remote}:",
            str(mydrive_root),
            "--daemon",
            "--rc",
            "--rc-addr",
            RCLONE_RC_ADDR,
            "--allow-other",
            "--vfs-cache-mode",
            "full",
            "--dir-cache-time",
            "30s",
        ]
    )
    _require_success(result, action="rclone mount")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
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
