"""Bootstrap Google Drive paths on Colab without interactive OAuth.

This module must stay free of ``jlens_reasoning`` imports so
``scripts/run_colab_notebook.sh`` can upload it to the VM and execute it with
``runpy`` before the project wheel is installed.

Unattended CLI runs upload a minimal user rclone config or a service-account
JSON to ``/content/jlens-credentials/``. Interactive browser sessions keep
using ``google.colab.drive.mount``.
"""

from __future__ import annotations

import argparse
import configparser
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

SA_JSON_VM_PATH = Path("/content/jlens-credentials/drive-sa.json")
CREDENTIALS_DIR = SA_JSON_VM_PATH.parent
RCLONE_CONFIG_VM_PATH = CREDENTIALS_DIR / "rclone.conf"
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


def extract_rclone_remote(path: Path) -> dict[str, str] | None:
    """Read only user-OAuth settings for jlens; never expose parser/token errors."""
    try:
        config = configparser.ConfigParser(interpolation=None)
        with path.open(encoding="utf-8") as stream:
            config.read_file(stream)
        if not config.has_section("jlens"):
            return None
        remote = dict(config["jlens"])
        if config.defaults() or remote.get("type") != "drive":
            raise ValueError
        if any(
            remote.get(key)
            for key in ("service_account_file", "service_account_credentials")
        ):
            raise ValueError
        if remote.get("env_auth", "false").lower() not in ("false", "0", ""):
            raise ValueError
        token = json.loads(remote.get("token", ""))
        if (
            not isinstance(token, dict)
            or not isinstance(token.get("refresh_token"), str)
            or not token["refresh_token"].strip()
        ):
            raise ValueError
    except (OSError, UnicodeError, configparser.Error, ValueError):
        raise RuntimeError(
            "Invalid user rclone config: jlens must be a Drive OAuth remote with a refresh token; reauthorize locally"
        ) from None
    allowed = {
        "type",
        "token",
        "client_id",
        "client_secret",
        "scope",
        "root_folder_id",
        "team_drive",
        "resource_key",
        "encoding",
    }
    return {key: value for key, value in remote.items() if key in allowed}


def select_drive_auth(
    *,
    rclone_present: bool,
    sa_present: bool,
    mode: str = "auto",
    allow_interactive: bool = False,
) -> str:
    """Resolve auth consistently on the host and VM, without silent downgrades."""
    if mode not in ("auto", "rclone", "service_account", "interactive"):
        raise RuntimeError(
            "JLENS_DRIVE_AUTH must be auto, rclone, service_account, or interactive"
        )
    if mode == "interactive":
        return mode
    if mode in ("auto", "rclone") and rclone_present:
        return "rclone"
    if mode == "rclone":
        raise RuntimeError(
            "User rclone credentials are required by JLENS_DRIVE_AUTH=rclone"
        )
    if sa_present:
        return "service_account"
    if mode == "auto" and allow_interactive:
        return "interactive"
    raise RuntimeError(
        "unattended Colab CLI Drive access requires a user rclone jlens remote or "
        "JLENS_DRIVE_SA_JSON; explicitly allow interactive drivemount for browser authorization"
    )


def _write_private(path: Path, content: str) -> None:
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write(content)


def stage_colab_credentials(
    directory: Path, *, environ: Mapping[str, str] | None = None
) -> str:
    """Stage selected credentials for upload; stdout must contain only the mode."""
    env = dict(os.environ if environ is None else environ)
    home = Path(env.get("HOME", str(Path.home())))
    mode = env.get("JLENS_DRIVE_AUTH", "auto")
    explicit_config = env.get("JLENS_RCLONE_CONFIG") or env.get("RCLONE_CONFIG")
    config_path = (
        Path(explicit_config).expanduser()
        if explicit_config
        else (
            Path(env.get("XDG_CONFIG_HOME") or str(home / ".config"))
            / "rclone"
            / "rclone.conf"
        )
    )
    remote = None
    if mode in ("auto", "rclone"):
        if explicit_config or config_path.is_file():
            remote = extract_rclone_remote(config_path)
            if explicit_config and remote is None:
                raise RuntimeError("The explicit rclone config has no jlens remote")
    sa_json = Path(
        env.get(
            "JLENS_DRIVE_SA_JSON", str(home / ".config" / "jlens" / "drive-sa.json")
        )
    ).expanduser()
    selected = select_drive_auth(
        rclone_present=remote is not None,
        sa_present=sa_json.is_file(),
        mode=mode,
        allow_interactive=env.get("JLENS_COLAB_ALLOW_INTERACTIVE_DRIVEMOUNT") == "1",
    )
    values = {"JLENS_DRIVE_AUTH": selected}
    sa_content = ""
    if selected == "service_account":
        for name in (SHARED_DRIVE_ID_ENV, ROOT_FOLDER_ID_ENV):
            values[name] = _required_drive_id(env, name)
        try:
            sa_content = sa_json.read_text(encoding="utf-8")
            key = json.loads(sa_content)
            if not isinstance(key, dict) or key.get("type") != "service_account":
                raise ValueError
            for field in ("client_email", "private_key", "token_uri"):
                if not isinstance(key.get(field), str) or not key[field].strip():
                    raise ValueError
        except (OSError, UnicodeError, ValueError):
            raise RuntimeError(
                "Invalid service-account JSON: required key fields are missing or malformed"
            ) from None
    if selected == "rclone":
        for name, field in (
            (ROOT_FOLDER_ID_ENV, "root_folder_id"),
            (SHARED_DRIVE_ID_ENV, "team_drive"),
        ):
            if env.get(name):
                remote[field] = _required_drive_id(env, name)
        if env.get(SHARED_DRIVE_ID_ENV) and not env.get(ROOT_FOLDER_ID_ENV):
            raise RuntimeError(
                "Set JLENS_DRIVE_ROOT_FOLDER_ID with JLENS_DRIVE_SHARED_DRIVE_ID"
            )
    if env.get("WANDB_API_KEY"):
        if "\n" in env["WANDB_API_KEY"] or "\r" in env["WANDB_API_KEY"]:
            raise RuntimeError("WANDB_API_KEY must be a single line")
        values["WANDB_API_KEY"] = env["WANDB_API_KEY"]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    if selected == "rclone":
        config = configparser.ConfigParser(interpolation=None)
        config["jlens"] = remote
        buffer = io.StringIO()
        config.write(buffer)
        _write_private(directory / "rclone.conf", buffer.getvalue())
    elif selected == "service_account":
        _write_private(directory / "drive-sa.json", sa_content)
    if selected != "interactive":
        _write_private(
            directory / "jlens.env",
            "".join(f"{key}={value}\n" for key, value in values.items()),
        )
    return selected


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


def _verify_remote_write(
    rclone: str, remote: str, runner: CommandRunner, *, rclone_args: Sequence[str] = ()
) -> None:
    """Verify persistence through the API, bypassing the mount's local cache."""
    probe_name = f".jlens-write-probe-{uuid.uuid4().hex}"
    destination = f"{remote}:{probe_name}"
    with tempfile.TemporaryDirectory() as directory:
        probe = Path(directory) / probe_name
        probe.write_text(probe_name, encoding="utf-8")
        _require_success(
            runner([rclone, "copyto", str(probe), destination, *rclone_args]),
            action="Drive write probe (check the selected account can write to the project root)",
        )
        try:
            result = runner([rclone, "cat", destination, *rclone_args])
            _require_success(result, action="Drive write probe readback")
            if result.stdout != probe_name:
                raise RuntimeError("Drive write probe readback did not match")
        finally:
            _require_success(
                runner(
                    [
                        rclone,
                        "deletefile",
                        destination,
                        "--drive-use-trash=true",
                        *rclone_args,
                    ]
                ),
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
        "Drive mount did not expose "
        f"{mydrive_root / 'jlens-reasoning'} and "
        f"{mydrive_root / 'data' / 'jlens-reasoning'} in time"
    )


def _mount_rclone_remote(
    remote: str,
    *,
    mydrive_root: Path,
    runner: CommandRunner,
    sleep: Callable[[float], None],
    marker: Path,
    rclone_args: Sequence[str] = (),
) -> None:
    rclone = _ensure_rclone(runner)
    mydrive_root.mkdir(parents=True, exist_ok=True)
    if any(mydrive_root.iterdir()):
        if mydrive_root.is_mount() and drive_layout_ready(mydrive_root):
            return
        raise RuntimeError(
            f"{mydrive_root} is not empty; expected an existing Drive mount "
            "with the jlens-reasoning layout"
        )
    _verify_remote_write(rclone, remote, runner, rclone_args=rclone_args)
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
            "--vfs-cache-mode",
            "full",
            "--dir-cache-time",
            "30s",
            *rclone_args,
        ]
    )
    _require_success(result, action="rclone mount")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    _wait_for_layout(mydrive_root, sleep=sleep)


def mount_drive_with_service_account(
    sa_json: Path,
    *,
    mydrive_root: Path = MYDRIVE_ROOT,
    runner: CommandRunner = _default_runner,
    environ: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    marker: Path = MOUNT_MARKER_VM_PATH,
) -> None:
    """Mount an SA's writable Shared Drive project root using rclone."""
    if not sa_json.is_file():
        raise RuntimeError(f"Service-account JSON not found: {sa_json}")
    env = dict(os.environ if environ is None else environ)
    apply_env_files(environ=env)
    shared_drive_id = _required_drive_id(env, SHARED_DRIVE_ID_ENV)
    root_folder_id = _required_drive_id(env, ROOT_FOLDER_ID_ENV)
    remote = _drive_remote(
        sa_json, root_folder_id=root_folder_id, shared_drive_id=shared_drive_id
    )
    _mount_rclone_remote(
        remote, mydrive_root=mydrive_root, runner=runner, sleep=sleep, marker=marker
    )


def mount_drive_with_rclone_config(
    config: Path,
    *,
    mydrive_root: Path = MYDRIVE_ROOT,
    runner: CommandRunner = _default_runner,
    environ: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    marker: Path = MOUNT_MARKER_VM_PATH,
) -> None:
    """Mount personal My Drive using the injected jlens OAuth remote."""
    if extract_rclone_remote(config) is None:
        raise RuntimeError("Injected rclone config has no jlens remote")
    config.chmod(0o600)
    _mount_rclone_remote(
        "jlens",
        mydrive_root=mydrive_root,
        runner=runner,
        sleep=sleep,
        marker=marker,
        rclone_args=("--config", str(config)),
    )


def _mount_google_drive_interactive() -> None:
    from google.colab import drive

    drive.mount("/content/drive")


def ensure_colab_drive(
    *,
    sa_json: Path = SA_JSON_VM_PATH,
    rclone_config: Path = RCLONE_CONFIG_VM_PATH,
    mydrive_root: Path = MYDRIVE_ROOT,
    interactive_mounter: Callable[[], None] | None = None,
    sa_mounter: Callable[..., None] | None = None,
    rclone_mounter: Callable[..., None] | None = None,
    runner: CommandRunner = _default_runner,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Select injected credentials, or preserve interactive browser mounting."""
    env = dict(os.environ if environ is None else environ)
    apply_env_files(environ=env)
    mode = select_drive_auth(
        rclone_present=rclone_config.is_file(),
        sa_present=sa_json.is_file(),
        mode=env.get("JLENS_DRIVE_AUTH", "auto"),
        allow_interactive=True,
    )
    mountpoint = mydrive_root.parent if mode == "interactive" else mydrive_root
    if mountpoint.is_mount() and drive_layout_ready(mydrive_root):
        return mode
    if mode != "interactive":
        credential = rclone_config if mode == "rclone" else sa_json
        mounter = (
            (rclone_mounter or mount_drive_with_rclone_config)
            if mode == "rclone"
            else (sa_mounter or mount_drive_with_service_account)
        )
        label = "rclone" if mode == "rclone" else "service-account"
        try:
            mounter(credential, mydrive_root=mydrive_root, runner=runner, environ=env)
        except Exception:
            raise RuntimeError(
                f"Google Drive {label} mount failed; check credentials, permissions and project layout"
            ) from None
        if not drive_layout_ready(mydrive_root):
            raise RuntimeError(f"Google Drive {label} mount failed")
        return mode
    mounter = interactive_mounter or _mount_google_drive_interactive
    try:
        mounter()
    except Exception:
        raise RuntimeError("Google Drive mount failed") from None
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-credentials", type=Path)
    args, _ = parser.parse_known_args()
    try:
        if args.stage_credentials is not None:
            print(stage_colab_credentials(args.stage_credentials))
        else:
            mode = ensure_colab_drive()
            print(f"Colab Drive ready via {mode}")
    except (RuntimeError, OSError) as error:
        # Config parsing and mount boundaries redact backend/credential details.
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
