# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "huggingface-hub==1.24.0",
#   "PyYAML==6.0.3",
# ]
# ///
"""Download pinned Hugging Face assets and sync them to Google Drive."""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY / "assets" / "colab.yaml"
COMMIT_ID = re.compile(r"^[0-9a-f]{40}$")
SNAPSHOT_MARKER = ".jlens-sync.json"


@dataclass(frozen=True)
class Asset:
    id: str
    mode: str
    repo_id: str
    revision: str
    destination: str
    filename: str | None = None


@dataclass(frozen=True)
class DriveManifest:
    drive_root: str
    assets: tuple[Asset, ...]


@dataclass(frozen=True)
class SyncSummary:
    skipped: tuple[str, ...]
    synced: tuple[str, ...]


def _check_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")


def _validate_asset(entry: dict[str, Any]) -> None:
    if entry.get("source") != "huggingface":
        raise ValueError("assets.source must be 'huggingface'")
    if entry["mode"] not in {"snapshot", "file"}:
        raise ValueError("assets.mode must be 'snapshot' or 'file'")
    if not COMMIT_ID.fullmatch(entry["revision"]):
        raise ValueError("assets.revision must be a 40-character commit ID")
    
    filename = entry.get("filename")
    if entry["mode"] == "file":
        if not filename:
            raise ValueError("file assets require assets.filename")
        _check_relative_path(filename, "assets.filename")

def load_manifest(path: Path) -> DriveManifest:
    """Read and validate the small YAML asset manifest."""

    with path.open() as file:
        data = yaml.safe_load(file)

    drive_root = data["drive"]["root"]
    _check_relative_path(drive_root, "drive.root")

    assets = []
    for entry in data["assets"]:
        _validate_asset(entry)
        filename = entry.get("filename")

        assets.append(
            Asset(
                id=entry["id"],
                mode=entry["mode"],
                repo_id=entry["repo_id"],
                revision=entry["revision"],
                destination=entry["destination"],
                filename=filename if entry["mode"] == "file" else None,
            )
        )

    if not assets:
        raise ValueError("manifest must contain at least one asset")
    if len({asset.id for asset in assets}) != len(assets):
        raise ValueError("asset IDs must be unique")

    return DriveManifest(drive_root, tuple(assets))


def marker_matches(asset: Asset, marker: Mapping[str, Any]) -> bool:
    return marker == {"asset": asset.id, "revision": asset.revision}


def _marker_path(asset: Asset) -> PurePosixPath:
    destination = PurePosixPath(asset.destination)
    if asset.mode == "snapshot":
        return destination / SNAPSHOT_MARKER
    return destination.parent / f".{asset.id}.jlens-sync.json"


Runner = Callable[..., subprocess.CompletedProcess[str]]


class RcloneDrive:
    """Minimal wrapper around an existing rclone remote."""

    def __init__(
        self,
        drive_root: str,
        remote: str = "jlens",
        *,
        runner: Runner = subprocess.run,
    ) -> None:
        self.drive_root = PurePosixPath(drive_root)
        self.remote = remote.removesuffix(":")
        self.runner = runner

    def _remote_path(self, path: PurePosixPath | str) -> str:
        return f"{self.remote}:{self.drive_root / path}"

    def _run(
        self,
        *arguments: str,
        check: bool = True,
        show_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self.runner(
                ["rclone", *arguments],
                text=True,
                capture_output=not show_output,
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError("rclone is not installed") from error

        if check and result.returncode:
            raise RuntimeError((result.stderr or "").strip() or "rclone command failed")
        return result

    def is_current(self, asset: Asset) -> bool:
        result = self._run(
            "cat",
            self._remote_path(_marker_path(asset)),
            check=False,
        )
        if result.returncode:
            return False
        try:
            return marker_matches(asset, json.loads(result.stdout))
        except json.JSONDecodeError:
            return False

    def replace(self, asset: Asset, local_path: Path) -> None:
        marker_file = local_path.parent / f".{asset.id}.jlens-sync.json"
        marker_file.write_text(
            json.dumps({"asset": asset.id, "revision": asset.revision}, indent=2)
        )
        remote_marker = self._remote_path(_marker_path(asset))

        # Write the new marker only after the asset transfer succeeds.
        self._run("deletefile", remote_marker, check=False)
        if asset.mode == "snapshot":
            self._run(
                "sync",
                str(local_path),
                self._remote_path(asset.destination),
                "--ignore-times",
                "--drive-use-trash=true",
                "--exclude",
                ".cache/**",
                "--progress",
                show_output=True,
            )
        else:
            self._run(
                "copyto",
                str(local_path),
                self._remote_path(asset.destination),
                "--ignore-times",
                "--progress",
                show_output=True,
            )
        self._run("copyto", str(marker_file), remote_marker)


def download_asset(asset: Asset, temp_directory: Path) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    destination = temp_directory / asset.id
    if asset.mode == "snapshot":
        snapshot_download(
            repo_id=asset.repo_id,
            revision=asset.revision,
            local_dir=destination,
        )
        return destination

    return Path(
        hf_hub_download(
            repo_id=asset.repo_id,
            revision=asset.revision,
            filename=asset.filename,
            local_dir=destination,
        )
    )


Downloader = Callable[[Asset, Path], Path]


def sync_assets(
    manifest: DriveManifest,
    drive: RcloneDrive,
    downloader: Downloader = download_asset,
    *,
    force: bool = False,
) -> SyncSummary:
    skipped = []
    synced = []

    for asset in manifest.assets:
        if not force and drive.is_current(asset):
            print(f"skip {asset.id}: revision matches")
            skipped.append(asset.id)
            continue

        with tempfile.TemporaryDirectory(prefix=f"jlens-{asset.id}-") as temp:
            print(f"download {asset.id}")
            local_path = downloader(asset, Path(temp))  # 1. Download

            print(f"sync {asset.id}")
            drive.replace(asset, local_path)  # 2. Sync to Drive

        # 3. TemporaryDirectory cleans the local files here.
        synced.append(asset.id)

    return SyncSummary(tuple(skipped), tuple(synced))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--remote",
        default="jlens",
        help="name of the configured rclone Google Drive remote",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace assets even when the revision marker matches",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest)
        summary = sync_assets(
            manifest,
            RcloneDrive(manifest.drive_root, args.remote),
            force=args.force,
        )
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"finished: {len(summary.synced)} synced, {len(summary.skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
