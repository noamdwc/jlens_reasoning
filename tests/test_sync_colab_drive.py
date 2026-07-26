import json
import subprocess
from pathlib import Path

import pytest

from scripts.sync_colab_drive import (
    Asset,
    DriveManifest,
    RcloneDrive,
    load_manifest,
    marker_matches,
    sync_assets,
)


class FakeDrive:
    def __init__(self, current: set[str] | None = None, *, fail=False) -> None:
        self.current = current or set()
        self.fail = fail
        self.uploads: list[tuple[str, bool]] = []

    def is_current(self, asset: Asset) -> bool:
        return asset.id in self.current

    def replace(self, asset: Asset, local_path: Path) -> None:
        self.uploads.append((asset.id, local_path.exists()))
        if self.fail:
            raise RuntimeError("upload failed")


class FakeDownloader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.temp_directories: list[Path] = []

    def __call__(self, asset: Asset, temp_directory: Path) -> Path:
        self.calls.append(asset.id)
        self.temp_directories.append(temp_directory)

        if asset.mode == "snapshot":
            result = temp_directory / asset.id
            result.mkdir()
            (result / "config.json").write_text("{}")
            return result

        result = temp_directory / Path(asset.filename or "").name
        result.write_bytes(b"lens")
        return result


class FakeRclone:
    def __init__(
        self,
        responses: dict[tuple[str, ...], subprocess.CompletedProcess[str]]
        | None = None,
    ) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, ...]] = []
        self.visible_calls: list[tuple[str, ...]] = []

    def __call__(self, command, **kwargs):
        args = tuple(command[1:])
        self.calls.append(args)
        if not kwargs["capture_output"]:
            self.visible_calls.append(args)
        return self.responses.get(
            args,
            subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
        )


def example_manifest() -> DriveManifest:
    return DriveManifest(
        drive_root="data/jlens-reasoning",
        assets=(
            Asset(
                id="model",
                mode="snapshot",
                repo_id="example/model",
                revision="a" * 40,
                destination="assets/models/example",
            ),
            Asset(
                id="lens",
                mode="file",
                repo_id="example/lenses",
                revision="b" * 40,
                destination="assets/lenses/example.pt",
                filename="weights/example.pt",
            ),
        ),
    )


def rclone(fake: FakeRclone) -> RcloneDrive:
    return RcloneDrive(
        drive_root=example_manifest().drive_root,
        remote="jlens",
        runner=fake,
    )


def test_repository_manifest_contains_two_pinned_assets() -> None:
    manifest = load_manifest(Path("assets/colab.yaml"))

    assert manifest.drive_root == "data/jlens-reasoning"
    assert [asset.id for asset in manifest.assets] == [
        "qwen3.5-4b",
        "qwen3.5-4b-jacobian-lens-n1000",
    ]
    assert all(len(asset.revision) == 40 for asset in manifest.assets)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("drive.root", "/absolute/path"),
        ("assets.destination", "assets/../outside"),
        ("assets.filename", "../outside.pt"),
    ],
)
def test_manifest_rejects_unsafe_paths(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    drive_root = value if field == "drive.root" else "data/jlens-reasoning"
    destination = value if field == "assets.destination" else "assets/lens.pt"
    filename = value if field == "assets.filename" else "weights/lens.pt"
    path = tmp_path / "unsafe.yaml"
    path.write_text(
        f"""
drive:
  root: {drive_root}
assets:
  - id: unsafe
    source: huggingface
    mode: file
    repo_id: example/model
    revision: {"a" * 40}
    filename: {filename}
    destination: {destination}
"""
    )

    with pytest.raises(ValueError, match=field):
        load_manifest(path)


def test_marker_matches_only_the_asset_and_revision() -> None:
    asset = example_manifest().assets[0]
    marker = {"asset": asset.id, "revision": asset.revision}

    assert marker_matches(asset, marker)
    assert not marker_matches(asset, {**marker, "revision": "b" * 40})
    assert not marker_matches(asset, {**marker, "files": {"config.json": "md5"}})


def test_rclone_skips_when_revision_marker_matches() -> None:
    asset = example_manifest().assets[0]
    marker_path = "jlens:data/jlens-reasoning/assets/models/example/.jlens-sync.json"
    fake = FakeRclone(
        {
            ("cat", marker_path): subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps({"asset": asset.id, "revision": asset.revision}),
                stderr="",
            )
        }
    )

    assert rclone(fake).is_current(asset)
    assert fake.calls == [("cat", marker_path)]


def test_rclone_writes_snapshot_marker_after_sync(tmp_path: Path) -> None:
    asset = example_manifest().assets[0]
    local = tmp_path / "model"
    local.mkdir()
    (local / "config.json").write_text("{}")
    fake = FakeRclone()

    rclone(fake).replace(asset, local)

    remote = "jlens:data/jlens-reasoning/assets/models/example"
    remote_marker = f"{remote}/.jlens-sync.json"
    assert fake.calls[0] == ("deletefile", remote_marker)
    assert fake.calls[1] == (
        "sync",
        str(local),
        remote,
        "--ignore-times",
        "--drive-use-trash=true",
        "--exclude",
        ".cache/**",
        "--progress",
    )
    assert fake.calls[2][0] == "copyto"
    assert fake.calls[2][2] == remote_marker
    assert fake.visible_calls == [fake.calls[1]]
    marker = json.loads(Path(fake.calls[2][1]).read_text())
    assert marker == {"asset": asset.id, "revision": asset.revision}


def test_rclone_copies_a_file_then_writes_its_marker(tmp_path: Path) -> None:
    asset = example_manifest().assets[1]
    local = tmp_path / "example.pt"
    local.write_bytes(b"lens")
    fake = FakeRclone()

    rclone(fake).replace(asset, local)

    remote_file = "jlens:data/jlens-reasoning/assets/lenses/example.pt"
    remote_marker = "jlens:data/jlens-reasoning/assets/lenses/.lens.jlens-sync.json"
    assert fake.calls[0] == ("deletefile", remote_marker)
    assert fake.calls[1] == (
        "copyto",
        str(local),
        remote_file,
        "--ignore-times",
        "--progress",
    )
    assert fake.calls[2][0] == "copyto"
    assert fake.calls[2][2] == remote_marker
    assert fake.visible_calls == [fake.calls[1]]


def test_sync_skips_current_assets_by_default() -> None:
    downloader = FakeDownloader()
    drive = FakeDrive(current={"model", "lens"})

    summary = sync_assets(example_manifest(), drive, downloader)

    assert summary.skipped == ("model", "lens")
    assert summary.synced == ()
    assert downloader.calls == []
    assert drive.uploads == []


def test_force_syncs_current_assets_and_cleans_temp_files() -> None:
    downloader = FakeDownloader()
    drive = FakeDrive(current={"model", "lens"})

    summary = sync_assets(example_manifest(), drive, downloader, force=True)

    assert summary.skipped == ()
    assert summary.synced == ("model", "lens")
    assert downloader.calls == ["model", "lens"]
    assert drive.uploads == [("model", True), ("lens", True)]
    assert all(not path.exists() for path in downloader.temp_directories)


def test_sync_cleans_temp_files_when_upload_fails() -> None:
    downloader = FakeDownloader()
    manifest = DriveManifest(
        drive_root=example_manifest().drive_root,
        assets=(example_manifest().assets[0],),
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        sync_assets(manifest, FakeDrive(fail=True), downloader)

    assert all(not path.exists() for path in downloader.temp_directories)
