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
