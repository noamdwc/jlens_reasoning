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
        raise RuntimeError(f"Artifact root is not writable: {artifact_root}") from exc

    return paths
