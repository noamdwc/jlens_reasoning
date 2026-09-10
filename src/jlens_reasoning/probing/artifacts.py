"""Validated tensor checkpoints and readable metadata for frozen probes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from .contracts import validate_probe_input_contract
from .linear import unit_probe_direction


def _validate_checkpoint(checkpoint: Mapping) -> None:
    metadata = checkpoint.get("metadata", {})
    if checkpoint.get("format_version") != 2 or metadata.get("format_version") != 2:
        raise ValueError(
            "Unsupported probe checkpoint version. Retrain compatible probes."
        )
    layers = checkpoint.get("layers", {})
    count, width = metadata.get("num_layers"), metadata.get("hidden_dim")
    if type(count) is not int or count <= 0 or type(width) is not int or width <= 0:
        raise ValueError(
            "Probe metadata must specify positive num_layers and hidden_dim"
        )
    if set(layers) != set(range(count)):
        raise ValueError("Probe checkpoint must contain every declared layer")
    for probe in layers.values():
        if unit_probe_direction(probe).numel() != width:
            raise ValueError("Probe width does not match checkpoint metadata")


def save_probe_checkpoint(
    path: str | Path,
    layers: Mapping,
    *,
    metadata: Mapping,
    metadata_path: str | Path | None = None,
) -> None:
    """Save the v2 tensor layout; optionally write its matching JSON sidecar."""
    checkpoint = {
        "format_version": 2,
        "metadata": dict(metadata),
        "layers": dict(layers),
    }
    _validate_checkpoint(checkpoint)
    metadata_json = json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    if metadata_path is not None:
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).write_text(metadata_json, encoding="utf-8")


def load_probe_checkpoint(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    expected_contract: Mapping | None = None,
    model_name: str | None = None,
) -> dict:
    """Load safely on CPU and reject malformed, stale or incompatible probes."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _validate_checkpoint(checkpoint)
    metadata = checkpoint["metadata"]
    if metadata_path is not None:
        if json.loads(Path(metadata_path).read_text(encoding="utf-8")) != metadata:
            raise ValueError("Probe checkpoint and sidecar metadata disagree")
    if expected_contract is not None:
        validate_probe_input_contract(metadata, expected_contract)
    if model_name is not None and metadata.get("model_name") != model_name:
        raise ValueError("Probe checkpoint model does not match the requested model")
    return checkpoint
