"""Validated tensor checkpoints and readable metadata for frozen probes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from .contracts import validate_probe_input_contract
from .linear import _probe_tensors


def _validate_checkpoint(checkpoint: Mapping) -> None:
    metadata = checkpoint.get("metadata", {})
    if checkpoint.get("format_version") != 2 or metadata.get("format_version") != 2:
        raise ValueError(
            "Unsupported probe checkpoint version. Retrain compatible probes."
        )
    layers = checkpoint.get("layers", {})
    num_layers = metadata.get("num_layers")
    hidden_dim = metadata.get("hidden_dim")
    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError("Probe metadata must specify positive num_layers")
    if type(hidden_dim) is not int or hidden_dim <= 0:
        raise ValueError("Probe metadata must specify positive hidden_dim")
    if set(layers) != set(range(num_layers)):
        raise ValueError("Probe checkpoint must contain every declared layer")
    for probe in layers.values():
        weight, _, _ = _probe_tensors(probe)
        if weight.numel() != hidden_dim:
            raise ValueError("Probe width does not match checkpoint metadata")
        norm = weight.norm()
        if not torch.isfinite(norm) or norm == 0:
            raise ValueError("Probe weight must be a finite nonzero vector")


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
        sidecar_metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if sidecar_metadata != metadata:
            raise ValueError("Probe checkpoint and sidecar metadata disagree")
    if expected_contract is not None:
        validate_probe_input_contract(metadata, expected_contract)
    if model_name is not None and metadata.get("model_name") != model_name:
        raise ValueError("Probe checkpoint model does not match the requested model")
    return checkpoint
