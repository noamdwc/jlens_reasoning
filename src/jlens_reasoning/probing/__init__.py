"""Project-wide probing: features, fitting, scoring, artifacts and objectives.

Experiments supply their data, labels, splits, configurations and output
objectives. Probe/J-Lens analysis lives separately in jlens_reasoning.probe_jlens.
"""

from __future__ import annotations

from .artifacts import load_probe_checkpoint, save_probe_checkpoint
from .contracts import (
    ProbeConfig,
    probe_input_contract,
    validate_input_record,
    validate_probe_input_contract,
)
from .features import ProbeFeatures, extract_probe_features
from .linear import (
    ProbeEvaluation,
    binary_probe_metrics,
    evaluate_probe,
    fit_binary_probe,
    score_probe,
    unit_probe_direction,
)
from .objectives import token_margin

__all__ = [
    "ProbeEvaluation",
    "evaluate_probe",
    "ProbeConfig",
    "ProbeFeatures",
    "binary_probe_metrics",
    "extract_probe_features",
    "fit_binary_probe",
    "load_probe_checkpoint",
    "probe_input_contract",
    "save_probe_checkpoint",
    "score_probe",
    "token_margin",
    "unit_probe_direction",
    "validate_input_record",
    "validate_probe_input_contract",
]
