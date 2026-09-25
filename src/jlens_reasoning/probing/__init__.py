"""Project-wide probing: features, fitting, scoring, artifacts and objectives.

Experiments supply their data, labels, splits, configurations and output
objectives. Probe/J-Lens analysis lives separately in jlens_reasoning.probe_jlens.
"""

from __future__ import annotations

from .contracts import (
    ProbeConfig,
    probe_input_contract,
    validate_input_record,
    validate_probe_input_contract,
)
from .features import ProbeFeatures, extract_probe_features
from .objectives import token_margin

__all__ = [
    "ProbeConfig",
    "ProbeFeatures",
    "extract_probe_features",
    "probe_input_contract",
    "token_margin",
    "validate_input_record",
    "validate_probe_input_contract",
]
