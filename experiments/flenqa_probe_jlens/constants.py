"""FLenQA choices for the shared probing pipeline."""

from __future__ import annotations

from jlens_reasoning.inference import InferenceConfig
from jlens_reasoning.probing import ProbeConfig

PROBE_CONFIG = ProbeConfig(InferenceConfig.direct(max_input_tokens=4096))
