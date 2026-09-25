"""Input and feature identities shared by probe training and analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from jlens_reasoning.inference import (
    InferenceConfig,
    InferenceMode,
    chat_input_fingerprint,
)


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    """Choose chat preparation and the token whose features are probed.

    Layer l uses Transformers hidden_states[l + 1], including the final
    normalized entry. Positions index the complete, unpadded chat input.
    """

    inference: InferenceConfig
    token_position: int = -1

    def __post_init__(self) -> None:
        if type(self.token_position) is not int:
            raise ValueError("Probe token position must be an integer")


def probe_input_contract(tokenizer: Any, *, config: ProbeConfig) -> dict:
    """Identify tokenizer assets, chat settings and the feature boundary.

    Direct, final-token configurations retain the existing v2 artifact schema.
    Padding/truncation runtime state is excluded from the tokenizer fingerprint.
    """
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("padding", None)
    backend.pop("truncation", None)
    identity = {
        "backend": backend,
        "chat_template": tokenizer.get_chat_template(),
        "special_tokens": tokenizer.special_tokens_map,
        "tokenizer_class": type(tokenizer).__name__,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {
        "format_version": 2,
        "input_format": f"chat_template_{config.inference.mode.value}",
        "input_contract": {
            "tokenizer_sha256": digest,
            "add_generation_prompt": True,
            "enable_thinking": config.inference.mode is InferenceMode.REASONING,
            "max_input_tokens": config.inference.max_input_tokens,
        },
        "feature_contract": "hidden_states[layer + 1]; final entry is post-final-norm",
        "feature_position": (
            "final wrapped input token at index -1 before generation"
            if config.token_position == -1
            else f"wrapped input token at index {config.token_position} before generation"
        ),
    }


def validate_probe_input_contract(artifact: Mapping, expected: Mapping) -> None:
    """Reject probes/results produced with incompatible input or feature settings."""
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Incompatible probe input contract. Retrain probes and regenerate "
            "results with matching tokenizer, chat settings and feature position."
        )


def validate_input_record(record: Mapping, inputs: Mapping[str, torch.Tensor]) -> None:
    """Verify saved outputs against the exact token IDs and attention mask."""
    if (
        record.get("input_sha256") != chat_input_fingerprint(inputs)
        or record.get("n_input_tokens") != inputs["input_ids"].shape[1]
    ):
        raise ValueError(
            "Saved record lacks matching chat input tokens/mask. Regenerate "
            "the saved outputs using the current input configuration."
        )
