"""Input and artifact compatibility for the aligned probe experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch

from jlens_reasoning.inference import InferenceConfig, chat_input_fingerprint

PROBE_INPUT_CONFIG = InferenceConfig.direct(max_input_tokens=4096)


def probe_input_contract(tokenizer: Any) -> dict:
    """Fingerprint the fast tokenizer and the exact direct-chat feature boundary."""
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # These are transient settings mutated by tokenizer calls, not tokenizer assets.
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
        "input_format": "chat_template_direct",
        "input_contract": {
            "tokenizer_sha256": digest,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "max_input_tokens": PROBE_INPUT_CONFIG.max_input_tokens,
        },
        "feature_contract": "hidden_states[layer + 1]; final entry is post-final-norm",
        "feature_position": "final wrapped input token at index -1 before generation",
    }


def validate_probe_input_contract(artifact: Mapping, expected: Mapping) -> None:
    """Refuse raw-input or differently tokenized probes/results before analysis."""
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Incompatible probe input contract. Retrain chat-format probes with "
            "flenqa_probe_assets.ipynb and rerun flenqa_probe_eval.ipynb; "
            "keep the existing problem split."
        )


def validate_input_record(record: Mapping, inputs: Mapping[str, torch.Tensor]) -> None:
    """Require evidence that saved answers/scores used these very model inputs."""
    if (
        record.get("input_sha256") != chat_input_fingerprint(inputs)
        or record.get("n_input_tokens") != inputs["input_ids"].shape[1]
    ):
        raise ValueError(
            "Saved record lacks matching chat input tokens/mask. Regenerate "
            "model answers with the current wheel, then rerun probe evaluation."
        )
