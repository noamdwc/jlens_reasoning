"""One feature-extraction path for probe training, evaluation and gradients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from jlens_reasoning.inference import chat_input_fingerprint, prepare_chat_inputs

from .contracts import ProbeConfig, validate_input_record


@dataclass(frozen=True, slots=True)
class ProbeFeatures:
    """One vector per layer, final-token logits, and the input identity used.

    States and logits are detached CPU float32 tensors. Each state has shape
    [hidden_dim]; logits has shape [vocab_size].
    """

    states: tuple[torch.Tensor, ...]
    logits: torch.Tensor
    input_record: dict


def prepare_probe_inputs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    config: ProbeConfig,
    input_record: Mapping | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare aligned unpadded inputs on the model device without changing mode."""
    if model.training:
        raise ValueError("Probe extraction requires model.eval()")
    inputs = prepare_chat_inputs(tokenizer, prompt, config=config.inference)
    if input_record is not None:
        validate_input_record(input_record, inputs)
    num_tokens = inputs["input_ids"].shape[1]
    if not -num_tokens <= config.token_position < num_tokens:
        raise ValueError("Probe token position is outside the wrapped input")
    return {key: value.to(model.device) for key, value in inputs.items()}


def probe_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    *,
    num_layers: int,
) -> tuple[torch.Tensor, ...]:
    """Select layer outputs, retaining full tensors for autograd.

    The last entry is post-final-normalization, not a raw block output.
    """
    if hidden_states is None or len(hidden_states) != num_layers + 1:
        raise ValueError("Expected embedding state followed by one state per layer")
    return tuple(hidden_states[1:])


def extract_probe_features(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    config: ProbeConfig,
    input_record: Mapping | None = None,
) -> ProbeFeatures:
    """Return CPU float32 probe states and next-token logits for one chat input."""
    inputs = prepare_probe_inputs(
        model, tokenizer, prompt, config=config, input_record=input_record
    )
    with torch.inference_mode():
        outputs = model(
            **inputs, output_hidden_states=True, logits_to_keep=1, use_cache=False
        )
    states = probe_hidden_states(
        outputs.hidden_states, num_layers=int(model.config.num_hidden_layers)
    )
    return ProbeFeatures(
        states=tuple(
            state[0, config.token_position].detach().float().cpu() for state in states
        ),
        logits=outputs.logits[0, -1].detach().float().cpu(),
        input_record={
            "input_sha256": chat_input_fingerprint(inputs),
            "n_input_tokens": int(inputs["input_ids"].shape[1]),
        },
    )
