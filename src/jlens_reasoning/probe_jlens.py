"""Combine learned probes with J-Lens transport and prompt-specific derivatives.

This module owns the probe/J-Lens analysis layer and will host the routing
framework described in docs/probe_jlens_routing_framework.md. Core probing owns
features, fitting, scoring and artifacts; dependencies run from here to probing.

Current sensitivities differentiate a scalar output objective along the frozen
probe axis. They are not the framework's final-hidden-state J-sensitivity norm.
J-relevant components, J-merging and second-order J-gain are future work.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from jlens.hooks import ActivationRecorder

from .probing import (
    ProbeConfig,
    score_probe,
    unit_probe_direction,
    validate_input_record,
)
from .probing.features import prepare_probe_inputs, probe_hidden_states

__all__ = ["ProbeSensitivity", "probe_sensitivities", "static_probe_projection"]


def static_probe_projection(
    jlens_jacobian: torch.Tensor,
    probe_weight: torch.Tensor,
    unembedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return J @ unit_probe and W_U @ projected_probe as CPU float32 tensors.

    J rows are target coordinates; columns are source coordinates. Supply a map
    for the same feature boundary as the probe. Block-output maps cannot accept
    the final normalized probe. Scores omit final normalization; they are not
    normalized lens logits.
    """
    direction = unit_probe_direction(probe_weight)
    jacobian, unembedding = (
        jlens_jacobian.detach().float().cpu(),
        unembedding.detach().float().cpu(),
    )
    if (
        jacobian.ndim != 2
        or jacobian.shape[1] != direction.numel()
        or unembedding.ndim != 2
        or unembedding.shape[1] != jacobian.shape[0]
    ):
        raise ValueError("Projection dimensions do not match the probe and unembedding")
    projected = jacobian @ direction
    scores = unembedding @ projected
    if not torch.isfinite(projected).all() or not torch.isfinite(scores).all():
        raise ValueError("Non-finite projected vocabulary scores")
    return projected, scores


@dataclass(frozen=True, slots=True)
class ProbeSensitivity:
    layer: int
    probe_score: float
    output_margin: float
    sensitivity: float


def probe_sensitivities(
    model: Any,
    tokenizer: Any,
    prompt: str,
    probes: Mapping[int, Mapping],
    *,
    config: ProbeConfig,
    blocks: Sequence[torch.nn.Module],
    objective: Callable[[torch.Tensor], torch.Tensor],
    saved_records: Mapping[int, Mapping] | None = None,
    rtol: float = 0.02,
    atol: float = 0.05,
) -> tuple[ProbeSensitivity, ...]:
    """Differentiate an output scalar along each positive-class unit probe.

    The caller supplies model blocks and the next-token logit objective. This
    function owns autograd hooks, feature selection, scoring and saved-result
    verification. It does not change model parameter requires_grad flags.
    """
    num_layers = int(model.config.num_hidden_layers)
    if not probes or any(
        type(layer) is not int or not 0 <= layer < num_layers for layer in probes
    ):
        raise ValueError("Probe layers must be nonempty valid model layer indices")
    if len(blocks) != num_layers:
        raise ValueError("Expected one model block per hidden-state layer")
    inputs = prepare_probe_inputs(model, tokenizer, prompt, config=config)
    if saved_records is not None:
        for layer in probes:
            if layer not in saved_records:
                raise ValueError("Missing saved probe record for a requested layer")
            validate_input_record(saved_records[layer], inputs)
    layers = sorted(probes)
    with (
        torch.enable_grad(),
        ActivationRecorder(blocks, at=range(num_layers), start_graph_at=0) as recorder,
    ):
        outputs = model(
            **inputs, output_hidden_states=True, logits_to_keep=1, use_cache=False
        )
        states = probe_hidden_states(outputs.hidden_states, num_layers=num_layers)
        # Transformers replaces the last entry with the final normalized state.
        for layer in layers:
            if layer < num_layers - 1:
                torch.testing.assert_close(
                    states[layer][0, config.token_position],
                    recorder.activations[layer][0, config.token_position],
                )
        margin = objective(outputs.logits[0, -1].float())
        if margin.ndim != 0 or not torch.isfinite(margin):
            raise ValueError("Probe output objective must return a finite scalar")
        gradients = torch.autograd.grad(
            margin, tuple(states[layer] for layer in layers)
        )
    value = float(margin.detach().cpu())
    result = []
    for layer, gradient in zip(layers, gradients, strict=True):
        score = float(
            score_probe(probes[layer], states[layer][0, config.token_position])
        )
        sensitivity = float(
            gradient[0, config.token_position].detach().float().cpu()
            @ unit_probe_direction(probes[layer])
        )
        if not math.isfinite(sensitivity):
            raise ValueError("Non-finite probe sensitivity")
        if saved_records is not None:
            saved = saved_records[layer]
            for name, recomputed in (("probe_score", score), ("output_margin", value)):
                if name not in saved or not np.isclose(
                    recomputed, float(saved[name]), rtol=rtol, atol=atol
                ):
                    raise ValueError(
                        f"Recomputed {name} disagrees with saved probe results"
                    )
        result.append(ProbeSensitivity(layer, score, value, sensitivity))
    return tuple(result)
