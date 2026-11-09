"""Combine learned probes with J-Lens transport and prompt-specific derivatives.

This module owns the probe/J-Lens analysis layer and the routing
framework described in docs/probe_jlens_routing_framework.md. Core probing owns
features, fitting, scoring and artifacts; dependencies run from here to probing.

Output-margin sensitivities and final-hidden-state J-sensitivity are distinct
quantities. J-gain changes the latter outside a fixed one-dimensional probe axis.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
from jlens.hooks import ActivationRecorder
from torch.utils.checkpoint import checkpoint

from .inference import chat_input_fingerprint
from .probing import (
    ProbeConfig,
    score_probe,
    unit_probe_direction,
    validate_input_record,
)
from .probing.features import prepare_probe_inputs, probe_hidden_states

__all__ = [
    "JGain",
    "PromptHiddenMap",
    "ProbeSensitivity",
    "j_gain",
    "j_sensitivity",
    "probe_component",
    "prompt_hidden_map",
    "probe_sensitivities",
    "static_probe_projection",
]


def static_probe_projection(
    jlens_jacobian: torch.Tensor,
    probe_weight: torch.Tensor,
    unembedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return J @ unit_probe and W_U @ projected_probe as CPU float32 tensors.

    For probe weight w, Jacobian J and unembedding matrix W_U::

        unit_probe = w / ||w||_2
        projected = J @ unit_probe
        scores = W_U @ projected

    Returns (projected, scores), where scores has one entry per vocabulary token.

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


@contextmanager
def _checkpoint_blocks(blocks: Sequence[torch.nn.Module]) -> Iterator[None]:
    """Recompute block internals during backward, retaining the probed outputs."""
    forwards = [block.forward for block in blocks]
    try:
        for block, forward in zip(blocks, forwards, strict=True):
            # Non-reentrant checkpointing supports autograd.grad in eval mode.
            # Wrap forward itself so recorder hooks run only on the initial pass.
            block.forward = partial(checkpoint, forward, use_reentrant=False)
        yield
    finally:
        for block, forward in zip(blocks, forwards, strict=True):
            block.forward = forward


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

    Let h_l be the layer-l state at config.token_position, w_l the probe weight,
    mu_l its training mean, b_l its bias, and z the final-position next-token
    logits. Each returned record contains::

        probe_score = (h_l - mu_l) @ w_l + b_l
        output_margin = objective(z)
        unit_probe_l = w_l / ||w_l||_2
        sensitivity = grad_h_l(output_margin) @ unit_probe_l

    Sensitivity is the local derivative for perturbing only the selected token
    along unit_probe_l. For a small step epsilon, the first-order output change
    is epsilon * sensitivity. Despite its name, output_margin can be any scalar
    supplied by objective; it need not be a token-logit margin.

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
    # Rebuild the input and check that it matches any saved probe results.
    inputs = prepare_probe_inputs(model, tokenizer, prompt, config=config)
    if saved_records is not None:
        for layer in probes:
            if layer not in saved_records:
                raise ValueError("Missing saved probe record for a requested layer")
            validate_input_record(saved_records[layer], inputs)
    layers = sorted(probes)
    with (
        torch.enable_grad(),
        # Offloading every intermediate to CPU can exhaust Colab host RAM.
        _checkpoint_blocks(blocks),
        ActivationRecorder(blocks, at=range(num_layers), start_graph_at=0) as recorder,
    ):
        # Keep the computation graph linking layer states to next-token logits.
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
        # output_margin = objective(z), where z is the final-position logit vector.
        margin = objective(outputs.logits[0, -1].float())
        if margin.ndim != 0 or not torch.isfinite(margin):
            raise ValueError("Probe output objective must return a finite scalar")
        # Differentiate the scalar margin with respect to each full layer tensor.
        # Each gradient has shape [batch, tokens, hidden_dim]; selecting the
        # probed token below gives grad_h_l(output_margin).
        gradients = torch.autograd.grad(
            margin, tuple(states[layer] for layer in layers)
        )
    value = float(margin.detach().cpu())
    result = []
    for layer, gradient in zip(layers, gradients, strict=True):
        # Probe readout at h_l: (h_l - training_mean_l) @ w_l + b_l.
        score = float(
            score_probe(probes[layer], states[layer][0, config.token_position])
        )
        # Select grad_h_l and project it onto w_l / ||w_l||_2. This scalar
        # measures the local margin change per unit step along the probe axis.
        sensitivity = float(
            gradient[0, config.token_position].detach().float().cpu()
            @ unit_probe_direction(probes[layer])
        )
        if not math.isfinite(sensitivity):
            raise ValueError("Non-finite probe sensitivity")
        # Confirm this forward pass reproduces the saved score and objective.
        if saved_records is not None:
            saved = saved_records[layer]
            for name, recomputed in (("probe_score", score), ("output_margin", value)):
                if name not in saved:
                    raise ValueError(f"Layer {layer}: missing saved {name}")
                saved_value = float(saved[name])
                if not np.isclose(recomputed, saved_value, rtol=rtol, atol=atol):
                    raise ValueError(
                        f"Layer {layer}: recomputed {name}={recomputed:.6g} "
                        f"disagrees with saved {name}={saved_value:.6g} "
                        f"(rtol={rtol}, atol={atol})"
                    )
        result.append(ProbeSensitivity(layer, score, value, sensitivity))
    return tuple(result)


def probe_component(probe: Mapping, hidden: torch.Tensor) -> torch.Tensor:
    """Return P @ (h - training_mean) for a loaded one-axis linear probe.

    Bias is deliberately excluded: this is centered geometric content, not the
    classification score. A zero component is returned as zero, not normalized.
    Geometry uses at least float32 and stays on the hidden state's device.
    """
    if hidden.ndim != 1 or not hidden.is_floating_point():
        raise ValueError("Hidden state must be a floating-point vector")
    working = hidden.detach().to(dtype=torch.promote_types(hidden.dtype, torch.float32))
    axis = unit_probe_direction(probe).to(working)
    mean = probe["training_mean"].to(working)
    if hidden.shape != axis.shape or mean.shape != axis.shape:
        raise ValueError("Hidden state and probe mean must match the probe axis")
    centered = working - mean
    if not torch.isfinite(centered).all():
        raise ValueError("Non-finite centered hidden state")
    return (centered @ axis) * axis


@dataclass(frozen=True)
class PromptHiddenMap:
    """Clean source state and a differentiable map to the final normalized state."""

    hidden: torch.Tensor
    final_hidden: Callable[[torch.Tensor], torch.Tensor]
    input_record: dict


def prompt_hidden_map(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    layer: int,
    blocks: Sequence[torch.nn.Module],
    config: ProbeConfig,
    input_record: Mapping | None = None,
) -> PromptHiddenMap:
    """Map one raw block-output token to the final normalized last-token state.

    The source token is config.token_position. All other source activations and
    the prompt are held fixed. Layer numbering matches probes; the last probe
    layer is post-normalization, so it cannot be used as a raw block source.
    The model must be in eval mode with frozen parameters. For the final token,
    cache the fixed causal prefix and differentiate a single-token continuation.
    Earlier source positions replay the full prompt with block checkpointing.
    Attention kernels must support higher-order autograd; unsupported kernels fail.
    Our hooks and changed forwards are removed between calls.
    """
    num_layers = int(model.config.num_hidden_layers)
    if type(layer) is not int or not 0 <= layer < num_layers - 1:
        raise ValueError("Source must be a raw block-output probe layer")
    if len(blocks) != num_layers:
        raise ValueError("Expected one block per model layer")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Freeze model parameters with model.requires_grad_(False)")
    inputs = prepare_probe_inputs(
        model, tokenizer, prompt, config=config, input_record=input_record
    )
    with torch.no_grad(), ActivationRecorder(blocks, at=[layer]) as recorder:
        outputs = model(
            **inputs, output_hidden_states=True, logits_to_keep=1, use_cache=False
        )
        states = probe_hidden_states(outputs.hidden_states, num_layers=num_layers)
        torch.testing.assert_close(states[layer], recorder.activations[layer])
        clean = states[layer][0, config.token_position].detach().clone()
        clean_target = states[-1][0, -1].detach().clone()

    num_tokens = inputs["input_ids"].shape[1]
    cache_prefix = config.token_position in (-1, num_tokens - 1) and num_tokens > 1
    prefix_cache = None
    if cache_prefix:
        with torch.no_grad():
            prefix = model(
                **{key: value[:, :-1] for key, value in inputs.items()},
                logits_to_keep=1,
                use_cache=True,
            )
        prefix_cache = prefix.past_key_values
        if prefix_cache is None:
            raise ValueError("Final-token derivatives require a model prefix cache")

    def final_hidden(hidden: torch.Tensor) -> torch.Tensor:
        if (
            hidden.shape != clean.shape
            or hidden.device != clean.device
            or hidden.dtype != clean.dtype
        ):
            raise ValueError(
                "Replacement must match the clean state's shape, device and dtype"
            )

        def replace_token(module, args, output):
            state = output if torch.is_tensor(output) else output[0]
            # Only this token becomes the independent variable; do not retain
            # gradients through the prefix or other source-layer positions.
            patched = state.detach().clone()
            patched[0, -1 if cache_prefix else config.token_position] = hidden
            return patched if torch.is_tensor(output) else (patched, *output[1:])

        replay_inputs = inputs
        cache_kwargs = {"use_cache": False}
        if cache_prefix:
            # Every derivative/finite-difference call starts from the same prefix.
            cache = copy.deepcopy(prefix_cache)

            def keep_prefix_state(recurrent_states, layer_idx, **kwargs):
                # Qwen overwrites this state in-place after using it in the
                # graph. There is no next decode step here, so discard the update.
                return recurrent_states

            cache.update_recurrent_state = keep_prefix_state
            replay_inputs = {**inputs, "input_ids": inputs["input_ids"][:, -1:]}
            cache_kwargs = {"past_key_values": cache, "use_cache": True}

        handle = blocks[layer].register_forward_hook(replace_token)
        try:
            # Checkpoint recomputation would advance a mutable cache twice.
            checkpointing = (
                nullcontext()
                if cache_prefix
                else _checkpoint_blocks(blocks[layer + 1 :])
            )
            with torch.enable_grad(), checkpointing:
                result = model(
                    **replay_inputs,
                    output_hidden_states=True,
                    logits_to_keep=1,
                    **cache_kwargs,
                )
                return probe_hidden_states(result.hidden_states, num_layers=num_layers)[
                    -1
                ][0, -1]
        finally:
            handle.remove()

    if cache_prefix:
        # Cached recurrent kernels can round differently from full prefill.
        # Check the target boundary before trusting any derivative from this map.
        torch.testing.assert_close(
            final_hidden(clean), clean_target, atol=1e-4, rtol=1e-4
        )

    return PromptHiddenMap(
        hidden=clean,
        final_hidden=final_hidden,
        input_record={
            "input_sha256": chat_input_fingerprint(inputs),
            "n_input_tokens": int(inputs["input_ids"].shape[1]),
        },
    )


def _routing_inputs(
    hidden: torch.Tensor, direction: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Compute routing derivatives outside torch.inference_mode()")
    if hidden.ndim != 1 or hidden.numel() == 0 or not hidden.is_floating_point():
        raise ValueError("Hidden state must be a nonempty floating-point vector")
    if direction.shape != hidden.shape or not direction.is_floating_point():
        raise ValueError("Direction must be a floating-point vector matching the state")
    if not torch.isfinite(hidden).all() or not torch.isfinite(direction).all():
        raise ValueError("Hidden state and direction must be finite")
    # Normalize in at least float32, then use the model's source coordinates.
    axis = direction.detach().to(
        device=hidden.device, dtype=torch.promote_types(hidden.dtype, torch.float32)
    )
    norm = axis.norm()
    if norm == 0 or not torch.isfinite(norm):
        raise ValueError("Direction must be nonzero with a finite norm")
    return hidden.detach().requires_grad_(True), axis / norm


def _directional_jvp(
    final_hidden: Callable[[torch.Tensor], torch.Tensor],
    hidden: torch.Tensor,
    axis: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    target, propagated = torch.autograd.functional.jvp(
        final_hidden, hidden, axis.to(dtype=hidden.dtype), create_graph=create_graph
    )
    if target.ndim != 1 or target.numel() == 0:
        raise ValueError("The final-hidden map must return a nonempty vector")
    if not torch.isfinite(target).all() or not torch.isfinite(propagated).all():
        raise ValueError("Non-finite final hidden state or Jacobian-vector product")
    return propagated.to(dtype=torch.promote_types(propagated.dtype, torch.float32))


def j_sensitivity(
    final_hidden: Callable[[torch.Tensor], torch.Tensor],
    hidden: torch.Tensor,
    direction: torch.Tensor,
) -> float:
    """Return ||J(h) @ unit(direction)|| without materializing the Jacobian.

    Supply the same clean direction when remeasuring an edited state. The
    callable must be deterministic and differentiable in the supplied state.
    A zero direction is undefined and rejected; a zero response is valid.
    """
    with torch.enable_grad():
        state, axis = _routing_inputs(hidden, direction)
        propagated = _directional_jvp(final_hidden, state, axis, create_graph=False)
    return float(propagated.norm().detach())


@dataclass(frozen=True)
class JGain:
    sensitivity: float
    gain: torch.Tensor


def j_gain(
    final_hidden: Callable[[torch.Tensor], torch.Tensor],
    hidden: torch.Tensor,
    direction: torch.Tensor,
) -> JGain:
    """Return s and (I - uu.T) grad_h(0.5 * ||J(h)u||^2), with u fixed.

    This implements a one-dimensional probe subspace. `gain` is unnormalized,
    detached, on the source device, and at least float32. Zero gain is a valid
    outcome (e.g. an affine map); callers must not normalize it blindly.
    The matrix-free reverse-over-reverse JVP keeps its graph for the second
    derivative. No model parameter gradients are accumulated.
    """
    with torch.enable_grad():
        state, axis = _routing_inputs(hidden, direction)
        propagated = _directional_jvp(final_hidden, state, axis, create_graph=True)
        energy = 0.5 * propagated.square().sum()
        if energy.requires_grad:
            (gradient,) = torch.autograd.grad(
                energy, state, allow_unused=True, materialize_grads=True
            )
        else:
            # An affine map has a constant Jacobian and exactly zero J-gain.
            gradient = torch.zeros_like(state)
        gradient = gradient.to(dtype=axis.dtype)
        gain = gradient - axis * (axis @ gradient)
        if not torch.isfinite(gain).all():
            raise ValueError("Non-finite J-gain")
    return JGain(float(propagated.norm().detach()), gain.detach())
