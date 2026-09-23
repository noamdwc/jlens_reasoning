from __future__ import annotations

import pytest
import torch
from jlens import JacobianLens

from jlens_reasoning.experiments_utils.interventions import jlens_vector
from jlens_reasoning.inference import (
    InferenceConfig,
    generate_chat,
    prepare_chat_inputs,
)
from jlens_reasoning.probe_jlens import probe_sensitivities, static_probe_projection
from jlens_reasoning.probing import ProbeConfig, extract_probe_features, token_margin


def test_projection_matches_transport_and_existing_jlens_vectors():
    # Nonsymmetric J catches an accidental transpose. ||w|| = 5.
    jacobian = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    unembedding = torch.tensor([[2.0, -1.0], [0.0, 3.0], [-1.0, 0.0]])
    lens = JacobianLens(jacobians={0: jacobian}, n_prompts=1, d_model=2)
    projected, scores = static_probe_projection(
        jacobian, torch.tensor([3.0, 4.0]), unembedding
    )
    torch.testing.assert_close(projected, torch.tensor([2.2, 5.0]))
    torch.testing.assert_close(scores, torch.tensor([-0.6, 15.0, -2.2]))
    direction = torch.tensor([0.6, 0.8])
    torch.testing.assert_close(projected, lens.transport(direction, 0))
    for token in range(3):
        pulled_back = jlens_vector(lens, unembedding, layer=0, token_id=token)
        torch.testing.assert_close(scores[token], pulled_back @ direction)
    # The same sign on gold output margin and probe direction cancels for False.
    for gold_sign in (-1, 1):
        h = torch.zeros(2, requires_grad=True)
        margin = gold_sign * ((unembedding[0] - unembedding[1]) @ jacobian @ h)
        (grad,) = torch.autograd.grad(margin, h)
        torch.testing.assert_close(
            grad @ (gold_sign * direction), scores[0] - scores[1]
        )


def test_zero_probe_direction_is_rejected():
    with pytest.raises(ValueError):
        static_probe_projection(torch.eye(2), torch.zeros(2), torch.eye(2))


def test_sensitivity_matches_direct_autograd_and_removes_hooks_on_error(
    model, tokenizer
):
    config = ProbeConfig(InferenceConfig.direct(max_new_tokens=1))
    weight = torch.arange(1, 9).float() / 10
    probes = {
        layer: {
            "weight": weight,
            "bias": torch.tensor(0.3),
            "training_mean": torch.zeros(8),
        }
        for layer in range(2)
    }
    encoded = prepare_chat_inputs(tokenizer, "cat", config=config.inference)
    outputs = model(**encoded, output_hidden_states=True)
    margin = outputs.logits[0, -1, 5] - outputs.logits[0, -1, 6]
    gradients = torch.autograd.grad(margin, outputs.hidden_states[1:])
    expected = [float(g[0, -1] @ (weight / weight.norm())) for g in gradients]
    generated = generate_chat(model, tokenizer, "cat", config=config.inference)
    features = extract_probe_features(model, tokenizer, "cat", config=config)
    records = {
        layer: {
            "input_sha256": generated.input_sha256,
            "n_input_tokens": 5,
            "probe_score": float(state @ weight + 0.3),
            "output_margin": float(margin.detach()),
        }
        for layer, state in enumerate(features.states)
    }
    hooks = [dict(layer._forward_hooks) for layer in model.model.layers]
    model.requires_grad_(False)
    result = probe_sensitivities(
        model,
        tokenizer,
        "cat",
        probes,
        config=config,
        blocks=model.model.layers,
        objective=lambda logits: token_margin(logits, (5,), (6,)),
        saved_records=records,
    )
    assert [r.sensitivity for r in result] == pytest.approx(expected, abs=1e-7)
    records[1]["probe_score"] += 10
    with pytest.raises(ValueError, match="score"):
        probe_sensitivities(
            model,
            tokenizer,
            "cat",
            probes,
            config=config,
            blocks=model.model.layers,
            saved_records=records,
            objective=lambda logits: token_margin(logits, (5,), (6,)),
        )

    def failing_objective(logits):
        raise RuntimeError("objective failed")

    with pytest.raises(RuntimeError, match="objective failed"):
        probe_sensitivities(
            model,
            tokenizer,
            "cat",
            probes,
            config=config,
            blocks=model.model.layers,
            objective=failing_objective,
        )
    assert [dict(layer._forward_hooks) for layer in model.model.layers] == hooks


def test_saved_margin_keeps_additive_tolerance_for_existing_exports(model, tokenizer):
    config = ProbeConfig(InferenceConfig.direct(max_new_tokens=1))
    features = extract_probe_features(model, tokenizer, "cat", config=config)
    probes = {
        layer: {
            "weight": torch.ones(8),
            "bias": torch.tensor(0.0),
            "training_mean": torch.zeros(8),
        }
        for layer in range(2)
    }
    records = {
        layer: {
            **features.input_record,
            "probe_score": float(state.sum()),
            "output_margin": 2.0,
        }
        for layer, state in enumerate(features.states)
    }
    model.requires_grad_(False)
    result = probe_sensitivities(
        model,
        tokenizer,
        "cat",
        probes,
        config=config,
        blocks=model.model.layers,
        objective=lambda logits: logits.sum() * 0 + 2.06,
        saved_records=records,
    )
    # Existing exports allow .05 + .02 * abs(saved): .06 < .09.
    assert all(item.output_margin == pytest.approx(2.06) for item in result)
    records[0]["output_margin"] = 1.9
    with pytest.raises(ValueError, match="output_margin"):
        probe_sensitivities(
            model,
            tokenizer,
            "cat",
            probes,
            config=config,
            blocks=model.model.layers,
            objective=lambda logits: logits.sum() * 0 + 2.06,
            saved_records=records,
        )
