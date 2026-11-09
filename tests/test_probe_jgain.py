import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from jlens_reasoning.inference import InferenceConfig, prepare_chat_inputs
from jlens_reasoning.probe_jlens import (
    j_gain,
    j_sensitivity,
    probe_component,
    prompt_hidden_map,
)
from jlens_reasoning.probing import ProbeConfig, score_probe


def nonlinear_map(h):
    return torch.stack((h[0] * h[1].exp(), h[2].square()))


def test_gain_has_analytic_value_and_preserves_probe_score():
    h = torch.tensor([0.7, 0.2, -0.5], dtype=torch.float64)
    axis = torch.tensor([1.0, 0.0, 0.0], dtype=h.dtype)
    result = j_gain(nonlinear_map, h, axis)
    expected = torch.tensor([0.0, (2 * h[1]).exp(), 0.0], dtype=h.dtype)
    torch.testing.assert_close(result.gain, expected)
    assert result.sensitivity == pytest.approx(h[1].exp().item())
    assert not result.gain.requires_grad
    probe = {
        "weight": axis.float(),
        "bias": torch.tensor(0.3),
        "training_mean": torch.zeros(3),
    }
    plus, minus = h + 1e-3 * result.gain, h - 1e-3 * result.gain
    assert j_sensitivity(nonlinear_map, plus, axis) > result.sensitivity
    assert j_sensitivity(nonlinear_map, minus, axis) < result.sensitivity
    torch.testing.assert_close(score_probe(probe, plus), score_probe(probe, h))
    torch.testing.assert_close(score_probe(probe, minus), score_probe(probe, h))
    assert float(axis @ result.gain) == 0


def test_gain_matches_finite_differences_with_clean_direction_frozen():
    h = torch.tensor([0.7, 0.2, -0.5], dtype=torch.float64, requires_grad=True)
    direction = h.square()  # Deliberately connected: the utility must detach it.
    fixed = direction.detach() / direction.detach().norm()
    result = j_gain(nonlinear_map, h, direction)
    eps = 1e-5
    basis = torch.eye(3, dtype=h.dtype)
    gradient = torch.tensor(
        [
            (
                j_sensitivity(nonlinear_map, h + eps * e, fixed) ** 2
                - j_sensitivity(nonlinear_map, h - eps * e, fixed) ** 2
            )
            / (4 * eps)
            for e in basis
        ],
        dtype=h.dtype,
    )
    expected = gradient - fixed * (fixed @ gradient)
    torch.testing.assert_close(result.gain, expected, atol=1e-8, rtol=1e-7)
    finite_jvp = (nonlinear_map(h + eps * fixed) - nonlinear_map(h - eps * fixed)) / (
        2 * eps
    )
    assert result.sensitivity == pytest.approx(finite_jvp.norm().item(), rel=1e-8)
    assert h.grad is None
    reverse = j_gain(nonlinear_map, h, -direction)
    torch.testing.assert_close(result.gain, reverse.gain)
    assert reverse.sensitivity == pytest.approx(result.sensitivity)


@pytest.mark.parametrize("constant", [False, True])
def test_affine_or_constant_map_has_zero_gain(constant):
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    if constant:
        matrix.zero_()
    result = j_gain(
        lambda h: matrix @ h + 3,
        torch.ones(2, dtype=matrix.dtype),
        torch.tensor([1.0, 0.0], dtype=matrix.dtype),
    )
    torch.testing.assert_close(result.gain, torch.zeros(2, dtype=matrix.dtype))
    assert result.sensitivity == pytest.approx(matrix[:, 0].norm().item())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_centered_component_keeps_sign_and_excludes_bias(dtype):
    probe = {
        "weight": torch.tensor([2.0, 0.0]),
        "training_mean": torch.tensor([1.0, 3.0]),
        "bias": torch.tensor(20.0),
    }
    torch.testing.assert_close(
        probe_component(probe, torch.tensor([-2.0, 5.0], dtype=dtype)),
        torch.tensor([-3.0, 0.0], dtype=torch.promote_types(dtype, torch.float32)),
    )
    zero = probe_component(probe, torch.tensor([1.0, 5.0]))
    assert zero.norm() == 0
    with pytest.raises(ValueError, match="nonzero"):
        j_gain(lambda h: h.square(), torch.ones(2), zero)


@pytest.mark.parametrize(
    "direction", [torch.ones(3), torch.zeros(2), torch.tensor([float("nan"), 1.0])]
)
def test_invalid_direction_is_rejected(direction):
    with pytest.raises(ValueError):
        j_sensitivity(lambda h: h, torch.ones(2), direction)


def test_inference_mode_cannot_silently_zero_derivatives():
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        j_gain(nonlinear_map, torch.ones(3), torch.ones(3))


def test_prompt_map_matches_normalized_target_and_finite_differences(model, tokenizer):
    model.double().requires_grad_(False)
    config = ProbeConfig(InferenceConfig.direct(max_new_tokens=1))
    blocks = model.model.layers
    forwards = [block.forward for block in blocks]
    encoded = prepare_chat_inputs(tokenizer, "cat", config=config.inference)
    with torch.no_grad():
        clean = model(**encoded, output_hidden_states=True, use_cache=False)
    # Transformers lazily installs its own persistent output-capture hooks.
    hooks = [dict(block._forward_hooks) for block in blocks]
    mapping = prompt_hidden_map(
        model, tokenizer, "cat", layer=0, blocks=blocks, config=config
    )
    torch.testing.assert_close(mapping.hidden, clean.hidden_states[1][0, -1])
    torch.testing.assert_close(
        mapping.final_hidden(mapping.hidden), clean.hidden_states[-1][0, -1]
    )
    axis = torch.arange(1, 9, dtype=mapping.hidden.dtype)
    axis /= axis.norm()
    result = j_gain(mapping.final_hidden, mapping.hidden, axis)
    # Llama RMSNorm casts to float32 even with double weights. A smaller step
    # measures rounding noise; 1e-4 is in the converged finite-difference range.
    eps = 1e-4
    difference = (
        mapping.final_hidden(mapping.hidden + eps * axis)
        - mapping.final_hidden(mapping.hidden - eps * axis)
    ) / (2 * eps)
    assert result.sensitivity == pytest.approx(difference.norm().item(), rel=1e-5)
    # This checks the actual model's second derivative, including final norm.
    random = torch.tensor([0.0, 1.0, -1.0, 0.2, -0.1, 0.7, 0.3, -0.5], dtype=axis.dtype)
    orthogonal = random - axis * (axis @ random)
    upper = j_sensitivity(mapping.final_hidden, mapping.hidden + eps * orthogonal, axis)
    lower = j_sensitivity(mapping.final_hidden, mapping.hidden - eps * orthogonal, axis)
    assert float(result.gain @ orthogonal) == pytest.approx(
        (upper**2 - lower**2) / (4 * eps), rel=2e-4, abs=1e-5
    )
    assert [dict(block._forward_hooks) for block in blocks] == hooks
    assert [block.forward for block in blocks] == forwards
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(ValueError, match="raw block-output"):
        prompt_hidden_map(
            model, tokenizer, "cat", layer=1, blocks=blocks, config=config
        )
    with pytest.raises(ValueError, match="input"):
        prompt_hidden_map(
            model,
            tokenizer,
            "cat",
            layer=0,
            blocks=blocks,
            config=config,
            input_record={"input_sha256": "wrong", "n_input_tokens": 5},
        )


def test_prompt_map_restores_hooks_and_forwards_on_failure(
    model, tokenizer, monkeypatch
):
    model.requires_grad_(False)
    blocks = model.model.layers
    mapping = prompt_hidden_map(
        model,
        tokenizer,
        "cat",
        layer=0,
        blocks=blocks,
        config=ProbeConfig(InferenceConfig.direct()),
    )
    hooks = [dict(block._forward_hooks) for block in blocks]

    def fail(*args, **kwargs):
        raise RuntimeError("downstream failure")

    monkeypatch.setattr(blocks[1], "forward", fail)
    with pytest.raises(RuntimeError, match="downstream failure"):
        mapping.final_hidden(mapping.hidden)
    assert blocks[1].forward is fail
    assert [dict(block._forward_hooks) for block in blocks] == hooks


@pytest.mark.parametrize("token_position", [-1, -2])
def test_cached_map_matches_full_qwen_derivatives(tokenizer, token_position):
    torch.manual_seed(3)
    # Include both attention kinds and cross the recurrent kernel's 64-token
    # chunk boundary. CUDA exercises actual transfers when it is available.
    model = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(
            vocab_size=12,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["full_attention", "linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
            attn_implementation="eager",
        )
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval().requires_grad_(
        False
    )
    mapping = prompt_hidden_map(
        model,
        tokenizer,
        "cat " * 70,
        layer=0,
        blocks=model.model.layers,
        config=ProbeConfig(InferenceConfig.direct(), token_position=token_position),
    )
    axis = torch.arange(1, 17, device=model.device).float()
    inputs = prepare_chat_inputs(
        tokenizer, "cat " * 70, config=InferenceConfig.direct()
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    def full_prompt(hidden):
        def replace_token(module, args, output):
            patched = output.detach().clone()
            patched[0, token_position] = hidden
            return patched

        handle = model.model.layers[0].register_forward_hook(replace_token)
        try:
            return model(
                **inputs, use_cache=False, output_hidden_states=True, logits_to_keep=1
            ).hidden_states[-1][0, -1]
        finally:
            handle.remove()

    cached = j_gain(mapping.final_hidden, mapping.hidden, axis)
    reference = j_gain(full_prompt, mapping.hidden, axis)
    torch.testing.assert_close(cached.gain, reference.gain, rtol=1e-4, atol=1e-4)
    assert cached.sensitivity == pytest.approx(reference.sensitivity, rel=1e-5)
    assert j_sensitivity(mapping.final_hidden, mapping.hidden, axis) == pytest.approx(
        reference.sensitivity, rel=1e-5
    )
    edited = (
        mapping.hidden
        + 0.001 * mapping.hidden.norm() * cached.gain / cached.gain.norm()
    )
    expected = full_prompt(edited)
    # Reusing the callable must not advance the cached prefix or alter the map.
    for _ in range(2):
        torch.testing.assert_close(
            mapping.final_hidden(edited), expected, rtol=1e-4, atol=1e-4
        )
    assert cached.gain.norm() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
