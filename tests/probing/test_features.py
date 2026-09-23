from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from jlens_reasoning.inference import (
    InferenceConfig,
    prepare_chat_inputs,
)
from jlens_reasoning.probing import (
    ProbeConfig,
    extract_probe_features,
    probe_input_contract,
    token_margin,
    validate_probe_input_contract,
)


def test_contract_rejects_changed_mode_and_position(tokenizer):
    config = ProbeConfig(InferenceConfig.direct(max_input_tokens=4096))
    original = probe_input_contract(tokenizer, config=config)
    assert original["input_format"] == "chat_template_direct"
    assert (
        original["feature_position"]
        == "final wrapped input token at index -1 before generation"
    )
    for changed in [
        replace(config, token_position=0),
        replace(config, inference=InferenceConfig.reasoning(max_new_tokens=1)),
    ]:
        with pytest.raises(ValueError):
            validate_probe_input_contract(
                original, probe_input_contract(tokenizer, config=changed)
            )


def test_extraction_uses_chat_features_and_selected_position(model, tokenizer):
    config = ProbeConfig(InferenceConfig.direct(max_new_tokens=1))
    encoded = prepare_chat_inputs(tokenizer, "cat", config=config.inference)
    with torch.no_grad():
        reference = model(**encoded, output_hidden_states=True)
    for position in [-1, 0]:
        features = extract_probe_features(
            model, tokenizer, "cat", config=replace(config, token_position=position)
        )
        for layer in range(2):
            torch.testing.assert_close(
                features.states[layer], reference.hidden_states[layer + 1][0, position]
            )
        torch.testing.assert_close(features.logits, reference.logits[0, -1])
        assert features.input_record["n_input_tokens"] == 5
    with pytest.raises(ValueError, match="position"):
        extract_probe_features(
            model, tokenizer, "cat", config=replace(config, token_position=5)
        )
    with pytest.raises(ValueError, match="Regenerate"):
        extract_probe_features(
            model,
            tokenizer,
            "cat",
            config=config,
            input_record={"n_input_tokens": 5, "input_sha256": "stale"},
        )


def test_token_margin_uses_group_means_and_rejects_overlap():
    assert float(token_margin(torch.tensor([1.0, 3.0, -2.0]), (0, 1), (2,))) == 4
    with pytest.raises(ValueError):
        token_margin(torch.tensor([1.0, 3.0]), (0,), (0,))
