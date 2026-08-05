from dataclasses import FrozenInstanceError

import pytest

from jlens_reasoning.inference import (
    InferenceConfig,
    InferenceConfigurationError,
    InferenceMode,
)


def test_direct_profile_matches_paper_style_generation() -> None:
    config = InferenceConfig.direct(max_input_tokens=4096)

    assert config.mode is InferenceMode.DIRECT
    assert config.max_new_tokens == 400
    assert config.max_input_tokens == 4096
    assert config.do_sample is False
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None
    assert config.min_p is None
    assert config.generation_kwargs() == {
        "do_sample": False,
        "max_new_tokens": 400,
    }


def test_reasoning_profile_uses_qwen_sampling_defaults() -> None:
    config = InferenceConfig.reasoning(
        max_new_tokens=2048,
        max_input_tokens=4096,
    )

    assert config.mode is InferenceMode.REASONING
    assert config.max_new_tokens == 2048
    assert config.max_input_tokens == 4096
    assert config.do_sample is True
    assert config.temperature == 1.0
    assert config.top_p == 0.95
    assert config.top_k == 20
    assert config.min_p == 0.0
    assert config.generation_kwargs() == {
        "do_sample": True,
        "max_new_tokens": 2048,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    }


@pytest.mark.parametrize("max_new_tokens", [0, -1, True])
def test_profiles_reject_invalid_completion_budgets(max_new_tokens: object) -> None:
    with pytest.raises(
        InferenceConfigurationError,
        match="max_new_tokens must be a positive integer",
    ):
        InferenceConfig.direct(max_new_tokens=max_new_tokens)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_input_tokens", [0, -1, True])
def test_profiles_reject_invalid_input_limits(max_input_tokens: object) -> None:
    with pytest.raises(
        InferenceConfigurationError,
        match="max_input_tokens must be a positive integer or None",
    ):
        InferenceConfig.reasoning(
            max_new_tokens=32,
            max_input_tokens=max_input_tokens,  # type: ignore[arg-type]
        )


def test_inference_profiles_are_immutable() -> None:
    config = InferenceConfig.direct()

    with pytest.raises(FrozenInstanceError):
        config.max_new_tokens = 10  # type: ignore[misc]
