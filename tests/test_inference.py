from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from jlens_reasoning.evaluation import GenerationStatus
from jlens_reasoning.evaluation_utils import ReasoningStatus
from jlens_reasoning.inference import (
    InferenceConfig,
    InferenceConfigurationError,
    InferenceGenerationError,
    InferenceInputError,
    InferenceMode,
    generate_chat,
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


class FakeTokenizer:
    chat_template = "qwen-chat-template"

    def __init__(self) -> None:
        self.template_calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []
        self.pieces = {
            2: "",
            19: "<think>",
            20: "reasoning",
            21: "</think>",
            22: "True",
        }

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        self.template_calls.append((messages, kwargs))
        return {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del kwargs
        return "".join(self.pieces[token_id] for token_id in token_ids)


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, generated_ids: list[int]) -> None:
        self.generated_ids = generated_ids
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.generate_kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_kwargs = kwargs
        suffix = torch.tensor([self.generated_ids], dtype=torch.long)
        return torch.cat((kwargs["input_ids"], suffix), dim=1)


def test_generate_chat_direct_uses_template_and_preserves_output() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel([22, 2])
    config = InferenceConfig.direct(max_input_tokens=4)

    result = generate_chat(model, tokenizer, "Question?", config=config)

    messages, template_kwargs = tokenizer.template_calls[0]
    assert messages == [{"role": "user", "content": "Question?"}]
    assert template_kwargs == {
        "add_generation_prompt": True,
        "enable_thinking": False,
        "return_dict": True,
        "return_tensors": "pt",
        "tokenize": True,
    }
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["attention_mask"].tolist() == [[1, 1]]
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 400
    assert "temperature" not in model.generate_kwargs
    assert result.raw_text == "True"
    assert result.reasoning_text is None
    assert result.answer_text == "True"
    assert result.reasoning_status is ReasoningStatus.NOT_PRESENT
    assert result.output.token_ids == (22, 2)
    assert result.output.token_pieces == ("True", "")
    assert result.output.generation_status is GenerationStatus.COMPLETE
    assert result.output.finish_reason == "eos"
    assert result.input_token_count == 2
    assert result.generated_token_count == 2
    assert result.config is config


def test_generate_chat_rejects_empty_and_overlong_inputs() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel([22, 2])

    with pytest.raises(InferenceInputError, match="prompt must be non-empty"):
        generate_chat(model, tokenizer, "   ", config=InferenceConfig.direct())

    with pytest.raises(InferenceInputError, match="exceeds configured limit 1"):
        generate_chat(
            model,
            tokenizer,
            "Question?",
            config=InferenceConfig.direct(max_input_tokens=1),
        )


def test_generate_chat_splits_native_qwen_reasoning() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel([20, 21, 22, 2])
    config = InferenceConfig.reasoning(max_new_tokens=2048)

    result = generate_chat(model, tokenizer, "Question?", config=config)

    _, template_kwargs = tokenizer.template_calls[0]
    assert template_kwargs["enable_thinking"] is True
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["do_sample"] is True
    assert model.generate_kwargs["temperature"] == 1.0
    assert model.generate_kwargs["top_p"] == 0.95
    assert model.generate_kwargs["top_k"] == 20
    assert model.generate_kwargs["min_p"] == 0.0
    assert result.raw_text == "reasoning</think>True"
    assert result.reasoning_text == "reasoning"
    assert result.answer_text == "True"
    assert result.reasoning_status is ReasoningStatus.PARSED


def test_generate_chat_accepts_explicit_complete_think_block() -> None:
    result = generate_chat(
        FakeModel([19, 20, 21, 22, 2]),
        FakeTokenizer(),
        "Question?",
        config=InferenceConfig.reasoning(max_new_tokens=128),
    )

    assert result.reasoning_text == "reasoning"
    assert result.answer_text == "True"
    assert result.reasoning_status is ReasoningStatus.PARSED


def test_generate_chat_preserves_valid_boundary_with_empty_answer() -> None:
    result = generate_chat(
        FakeModel([20, 21, 2]),
        FakeTokenizer(),
        "Question?",
        config=InferenceConfig.reasoning(max_new_tokens=128),
    )

    assert result.reasoning_text == "reasoning"
    assert result.answer_text is None
    assert result.reasoning_status is ReasoningStatus.PARSED


def test_generate_chat_preserves_truncated_reasoning_without_answer() -> None:
    result = generate_chat(
        FakeModel([20]),
        FakeTokenizer(),
        "Question?",
        config=InferenceConfig.reasoning(max_new_tokens=1),
    )

    assert result.raw_text == "reasoning"
    assert result.reasoning_text == "reasoning"
    assert result.answer_text is None
    assert result.reasoning_status is ReasoningStatus.MALFORMED
    assert result.output.generation_status is GenerationStatus.TRUNCATED
    assert result.output.finish_reason == "length"


@pytest.mark.parametrize(
    "generated_ids",
    [
        [19, 20, 21, 21, 22, 2],
        [19, 19, 20, 21, 22, 2],
    ],
)
def test_generate_chat_rejects_ambiguous_reasoning_boundaries(
    generated_ids: list[int],
) -> None:
    result = generate_chat(
        FakeModel(generated_ids),
        FakeTokenizer(),
        "Question?",
        config=InferenceConfig.reasoning(max_new_tokens=128),
    )

    assert result.reasoning_status is ReasoningStatus.MALFORMED
    assert result.answer_text is None


def test_direct_mode_does_not_accept_generated_think_blocks() -> None:
    result = generate_chat(
        FakeModel([19, 20, 21, 22, 2]),
        FakeTokenizer(),
        "Question?",
        config=InferenceConfig.direct(),
    )

    assert result.raw_text == "<think>reasoning</think>True"
    assert result.reasoning_status is ReasoningStatus.MALFORMED
    assert result.reasoning_text is None
    assert result.answer_text is None


def test_generate_chat_requires_a_chat_template() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.chat_template = None

    with pytest.raises(
        InferenceConfigurationError,
        match="tokenizer must define a chat template",
    ):
        generate_chat(
            FakeModel([22, 2]),
            tokenizer,
            "Question?",
            config=InferenceConfig.direct(),
        )
    assert tokenizer.template_calls == []


def test_generate_chat_wraps_chat_template_failures() -> None:
    class FailingTokenizer(FakeTokenizer):
        def apply_chat_template(
            self, messages: list[dict[str, str]], **kwargs: Any
        ) -> dict[str, torch.Tensor]:
            del messages, kwargs
            raise TypeError("unsupported template argument")

    with pytest.raises(
        InferenceConfigurationError,
        match="chat template application failed",
    ) as exc:
        generate_chat(
            FakeModel([22, 2]),
            FailingTokenizer(),
            "Question?",
            config=InferenceConfig.direct(),
        )

    assert isinstance(exc.value.__cause__, TypeError)


def test_generate_chat_wraps_model_failures() -> None:
    class FailingModel(FakeModel):
        def generate(self, **kwargs: Any) -> torch.Tensor:
            del kwargs
            raise RuntimeError("CUDA failed")

    with pytest.raises(InferenceGenerationError, match="chat generation failed") as exc:
        generate_chat(
            FailingModel([]),
            FakeTokenizer(),
            "Question?",
            config=InferenceConfig.direct(),
        )

    assert isinstance(exc.value.__cause__, RuntimeError)
