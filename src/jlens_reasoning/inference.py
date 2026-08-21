from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch

from .evaluation import GenerationStatus, ModelOutput
from .evaluation_utils import ReasoningStatus


class InferenceConfigurationError(ValueError):
    """Raised when a chat-inference profile or template is invalid."""


class InferenceInputError(ValueError):
    """Raised when a prompt cannot be supplied to the configured model."""


class InferenceGenerationError(RuntimeError):
    """Raised when model generation fails operationally."""


class InferenceMode(StrEnum):
    DIRECT = "direct"
    REASONING = "reasoning"


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    mode: InferenceMode
    max_new_tokens: int
    max_input_tokens: int | None
    do_sample: bool
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None

    def __post_init__(self) -> None:
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise InferenceConfigurationError(
                "max_new_tokens must be a positive integer"
            )
        if self.max_input_tokens is not None and (
            type(self.max_input_tokens) is not int or self.max_input_tokens <= 0
        ):
            raise InferenceConfigurationError(
                "max_input_tokens must be a positive integer or None"
            )
        sampling_values = (self.temperature, self.top_p, self.top_k, self.min_p)
        if self.mode is InferenceMode.DIRECT:
            if self.do_sample or any(value is not None for value in sampling_values):
                raise InferenceConfigurationError(
                    "direct inference must use deterministic decoding"
                )
        elif self.mode is InferenceMode.REASONING:
            if not self.do_sample or any(value is None for value in sampling_values):
                raise InferenceConfigurationError(
                    "reasoning inference requires complete sampling settings"
                )
            if self.temperature is None or self.temperature <= 0:
                raise InferenceConfigurationError("temperature must be positive")
            if self.top_p is None or not 0 < self.top_p <= 1:
                raise InferenceConfigurationError("top_p must be in (0, 1]")
            if self.top_k is None or self.top_k < 0:
                raise InferenceConfigurationError("top_k must be non-negative")
            if self.min_p is None or not 0 <= self.min_p <= 1:
                raise InferenceConfigurationError("min_p must be in [0, 1]")
        else:
            raise InferenceConfigurationError(
                f"Unsupported inference mode: {self.mode}"
            )

    @classmethod
    def direct(
        cls,
        *,
        max_new_tokens: int = 400,
        max_input_tokens: int | None = None,
    ) -> InferenceConfig:
        return cls(
            mode=InferenceMode.DIRECT,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            do_sample=False,
        )

    @classmethod
    def reasoning(
        cls,
        *,
        max_new_tokens: int,
        max_input_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
    ) -> InferenceConfig:
        return cls(
            mode=InferenceMode.REASONING,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

    def generation_kwargs(self) -> dict[str, bool | int | float]:
        values: dict[str, bool | int | float] = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.do_sample:
            assert self.temperature is not None
            assert self.top_p is not None
            assert self.top_k is not None
            assert self.min_p is not None
            values.update(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
            )
        return values


@dataclass(frozen=True, slots=True)
class InferenceResult:
    output: ModelOutput
    reasoning_text: str | None
    answer_text: str | None
    reasoning_status: ReasoningStatus
    input_token_count: int
    generated_token_count: int
    config: InferenceConfig

    def __post_init__(self) -> None:
        if self.input_token_count <= 0:
            raise ValueError("input_token_count must be positive")
        if self.generated_token_count != len(self.output.token_ids):
            raise ValueError("generated token count must match ModelOutput")

    @property
    def raw_text(self) -> str:
        return self.output.text


def _split_response(
    text: str,
    *,
    mode: InferenceMode,
) -> tuple[str | None, str | None, ReasoningStatus]:
    opening = "<think>"
    closing = "</think>"
    opening_count = text.count(opening)
    closing_count = text.count(closing)

    if mode is InferenceMode.DIRECT:
        if opening_count or closing_count:
            return None, None, ReasoningStatus.MALFORMED
        return None, text.strip() or None, ReasoningStatus.NOT_PRESENT

    if opening_count == 0 and closing_count == 1:
        reasoning, answer = text.split(closing, 1)
        return reasoning.strip(), answer.strip() or None, ReasoningStatus.PARSED

    if opening_count == 1 and closing_count == 1:
        prefix, tagged = text.split(opening, 1)
        if not prefix.strip():
            reasoning, answer = tagged.split(closing, 1)
            return reasoning.strip(), answer.strip() or None, ReasoningStatus.PARSED

    if closing_count == 0:
        partial = text
        if opening_count == 1:
            prefix, partial = text.split(opening, 1)
            if prefix.strip():
                return None, None, ReasoningStatus.MALFORMED
        elif opening_count > 1:
            return None, None, ReasoningStatus.MALFORMED
        return partial.strip() or None, None, ReasoningStatus.MALFORMED

    return None, None, ReasoningStatus.MALFORMED


def _eos_token_ids(value: int | Sequence[int] | None) -> set[int]:
    if isinstance(value, int):
        return {value}
    return {int(token_id) for token_id in value or ()}


def generate_chat(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    config: InferenceConfig,
) -> InferenceResult:
    if not isinstance(prompt, str) or not prompt.strip():
        raise InferenceInputError("prompt must be non-empty text")
    if not getattr(tokenizer, "chat_template", None):
        raise InferenceConfigurationError("tokenizer must define a chat template")

    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=config.mode is InferenceMode.REASONING,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as error:
        raise InferenceConfigurationError("chat template application failed") from error
    if not isinstance(encoded, Mapping) or not {
        "input_ids",
        "attention_mask",
    }.issubset(encoded):
        raise InferenceConfigurationError(
            "chat template must return input_ids and attention_mask"
        )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise InferenceInputError("chat inference requires exactly one input sequence")
    input_token_count = int(input_ids.shape[1])
    if (
        config.max_input_tokens is not None
        and input_token_count > config.max_input_tokens
    ):
        raise InferenceInputError(
            f"wrapped input has {input_token_count} tokens and exceeds configured "
            f"limit {config.max_input_tokens}"
        )

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    try:
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **config.generation_kwargs(),
            )
    except Exception as error:
        raise InferenceGenerationError("chat generation failed") from error

    generated_ids = generated[0, input_token_count:].tolist()
    eos_ids = _eos_token_ids(model.generation_config.eos_token_id)
    complete = bool(generated_ids and generated_ids[-1] in eos_ids)
    text_ids = generated_ids[:-1] if complete else generated_ids
    output = ModelOutput(
        text=tokenizer.decode(text_ids, skip_special_tokens=True),
        token_ids=tuple(generated_ids),
        token_pieces=tuple(
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            for token_id in generated_ids
        ),
        generation_status=(
            GenerationStatus.COMPLETE if complete else GenerationStatus.TRUNCATED
        ),
        finish_reason="eos" if complete else "length",
    )
    reasoning_text, answer_text, reasoning_status = _split_response(
        output.text,
        mode=config.mode,
    )
    return InferenceResult(
        output=output,
        reasoning_text=reasoning_text,
        answer_text=answer_text,
        reasoning_status=reasoning_status,
        input_token_count=input_token_count,
        generated_token_count=len(generated_ids),
        config=config,
    )
