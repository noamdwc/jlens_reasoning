from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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
            raise InferenceConfigurationError(f"Unsupported inference mode: {self.mode}")

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
