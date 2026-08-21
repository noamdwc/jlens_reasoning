# Chat Inference Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared Hugging Face chat-inference module with explicit direct and native-reasoning modes, then migrate FLenQA accuracy generation to the valid direct Qwen chat path.

**Architecture:** Create one functional `jlens_reasoning.inference` module with immutable configuration/result dataclasses and one `generate_chat()` entry point. It applies the tokenizer's chat template, selects mode-specific decoding, preserves `ModelOutput`, and separates Qwen reasoning from the final answer. The FLenQA notebook remains responsible for dataset iteration, paper weighting, evaluation, persistence, and plots.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers, dataclasses, pytest, nbformat, PyArrow, Ruff, uv.

---

## File map

- Create `src/jlens_reasoning/inference.py`: inference profiles, errors,
  structured result, chat wrapping, generation, and Qwen response splitting.
- Create `tests/test_inference.py`: CPU-only fake-model coverage for both modes,
  validation, completion state, response parsing, and failures.
- Modify `notebooks/flenqa_accuracy.ipynb`: replace local raw generation with the
  shared direct chat profile and persist inference audit fields.
- Modify `tests/test_notebooks.py`: enforce the shared inference boundary and
  reject direct generation inside the FLenQA accuracy notebook.
- Modify `README.md`: document corrected direct chat inference and rerun
  expectations.
- Modify `docs/llm-answer-evaluation.md`: distinguish inference mode from
  task-specific answer evaluation.

The Jacobian Lens sanity notebook is intentionally unchanged because its token
positions refer to the raw prompt and require a separate alignment design.

---

### Task 1: Define immutable inference profiles

**Files:**
- Create: `src/jlens_reasoning/inference.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_inference.py` with:

```python
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
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
uv run pytest tests/test_inference.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named
'jlens_reasoning.inference'`.

- [ ] **Step 3: Implement the profiles and focused errors**

Create `src/jlens_reasoning/inference.py` with:

```python
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
```

- [ ] **Step 4: Run configuration tests**

Run:

```bash
uv run pytest tests/test_inference.py -q
uv run ruff check src/jlens_reasoning/inference.py tests/test_inference.py
```

Expected: all configuration tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit the profiles**

```bash
git add src/jlens_reasoning/inference.py tests/test_inference.py
git commit -m "feat: define chat inference profiles"
```

---

### Task 2: Generate and split chat responses

**Files:**
- Modify: `src/jlens_reasoning/inference.py`
- Modify: `tests/test_inference.py`

- [ ] **Step 1: Add fake tokenizer/model fixtures and failing direct-mode tests**

Replace the import block at the top of `tests/test_inference.py` with the
following combined block, then append the fake objects and tests shown below:

```python
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
```

```python


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
```

- [ ] **Step 2: Add failing reasoning, truncation, and failure tests**

Append:

```python
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
```

- [ ] **Step 3: Run the new tests and confirm they fail for missing generation API**

Run:

```bash
uv run pytest tests/test_inference.py -q
```

Expected: configuration tests pass; generation tests fail because
`InferenceResult` and `generate_chat` do not exist.

- [ ] **Step 4: Implement structured results and response splitting**

Append the following imports near the top of `src/jlens_reasoning/inference.py`:

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .evaluation import GenerationStatus, ModelOutput
from .evaluation_utils import ReasoningStatus
```

Keep the existing single `dataclass` import rather than duplicating it. Add:

```python
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
```

- [ ] **Step 5: Implement chat wrapping and generation**

Append:

```python
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
```

- [ ] **Step 6: Format and run inference tests**

Run:

```bash
uv run ruff format src/jlens_reasoning/inference.py tests/test_inference.py
uv run ruff check src/jlens_reasoning/inference.py tests/test_inference.py
uv run pytest tests/test_inference.py -q
```

Expected: formatting and lint pass; all inference tests pass.

- [ ] **Step 7: Commit chat generation**

```bash
git add src/jlens_reasoning/inference.py tests/test_inference.py
git commit -m "feat: generate structured chat responses"
```

---

### Task 3: Migrate the FLenQA accuracy notebook

**Files:**
- Modify: `tests/test_notebooks.py`
- Modify: `notebooks/flenqa_accuracy.ipynb`

- [ ] **Step 1: Replace the old notebook contract with a failing shared-inference contract**

In `test_flenqa_accuracy_notebook_has_visible_full_run_workflow`, replace the
old generation assertions:

```python
assert "MAX_SEQ_LEN = 4096" in source
assert "MAX_NEW_TOKENS = 64" in source
assert "do_sample=False" in cells["define-generation"]
```

with:

```python
assert "from jlens_reasoning.inference import" in source
assert "InferenceConfig.direct(" in source
assert "max_input_tokens=4096" in source
assert "generate_chat(" in cells["run-accuracy"]
assert "causal_lm.generate(" not in source
assert "generated_text" in cells["run-accuracy"]
assert "reasoning_text" in cells["run-accuracy"]
assert "answer_text" in cells["run-accuracy"]
assert "reasoning_status" in cells["run-accuracy"]
assert "inference_mode" in cells["run-accuracy"]
assert "max_new_tokens" in cells["run-accuracy"]
```

Keep the existing assertions for the 9,862-prompt loop, paper weighting,
single Parquet output, plots, and forbidden lens workflow.

- [ ] **Step 2: Run the notebook contract and verify it fails**

Run:

```bash
uv run pytest tests/test_notebooks.py::test_flenqa_accuracy_notebook_has_visible_full_run_workflow -q
```

Expected: FAIL because the notebook still defines local raw generation.

- [ ] **Step 3: Replace notebook imports and settings**

In the `imports-settings` cell:

- replace the multiline evaluation import with
  `from jlens_reasoning.evaluation import evaluate_paper_binary`;
- add:

```python
from jlens_reasoning.inference import InferenceConfig, generate_chat
```

- remove `MAX_SEQ_LEN` and `MAX_NEW_TOKENS`;
- add:

```python
INFERENCE_CONFIG = InferenceConfig.direct(max_input_tokens=4096)
```

Delete the entire `define-generation` cell. Do not retain a notebook-local
wrapper around `generate_chat()`.

- [ ] **Step 4: Call shared inference and record its audit fields**

In `run-accuracy`, replace:

```python
n_input_tokens, output = generate_output(prompt.text)
evaluation = evaluate_paper_binary(output, expected=prompt.label)
```

with:

```python
inference = generate_chat(
    causal_lm,
    tokenizer,
    prompt.text,
    config=INFERENCE_CONFIG,
)
evaluation = evaluate_paper_binary(inference.output, expected=prompt.label)
```

Replace and extend the generation-related result fields with:

```python
"n_input_tokens": inference.input_token_count,
"paper_weight": sum(
    item.dispersion == "random" for item in prompt.provenance
),
"model_name": MODEL_NAME,
"code_revision": PROJECT_COMMIT,
"inference_mode": inference.config.mode.value,
"max_new_tokens": inference.config.max_new_tokens,
"do_sample": inference.config.do_sample,
"temperature": inference.config.temperature,
"top_p": inference.config.top_p,
"top_k": inference.config.top_k,
"min_p": inference.config.min_p,
"generated_token_ids": list(inference.output.token_ids),
"generated_token_pieces": list(inference.output.token_pieces),
"generated_text": inference.raw_text,
"reasoning_text": inference.reasoning_text,
"answer_text": inference.answer_text,
"reasoning_status": inference.reasoning_status.value,
"generation_status": inference.output.generation_status.value,
"finish_reason": inference.output.finish_reason,
```

Keep verdict and correctness fields unchanged.

- [ ] **Step 5: Extend the Arrow schema**

In `save-results`, insert these fields after `code_revision`:

```python
pa.field("inference_mode", pa.string(), nullable=False),
pa.field("max_new_tokens", pa.int32(), nullable=False),
pa.field("do_sample", pa.bool_(), nullable=False),
pa.field("temperature", pa.float32()),
pa.field("top_p", pa.float32()),
pa.field("top_k", pa.int32()),
pa.field("min_p", pa.float32()),
```

Insert these fields after `generated_text`:

```python
pa.field("reasoning_text", pa.string()),
pa.field("answer_text", pa.string()),
pa.field("reasoning_status", pa.string(), nullable=False),
```

The nullable sampling fields are expected to be null in direct mode.

- [ ] **Step 6: Format and validate the notebook**

Run:

```bash
uv run ruff format notebooks/flenqa_accuracy.ipynb tests/test_notebooks.py
uv run pytest tests/test_notebooks.py -q
jq empty notebooks/flenqa_accuracy.ipynb
.venv/bin/python -c 'import ast, nbformat; notebook = nbformat.read("notebooks/flenqa_accuracy.ipynb", as_version=4); [ast.parse("".join(cell.source)) for cell in notebook.cells if cell.cell_type == "code" and cell.id != "colab-loader"]; print("notebook cells compile")'
```

Expected: all notebook tests pass, `jq` exits zero, and the compile command
prints `notebook cells compile`.

- [ ] **Step 7: Commit notebook migration**

```bash
git add notebooks/flenqa_accuracy.ipynb tests/test_notebooks.py
git commit -m "fix: use chat inference for FLenQA accuracy"
```

---

### Task 4: Document and verify the corrected inference path

**Files:**
- Modify: `README.md`
- Modify: `docs/llm-answer-evaluation.md`

- [ ] **Step 1: Update the README run description**

In `README.md` under `FLenQA accuracy by prompt length`, replace the sentence
describing greedy 64-token generation with:

```markdown
Generation uses the shared Hugging Face chat-inference module. The
paper-compatible curve runs Qwen in direct mode with its native chat template,
thinking explicitly disabled, deterministic decoding, and the paper wrapper's
400-token completion allowance. The saved table records the effective inference
mode and decoding settings alongside raw output, structured reasoning/answer
fields, exact wrapped input length, parsed verdict, and correctness.
```

Add a warning immediately after it:

```markdown
Results produced by the earlier raw-prompt, 64-token notebook are not comparable
and should be regenerated rather than appended to the corrected result table.
```

- [ ] **Step 2: Document the inference/evaluation boundary**

Add this section before `FLenQA Paper-Compatible Generated Verdict` in
`docs/llm-answer-evaluation.md`:

```markdown
## Chat Inference Boundary

Inference and grading are separate contracts. Instruction-tuned Hugging Face
models are invoked through `jlens_reasoning.inference.generate_chat`, never by
tokenizing a user prompt as raw completion text. `InferenceConfig.direct()`
disables native thinking and uses deterministic direct-answer generation;
`InferenceConfig.reasoning(max_new_tokens=budget)` enables the model's native thinking template
and requires an explicit shared completion budget.

`InferenceResult` preserves raw generated text and exposes reasoning and final
answer text separately. Evaluators consume the appropriate visible answer or
`ModelOutput`; they do not choose chat templates, sampling settings, or token
budgets. A response truncated before its reasoning boundary has no clean final
answer and must remain distinguishable from an incorrect answer.
```

- [ ] **Step 3: Run focused regression checks**

Run:

```bash
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run pytest tests/test_inference.py tests/test_evaluation.py tests/test_notebooks.py -q
git diff --check
```

Expected: format, lint, and focused tests pass; `git diff --check` prints
nothing.

- [ ] **Step 4: Run full repository verification**

Run:

```bash
uv run pytest
uv lock --check
```

Expected: the complete CPU suite passes and uv reports that the lockfile is
current. If sandboxed uv cannot open its shared cache, rerun only
`uv lock --check` with the required cache permission; do not change the lockfile.

- [ ] **Step 5: Review final scope and artifact hygiene**

Run:

```bash
git status --short
git diff --stat feature/adding-flenqa-asset
rg -n "causal_lm\.generate\(|MAX_NEW_TOKENS = 64|generate_output\(" notebooks/flenqa_accuracy.ipynb
```

Expected:

- only the planned module, tests, notebook, and documentation are modified;
- the final branch diff contains no FLenQA-local generation function;
- the `rg` command returns no matches;
- no notebook outputs, execution counts, credentials, Parquet results, or
  Colab artifacts are staged.

- [ ] **Step 6: Commit documentation and verification state**

```bash
git add README.md docs/llm-answer-evaluation.md
git commit -m "docs: describe chat inference modes"
```

- [ ] **Step 7: Finish the development branch**

Invoke `superpowers:verification-before-completion`, rerun any verification it
requires against the final committed tree, then invoke
`superpowers:finishing-a-development-branch` and offer merge, PR, keep, or
discard options relative to `feature/adding-flenqa-asset`.
