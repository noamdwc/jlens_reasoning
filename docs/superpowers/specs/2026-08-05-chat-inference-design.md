# Chat Inference Module

Date: 2026-08-05
Status: approved design

## Purpose

Add one shared Hugging Face chat-inference module that invokes instruction-tuned
models through their chat template and supports both direct answers and native
reasoning. Replace the FLenQA accuracy notebook's raw Qwen completion path with
this module so its results reflect a valid model interface.

The module owns prompt wrapping, decoding policy, generation, response
splitting, and inference metadata. Task-specific evaluation remains separate.

## Motivation

The current FLenQA accuracy notebook tokenizes the task prompt directly and
calls `model.generate()` with greedy decoding and a 64-token limit. That
bypasses Qwen3.5's chat template and does not configure its thinking mode. The
same small output budget must then contain any reasoning and the final verdict.
The completed run consequently produced no parseable verdict for 35.5% of
unique prompts and showed a clearly suspicious task-specific collapse.

The paper's reference implementation sent each prompt as a chat user message
and allowed 400 completion tokens. Qwen3.5 additionally requires its native
chat template to select thinking or non-thinking behavior.

## Goals

- Provide one small, tested inference path for Hugging Face chat models.
- Support explicit direct and reasoning modes.
- Prevent silent fallback to raw completion prompting.
- Preserve raw generated text while separating reasoning from the final answer.
- Make every effective inference setting auditable.
- Reuse the existing `ModelOutput`, `GenerationStatus`, and `ReasoningStatus`
  contracts.
- Migrate the FLenQA accuracy notebook to direct chat inference.

## Non-goals

- Hosted API or multi-provider abstractions.
- Separate reasoning and answer generation passes.
- Separate native reasoning and answer token budgets; Qwen's native generation
  uses one shared completion budget.
- Task-specific answer extraction, grading, or correctness.
- Constrained-logit scoring, best-of-N, majority voting, or verifier search.
- Migrating the Jacobian Lens sanity notebook. Its intervention and readout
  positions depend on exact raw-prompt token alignment and require a separate
  design review before chat-wrapper tokens are introduced.

## Architecture

Add `src/jlens_reasoning/inference.py` as a functional module. It contains
immutable configuration and result records plus one generation function. It
does not retain model state or define a provider class hierarchy.

### Public types

```python
class InferenceMode(StrEnum):
    DIRECT = "direct"
    REASONING = "reasoning"


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    mode: InferenceMode
    max_new_tokens: int
    max_input_tokens: int | None
    do_sample: bool
    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None

    @classmethod
    def direct(...) -> InferenceConfig: ...

    @classmethod
    def reasoning(*, max_new_tokens: int, ...) -> InferenceConfig: ...


@dataclass(frozen=True, slots=True)
class InferenceResult:
    output: ModelOutput
    reasoning_text: str | None
    answer_text: str | None
    reasoning_status: ReasoningStatus
    input_token_count: int
    generated_token_count: int
    config: InferenceConfig

    @property
    def raw_text(self) -> str: ...


def generate_chat(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    config: InferenceConfig,
) -> InferenceResult: ...
```

`raw_text` delegates to `output.text`; the result does not maintain two
independent copies of the same value.

### Configuration profiles

`InferenceConfig.direct()` selects the paper-style direct profile:

- `InferenceMode.DIRECT`;
- `enable_thinking=False` in the chat template;
- deterministic generation with `do_sample=False`;
- 400 new tokens by default;
- no sampling-only keyword arguments passed to `model.generate()`.

The caller may override the direct completion or input limit explicitly.

`InferenceConfig.reasoning(max_new_tokens=...)` selects Qwen-native reasoning:

- `InferenceMode.REASONING`;
- `enable_thinking=True` in the chat template;
- `do_sample=True`;
- Qwen3.5's recommended core sampling settings: temperature `1.0`, top-p
  `0.95`, top-k `20`, and min-p `0.0`;
- an explicit caller-selected shared completion budget.

Reasoning has no hidden default budget. Requiring the experiment to choose it
prevents an arbitrary research policy from being concealed inside the module.
All profile values remain immutable and are returned with each result.

## Generation flow

For each prompt, `generate_chat()` performs the following steps:

1. Validate the prompt and configuration.
2. Build exactly one message: `{"role": "user", "content": prompt}`.
3. Call `tokenizer.apply_chat_template()` with `tokenize=True`,
   `return_dict=True`, `return_tensors="pt"`,
   `add_generation_prompt=True`, and the mode's explicit `enable_thinking`
   value so both input IDs and the attention mask are available.
4. Refuse to continue when the tokenizer has no usable chat template. There is
   no raw-prompt fallback.
5. Tokenize without truncation and count the complete wrapped model input.
6. Enforce `max_input_tokens` when configured.
7. Move the encoded inputs to the model device and call `model.generate()`
   under `torch.inference_mode()`.
8. Slice away the input IDs and retain only newly generated IDs.
9. Classify EOS termination as complete and limit termination as truncated.
10. Decode the generated continuation, retaining raw text, token IDs, and
    per-token decoded pieces in `ModelOutput`.
11. Split reasoning and answer text according to the selected mode.

The exact input-token diagnostic measures the complete chat-wrapped input that
the model actually receives. The dataset's declared `ctx_size` remains the
paper-compatible plot axis.

## Response splitting

### Direct mode

Qwen's non-thinking chat template places an empty think block in the input
prefix, before generated tokens. The generated continuation should therefore
contain only the visible answer:

- `reasoning_text=None`;
- `answer_text=raw_text.strip()` when non-empty;
- `reasoning_status=ReasoningStatus.NOT_PRESENT`.

If the generated continuation unexpectedly contains a think tag, preserve the
raw text, report `MALFORMED`, and do not claim a clean answer.

### Reasoning mode

Qwen's thinking chat template pre-fills the opening `<think>` tag in the input.
The newly generated continuation normally contains reasoning followed by
`</think>` and then the visible answer. The splitter therefore supports both
forms without assuming the opening tag is part of the generated IDs:

- `reasoning text </think> answer text`;
- `<think>reasoning text</think> answer text`.

When one well-formed boundary is present:

- text before the closing tag is `reasoning_text`;
- text after it is stripped into `answer_text`;
- status is `PARSED`.

When generation ends before `</think>`:

- preserve the available continuation as partial `reasoning_text`;
- set `answer_text=None`;
- report `MALFORMED` reasoning and `TRUNCATED` generation.

Unexpected nested, repeated, or misplaced tags are malformed. Raw text is
always preserved even when structured fields cannot be trusted.

## Failure policy

Operational failures must never be converted into incorrect benchmark answers.
Define focused public exceptions:

- `InferenceConfigurationError` for invalid budgets, modes, or decoding
  combinations and unusable chat templates;
- `InferenceInputError` for empty prompts or wrapped inputs above the declared
  limit;
- `InferenceGenerationError` for model-generation failures, preserving the
  original exception as the cause.

Reaching the configured completion limit is not an exception. It returns a
normal `InferenceResult` with `GenerationStatus.TRUNCATED` and
`finish_reason="length"`.

Malformed model text is also data rather than an operational exception. The
result records `ReasoningStatus.MALFORMED`, preserves raw text, and exposes no
final answer when a reliable boundary is unavailable.

## FLenQA integration

Update `notebooks/flenqa_accuracy.ipynb` to:

- import `InferenceConfig` and `generate_chat`;
- create one `InferenceConfig.direct(max_input_tokens=4096)` profile;
- delete the notebook-local `generate_output()` implementation;
- call `generate_chat()` for each of the 9,862 unique prompts;
- evaluate `result.output` with `evaluate_paper_binary()`;
- save the inference mode, effective decoding fields, wrapped input-token
  count, raw text, reasoning text, answer text, generated-token metadata,
  generation status, finish reason, parsed verdict, and correctness.

The notebook remains notebook-first for looping, persistence, aggregation, and
plots. Inference mechanics alone move into shared package code.

The direct profile uses a 400-token allowance for fidelity to the paper's
reference wrapper. Reasoning support is tested in the module but is not used to
produce the paper-compatible headline curve.

## Testing

Add CPU-only tests using fake tokenizer and model objects. They must not load a
real model, access the network, or require CUDA.

Configuration tests cover:

- exact direct defaults;
- exact reasoning defaults and required explicit budget;
- positive budgets and valid input limits;
- rejection of sampling arguments in deterministic direct mode;
- immutable effective configuration.

Chat-wrapping tests cover:

- one user message with the unmodified prompt;
- `add_generation_prompt=True`;
- correct `enable_thinking` value in both modes;
- clear failure for a missing or unusable chat template;
- no raw-tokenization fallback.

Generation tests cover:

- attention-mask forwarding and device movement;
- mode-specific generation keyword arguments;
- correct slicing of input tokens;
- EOS-complete versus length-truncated results;
- token ID, token piece, and raw-text preservation;
- wrapped input and generated-token counts;
- generation exceptions wrapped as `InferenceGenerationError`.

Response-splitting tests cover:

- direct answers without reasoning;
- native Qwen continuation with only a closing think tag;
- a complete explicit `<think>...</think>` block;
- truncated reasoning without a closing tag;
- empty final answers;
- nested, repeated, and unexpected tags;
- preservation of partial reasoning and raw output.

Notebook contract tests assert that FLenQA imports and calls `generate_chat()`,
uses the direct profile with the 4,096-token input limit, records the new audit
fields, and contains no direct `causal_lm.generate()` call.

## Verification

Run formatting, lint, the inference and notebook tests, the complete CPU test
suite, notebook JSON and Python-cell validation, and the lockfile check. The
model-backed notebook remains a Colab workflow and is not executed in CI.

## Success criteria

- Qwen receives a valid chat-formatted input in both supported modes.
- Direct and reasoning behavior are explicit and auditable.
- A reasoning response truncated before its final answer cannot masquerade as
  a clean answer.
- FLenQA no longer owns or duplicates low-level generation logic.
- Existing task-specific evaluation behavior remains independent of inference.
