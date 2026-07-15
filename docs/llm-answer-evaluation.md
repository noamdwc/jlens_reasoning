# Evaluating LLM Answers

**Status:** Normative project policy

**Version:** 1.0

**Date:** 2026-07-15

**Adoption note:** The current `readout_sanity.py` evaluator predates this
policy. Its one-token baseline and post-argument France check are not compliant
with version 1.0 and must be migrated before new results from that evaluator are
treated as policy-compliant.

## Purpose

This document defines how `jlens-reasoning` decides whether an LLM answered a
task correctly. Every experiment that grades model output must select and
record an evaluation contract before inspecting results. The contract must
separate model generation, answer extraction, normalization, and scoring so
that results remain reproducible and failures remain interpretable.

The central rule is:

> Preserve the raw response, extract a prediction without consulting the gold
> answer, and only then compare the prediction with the reference.

An evaluator must not search only for the expected answer inside arbitrary
model text. That can award false credit to responses such as `8 is wrong; the
final answer is 6`.

## Evaluation Layers

Every graded response passes through the following layers in order:

1. **Raw response:** Exact generated token IDs, decoded token pieces, and text.
2. **Visible response:** Text intended as the model's answer after applying a
   declared, model-specific reasoning-output protocol.
3. **Answer region:** A deterministic, gold-blind extraction from the visible
   response, such as the content after a final-answer marker.
4. **Normalized prediction and references:** Text transformed by the declared
   task-specific normalization profile.
5. **Score:** A declared comparison such as prefix match, exact match, numeric
   equivalence, candidate likelihood, or rubric judgment.

No layer may silently rewrite an earlier layer. All intermediate values and
parser statuses must be retained in the result artifact.

## Declare the Contract Before Running

Every experiment must define these fields before model outputs are examined:

- task type and primary metric;
- exact prompt or message sequence;
- whether a chat template or raw completion format is used;
- generation mode, stop conditions, and safety limit;
- reasoning-output protocol, if any;
- answer extractor and its version;
- normalization profile and its version;
- accepted references or answer choices;
- treatment of truncation, malformed output, and ambiguity;
- any auxiliary metrics used only for comparison with a paper.

Changing any of these fields changes the meaning of the evaluation. Results
from different contracts must not be aggregated under the same metric name.

## Choose the Scorer by Task Type

There is no universal LLM-output normalizer or scorer. Use the narrowest
contract that measures the behavior of interest.

| Task type | Primary scorer | Notes |
| --- | --- | --- |
| Fixed continuation | Greedy sequence or conditional log-likelihood | Scores the complete expected token sequence, not one decoded token. |
| Multiple choice | Conditional log-likelihood over every declared choice | Record raw and length-normalized variants when answer lengths differ. |
| Short factual answer | Gold-blind answer-region extraction, then prefix or exact match | Appropriate for the J-Lens completion prompts in this repository. |
| Numeric answer | Gold-blind extraction, then numeric equivalence | Define separators, signs, precision, units, and tolerance in advance. |
| Symbolic mathematics | Gold-blind extraction, then symbolic equivalence | Preserve the original strings and record parser failures. |
| Structured output | Parse the declared schema, then compare typed fields | Invalid syntax is `unparseable`, not an empty answer. |
| Open-ended answer | Fixed rubric with human or validated model judgment | Use only when deterministic scoring cannot express correctness. |

Exact match, prefix match, substring inclusion, and semantic judgment are
different metrics. An experiment must name the one it uses; it must not call
all of them "accuracy."

## Generation Policy

### Deterministic runs

Sanity checks and causal comparisons use deterministic decoding unless the
experiment explicitly studies sampling:

- greedy decoding (`do_sample=False`);
- a declared EOS token and task-specific stop strings;
- a declared `max_new_tokens` safety limit;
- identical generation settings for clean and intervened conditions.

The result must record every generation argument. Temperature zero and greedy
decoding are not assumed from a library default.

### Safety limits are not answer windows

`max_new_tokens` prevents runaway generation; it does not define where an
answer is allowed to appear. Generate until EOS or a declared stop condition,
subject to the safety limit, and then parse the response.

If a correct answer is in the visible answer region at generated token 64, it
can pass a contract whose safety limit includes that position. If the safety
limit is reached before EOS or a declared stop condition and no conclusive
answer can be parsed, the outcome is `truncated`, not `incorrect`.

An answer that appears only after a contradictory or unrelated preamble is not
automatically correct. Correctness depends on the declared answer-region
extractor, not on searching the full generation for a gold string.

## Reasoning and Thinking Output

Reasoning models may expose internal-style text using delimiters such as
`<think>...</think>`, return a distinct reasoning field, or hide reasoning
entirely. The experiment must use the protocol declared by the model adapter or
tokenizer configuration.

The default rules are:

1. Preserve reasoning fields and delimiters in the raw artifact.
2. If reasoning is returned in a separate field, grade only the declared final
   answer field.
3. If a model declares paired reasoning delimiters, remove complete reasoning
   spans and grade the remaining visible text.
4. When a closing reasoning delimiter is present, content after the final
   closing delimiter is visible answer text.
5. Do not invent or guess delimiters for a model that does not declare them.
6. Unbalanced, nested in an unsupported way, or otherwise malformed delimiters
   produce primary status `unparseable` with reason `malformed_reasoning`; they
   are not silently repaired.

For example, `<think>Spiders have eight legs.</think>\n8` has the visible
response `8`. An answer mentioned only inside a removed reasoning span does not
count unless the task contract explicitly grades reasoning content.

## Short Factual Completion Policy

The J-Lens spider and France prompts are short factual completions. They use
the following project scorer, named `short_factual_completion_v1`.

### Step 1: Produce the visible response

Apply the declared reasoning-output protocol. Normalize line endings, but do
not otherwise alter the text.

### Step 2: Select the answer region without the gold answer

Use the following precedence:

1. If the task requested a structured answer field and it parses, use that
   field.
2. Otherwise, if one or more declared answer markers appear, use the content
   after the final marker. The initial marker set is case-insensitive and
   covers `final answer:`, `final answer is`, `answer:`, `answer is`, and
   `the answer for/to this/the question is`.
3. Otherwise, use the entire visible response as the answer region.

Marker matching must be implemented independently of the reference answer.
Choosing the final marker ensures that `8 is wrong; final answer: 6` extracts
`6`.

### Step 3: Apply minimal text normalization

The `minimal_text_v1` profile performs only:

- Unicode NFC normalization;
- `\r\n` and `\r` to `\n` line-ending normalization;
- leading and trailing whitespace removal;
- internal whitespace-run collapse to one ASCII space;
- Unicode-aware case folding when the task is case-insensitive.

It does not remove articles, punctuation, signs, decimal points, units, or
diacritics. Those transformations can change a correct answer into an
incorrect one or vice versa and require a separately named task profile.

### Step 4: Prefix-match a declared reference

The normalized answer region must begin with one accepted normalized reference.
The reference must be followed by the end of the answer region or by declared
sentence-final punctuation such as `.`, `!`, or `?`. This rejects partial and
ambiguous continuations such as `eight` matching `eighteen`, `8 or 6`, and
`8 is wrong` while still accepting `8. This conclusion is based...`.

Tasks that need to accept a broader form, such as `Paris, France`, must declare
a different versioned scorer or use a validated semantic judge. They must not
weaken `reference_prefix_v1` after inspecting evaluation outputs.

Prefix matching, rather than unrestricted substring inclusion, gives the
following results:

| Raw completion | Extracted answer region | Reference | Result |
| --- | --- | --- | --- |
| ` 8.\n\nThis conclusion is based...` | full visible response | `8` | Correct |
| `The answer for this question is 8.` | `8.` | `8` | Correct |
| `<think>...</think>\n8` | `8` | `8` | Correct |
| `<think>8</think>\n6` | `6` | `8` | Incorrect |
| `8 is wrong; final answer: 6` | `6` | `8` | Incorrect |
| `The possibilities are 8 and 6.` | full visible response | `8` | Incorrect |
| `final answer: 8 or 6` | `8 or 6` | `8` | Ambiguous |
| `eighteen` | full visible response | `eight` | Incorrect |

Accepted references are part of the dataset contract. For the spider prompt,
`8` and `eight` are separate accepted references. Case-folding permits `Euro`
and `euro` to compare equal without adding output-dependent aliases after a
run.

## Paper-Faithful Metrics Versus Semantic Correctness

A reproduction may need a paper-faithful metric that is narrower than ordinary
semantic correctness. Record the two separately.

For the J-Lens flexible-generalization data, the released protocol checks
whether the greedy next token equals the answer. The public Qwen tokenizer can
instead represent the visible completion ` 8` as two tokens: a standalone space
followed by `8`.

Implementations governed by this policy must record:

- `paper_immediate_match`: the original paper-style next-token criterion;
- `greedy_reference_sequence`: whether a complete accepted reference surface
  is greedy token-by-token, including formatting tokens;
- `output_correct`: the primary semantic result from
  `short_factual_completion_v1`.

For a fixed continuation, sequence-level greedy scoring evaluates each target
token conditioned on the prompt and all preceding target tokens. It supports
an expected answer of any token length and must not be replaced by checking
whether every answer token is independently likely at the original prompt
boundary.

Auxiliary paper-faithful metrics never silently override the primary metric.
Reports must say which one gates experiment success.

## Other Normalization Profiles

### Extractive QA

SQuAD-style exact match lowercases, removes punctuation and English articles,
and collapses whitespace. That is appropriate when reproducing SQuAD, but it
is not the project default because it is unsafe for code, mathematics, units,
and many multilingual answers.

### Numeric answers

A numeric profile must declare:

- decimal and thousands separators;
- whether scientific notation is accepted;
- sign handling;
- units and unit conversion;
- exact versus tolerance-based equality;
- treatment of percentages, fractions, `NaN`, and infinity.

Parsing must yield a typed value or `unparseable`. Never remove every
non-numeric character and hope the remaining digits are the intended answer.

### Symbolic mathematics

Symbolic tasks should use a restricted parser and symbolic equivalence checker
where possible. Record both original expressions, parser versions, assumptions,
and timeouts. A timeout is `unresolved`, not `incorrect`.

## Ambiguity and Failure States

Every item has one primary status:

- `correct`;
- `incorrect`;
- `unparseable`;
- `ambiguous`;
- `truncated`;
- `unresolved`;
- `generation_error`;
- `scoring_error`.

Only `correct` contributes one to accuracy. Other statuses normally contribute
zero, but they must remain separate in reports so infrastructure failures are
not mistaken for model failures.

Examples of ambiguity include conflicting structured answer fields, a declared
answer region containing alternatives such as `8 or 6`, or an output grammar
that cannot determine which segment is final. Ordered text markers are resolved
by the declared final-marker rule. A parser must fail closed rather than select
whichever candidate matches the reference.

## Model-Graded Evaluation

Use a model judge only when deterministic parsing and equivalence cannot express
the task's correctness criterion. A judge contract must record:

- the complete rubric and judge prompt;
- judge model and immutable revision where available;
- generation parameters;
- reference answers and whether the judge sees them;
- allowed labels and invalid-output handling;
- calibration examples;
- agreement with a human-labeled validation set.

Judge quality must be evaluated as its own classifier. For consequential
results, report human agreement, class balance, and uncertainty. Do not use a
judge merely to avoid specifying a deterministic parser for simple factual
answers.

## Required Result Provenance

Every saved result must include enough information to replay the evaluation:

```json
{
  "model": {"name": "...", "revision": "..."},
  "tokenizer": {"name": "...", "revision": "..."},
  "prompt": "...",
  "chat_template": null,
  "generation": {
    "do_sample": false,
    "max_new_tokens": 128,
    "stop": ["<eos>"]
  },
  "raw": {
    "token_ids": [],
    "token_pieces": [],
    "text": "...",
    "finish_reason": "eos"
  },
  "evaluation": {
    "reasoning_protocol": "none",
    "extractor": "short_factual_completion_v1",
    "normalizer": "minimal_text_v1",
    "scorer": "reference_prefix_v1",
    "visible_response": "...",
    "answer_region": "...",
    "normalized_prediction": "...",
    "normalized_references": [],
    "matched_reference": null,
    "reason": null,
    "status": "incorrect"
  }
}
```

Also record the project commit, dependency versions, device and dtype, random
seed when sampling, and intervention parameters when applicable. Do not store
credentials or hidden provider reasoning.

## Regression Requirements

Every evaluator implementation must have tests for:

- multi-token answers and tokenizer-separated leading whitespace;
- case-folded accepted references;
- answer-marker extraction;
- complete and malformed reasoning delimiters;
- answers after a reasoning block;
- a gold answer mentioned only inside reasoning;
- conflicting preliminary and final answers;
- substring traps such as `eight` versus `eighteen`;
- output containing both correct and incorrect candidates;
- EOS and every configured stop condition;
- safety-limit truncation;
- empty, malformed, and Unicode-normalization edge cases;
- serialization of all intermediate evaluation layers.

Each bug found in a real model artifact must become a regression fixture before
the evaluator is changed.

## Anti-Patterns

Do not:

- grade only one token when the answer surface can span multiple tokens;
- skip a leading token and reuse logits from the old context;
- search the full response only for the gold answer;
- accept any response that contains the gold answer somewhere;
- strip all punctuation or all non-digits by default;
- infer reasoning delimiters from output text after seeing a result;
- treat truncation or parser failure as ordinary model incorrectness;
- change references, extraction rules, or normalization after inspecting the
  evaluation set without versioning the metric and rerunning all comparisons;
- report a judge score without validating the judge.

## References and Precedents

- The [J-Lens paper](https://transformer-circuits.pub/2026/workspace/index.html)
  and its
  [released experiment specification](https://github.com/anthropics/jacobian-lens/tree/main/data/experiments)
  define paper-specific next-token and lens-readout protocols.
- The
  [LM Evaluation Harness model interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)
  distinguishes conditional log-likelihood from bounded generation, and its
  [task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)
  makes answer filtering task-specific.
- The
  [official SQuAD 2.0 evaluator](https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/evaluate-v2.0.py)
  is the precedent for its named article, punctuation, case, and whitespace
  normalization profile.
- The [official GSM8K repository](https://github.com/openai/grade-school-math)
  uses an explicit `####` final-answer marker for numeric extraction.
- [OpenAI Evals templates](https://github.com/openai/evals/blob/main/docs/eval-templates.md)
  distinguish prefix match, inclusion, fuzzy match, structured comparison, and
  model-graded evaluation rather than treating them as interchangeable.
