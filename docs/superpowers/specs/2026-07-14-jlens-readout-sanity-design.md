# J-Lens Readout Sanity Design

**Date:** 2026-07-14

## Purpose

Establish that the repository can run the public Jacobian Lens implementation
against a released open-weight model and lens, and reproduce the qualitative
readout behavior behind two examples from *Verbalizable Representations Form a
Global Workspace in Language Models*. This milestone reads activations only. It
does not implement the paper's spider-to-ant or France-to-China causal swaps.

The run is a practical open-model reproduction, not an exact replication of the
paper's Claude 4.5 results. It uses the model and fitted lens published in the
upstream J-Lens walkthrough:

- Model: `Qwen/Qwen3.5-4B`
- Lens repository: `neuronpedia/jacobian-lens`
- Lens revision: `qwen-n1000`
- Lens file:
  `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt`

## Execution Environment

The experiment runs in Colab with CUDA through the repository's existing Colab
bootstrap and runtime initialization. Model and lens downloads use the shared
Hugging Face cache beneath the configured artifact root. Local macOS and CI runs
do not download model weights and do not execute the GPU experiment.

The notebook must fail early with a clear message if CUDA is unavailable, the
model or lens cannot be loaded, the lens width differs from the model residual
width, or the fitted layers are incompatible with the model.

## Experiment Cases

### Spider intermediate

Use the paper's exact prompt:

```text
The number of legs on the animal that spins webs is
```

The baseline next-token distribution should identify the expected answer `8`
or its tokenizer-equivalent surface form. The readout analysis tracks the
single-token forms of `spider`, including capitalization or leading-space
variants that the tokenizer represents as distinct tokens.

The analysis reads all fitted layers at every prompt position and compares the
Jacobian lens with the ordinary logit lens. It records the target concept's best
rank and the top 25 decoded tokens at every inspected layer and position. An
interactive slice lets a researcher inspect where the intermediate appears.

### France flexible-use prompts

Use the four released flexible-generalization templates with `France`:

```text
The capital of France is the city of
Most people in France speak
France is a country on the continent of
The single-word name for the currency now used in France is the
```

Their expected baseline answers are `Paris`, `French`, `Europe`, and `Euro`.
For each prompt, the analysis tracks tokenizer-supported forms of `France`,
reads all fitted layers at every prompt position, and records the same
Jacobian-lens and logit-lens rank data as the spider case. The automatic
qualitative check considers only positions strictly after the final token of
the literal `France` span, so merely decoding the visible input word cannot
satisfy the check. One representative France prompt, the capital prompt, also
receives an interactive slice across all positions.

Because `France` is explicit in these prompts, this case is a readout and
cross-operation persistence check. It does not by itself establish the paper's
broadcast claim; that requires the deferred causal swap.

## Workspace Range

Raw results include every fitted layer. The summary treats fitted layer indices
from `ceil(0.35 * n_layers)` through `floor(0.80 * n_layers)`, inclusive, as the
candidate workspace range. This is an explicit open-model heuristic rather than
a claim that Qwen has the same workspace boundaries as Claude. Keeping every
layer in the artifact allows the heuristic to be revised without rerunning
model inference.

## Components

### Analysis module

A focused experiment module owns reusable, non-notebook logic:

- resolve all single-token vocabulary IDs for a concept across leading-space
  and capitalization variants;
- locate the token span for a literal argument in an encoded prompt and select
  positions strictly after that span;
- rank one or more target token IDs in a logits vector;
- decode top-k tokens without collapsing tokenizer-distinct forms;
- select the workspace-range layers from model depth;
- evaluate baseline-answer variants;
- assemble and serialize the experiment result.

The module operates on tensors, token IDs, decoded strings, and plain metadata.
It does not download models, initialize Colab, or write to fixed paths.

### Colab notebook

The notebook is copied from the repository's stable bootstrap pattern. After
initialization it:

1. loads Qwen in `bfloat16` on CUDA and wraps it with `jlens.from_hf`;
2. loads the released fitted lens through
   `JacobianLens.from_pretrained`;
3. validates model/lens compatibility;
4. runs the five prompts with the Jacobian and logit lenses;
5. prints a compact table of baseline answers and best target ranks;
6. writes a JSON result under `context.runs_dir`;
7. renders self-contained interactive HTML slices for the spider and France
   capital prompts under the same run directory.

No W&B run is required for this local qualitative milestone. Colab
initialization explicitly disables W&B so the notebook needs only GitHub and
Hugging Face credentials.

### Result artifact

The JSON artifact contains:

- project commit, model, lens repository, revision, and filename;
- PyTorch, Transformers, and J-Lens package provenance;
- model layer count, residual width, and selected workspace layers;
- prompt text and expected baseline answer for every case;
- actual baseline top tokens and whether an expected variant is top-1;
- per-layer-and-position J-Lens and logit-lens top-25 tokens;
- per-layer-and-position best rank across every resolved target-token variant;
- the best workspace-range rank, layer, and position for each lens type;
- pass/fail checks and explanatory failure messages.

Tensor values are converted to ordinary JSON-compatible numbers and lists.
The artifact records enough raw ranked output to review a failed heuristic
without downloading the model again.

## Success Criteria

The overall sanity run passes only if all of the following hold:

1. The pinned `jlens` package imports and the public model and lens load.
2. Model/lens residual widths and layer indices are compatible.
3. Every prompt's baseline next-token top-1 matches an accepted expected-answer
   token variant.
4. The spider concept family reaches J-Lens rank 25 or better at any prompt
   position in at least one workspace-range layer.
5. The France concept family reaches J-Lens rank 25 or better for every France
   prompt, at a position strictly after the literal argument span, in at least
   one workspace-range layer.

Logit-lens ranks are comparison data, not pass conditions. Interactive slices
are required outputs but are not graded automatically. If a qualitative check
fails, the notebook still saves complete results before raising a final summary
error, provided inference and serialization remain possible.

## Testing

CPU-only unit tests use synthetic logits and a small fake tokenizer to verify:

- concept variants retain only single-token encodings and are deduplicated;
- target ranks use one-based ranking and choose the best variant;
- ties are handled deterministically according to the ascending token-ID order
  preserved by `torch.argsort(..., descending=True, stable=True)`;
- workspace layer selection respects the inclusive 35%-80% rule;
- accepted baseline variants are evaluated by token ID rather than fragile
  decoded-string whitespace;
- result serialization contains no tensors and round-trips through JSON.

Notebook structure tests verify that the experiment notebook contains the
standard Colab loader, requests CUDA, disables W&B explicitly, and identifies
the pinned model and lens coordinates. Existing tests, Ruff formatting, and Ruff
linting must continue to pass.

## Deferred Scope

This milestone does not:

- fit a new Jacobian lens;
- implement lens-coordinate interventions or activation clamping;
- run spider-to-ant or France-to-China swaps;
- claim quantitative agreement with Claude 4.5;
- reproduce the full prompt-set evaluations from the paper;
- require W&B tracking.

The saved prompt definitions and result schema should be reusable by a later
causal-intervention milestone without changing the meaning of this readout-only
run.
