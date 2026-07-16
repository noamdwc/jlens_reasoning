# J-Lens Read-and-Change Sanity Design

**Date:** 2026-07-16

## Purpose

Replace the current readout-only pass rule with a compact, paper-aligned sanity
experiment that demonstrates two capabilities on the released
`Qwen/Qwen3.5-4B` Jacobian lens:

1. **Read:** the J-Lens surfaces an unspoken intermediate used by the model.
2. **Change:** swapping J-Lens coordinates causally moves the model's next-token
   answer toward the answer associated with the swapped-in concept.

This is an open-model framework sanity test, not an exact numerical replication
of Anthropic's Claude 4.5 results. It uses the paper's released prompts,
coordinate-swap equation, next-token grading, and intervention strengths while
adopting an explicit capability-level aggregate gate for Qwen.

## Why the Existing Rule Changes

The current experiment requires `France` to remain within the J-Lens top 25 at
a position strictly after the literal argument. That is a custom persistence
heuristic, not the flexible-generalization test released with the paper. It
also grades a 64-token free-form continuation even though the released
flexible-generalization protocol grades the greedy next token.

The revised experiment removes both mismatches:

- clean factual answers are graded from the next-token distribution;
- `France` readout ranks and workspace loading remain diagnostics, not a hard
  post-literal top-25 requirement;
- flexible use is tested causally with `France` to `China` coordinate swaps.

## Model and Lens

The experiment retains the pinned public artifacts already used by the branch:

- Model: `Qwen/Qwen3.5-4B`
- Lens repository: `neuronpedia/jacobian-lens`
- Lens revision: `qwen-n1000`
- Lens file:
  `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt`

The existing model/lens residual-width and source-layer validation remains a
required early check.

## Experiment Cases

### Unspoken intermediate

The read case and causal swap use the paper's spider example:

| Prompt | Clean answer | Read concept | Swap | Target answer |
| --- | --- | --- | --- | --- |
| `The number of legs on the animal that spins webs is` | `8` | `spider` | `spider` to `ant` | `6` |

The clean answer must be top-1. The read analysis finds the best `spider` rank
over all prompt positions and workspace layers for both the Jacobian lens and
the logit lens.

### Flexible generalization

The four released country templates use the same `France` to `China` swap:

| Function | Prompt | Clean answer | Swapped target |
| --- | --- | --- | --- |
| Capital | `The capital of France is the city of` | `Paris` | `Beijing` |
| Language | `Most people in France speak` | `French` | `Chinese` |
| Continent | `France is a country on the continent of` | `Europe` | `Asia` |
| Currency | `The single-word name for the currency now used in France is the` | `Euro` | `Yuan` |

All four clean answers must be top-1. The identical semantic coordinate swap is
applied for every function; it is not selected or tuned per prompt.

## Token Resolution

Answer grading continues to accept every capitalization and leading-space
variant that the Qwen tokenizer represents as one token.

Coordinate swaps use one explicit, case-preserving, leading-space token for
each concept: ` spider`, ` ant`, ` France`, and ` China`. Startup validation
must fail clearly if any configured swap surface does not encode as exactly one
token. Using a fixed token ID per concept prevents prompt-specific variant
selection from changing the intervention across functions. The chosen surface
and token ID are recorded in the result artifact.

## Workspace Range

The candidate workspace remains the fitted layers from
`ceil(0.35 * n_layers)` through `floor(0.80 * n_layers)`, inclusive. This is an
explicit Qwen heuristic; the paper reports Claude layers on a reindexed
0-to-100 scale and does not publish Qwen-specific workspace boundaries.

Every configured workspace layer receives the intervention, and every token
position in the prompt is patched. Raw readouts for all fitted layers remain in
the result artifact so the range can be reviewed without rerunning inference.

## Coordinate-Swap Intervention

For token `t` and layer `l`, form the J-Lens vector from the fitted Jacobian and
the model's unembedding row:

```text
v[t,l] = J[l]^T W_U[t]
```

For source `s`, target `t`, and hidden activation `h`, form:

```text
V = [v[s,l], v[t,l]]
c = pinv(V) h
h_patched = h + alpha * V * (swap(c) - c)
```

The operation is vectorized over the batch and sequence dimensions. Components
orthogonal to the span of the two lens vectors remain unchanged.

The experiment runs each swap at:

- `alpha=1`: the paper's standard coordinate exchange;
- `alpha=2`: the paper's double-strength condition.

A reusable hook context manager patches the output of every workspace block at
all prompt positions. It supports transformer blocks that return either a
tensor or a tuple whose first element is the hidden tensor, preserves all
non-hidden tuple members, and removes every hook on normal exit or exception.

The notebook obtains clean and intervened next-token logits from the same
causal language model. It does not generate a multi-token continuation for
grading.

## Read Metrics

For the spider case, record:

- best Jacobian-lens rank, layer, and position;
- best logit-lens rank, layer, and position;
- whether `spider` reaches Jacobian-lens top-1, matching the released paper
  convention;
- whether it satisfies the Qwen capability gate: Jacobian-lens rank at most 5
  and strictly better than its best logit-lens rank.

For each France prompt, retain rank diagnostics and add workspace loading: the
cosine similarity between the clean residual and the configured `France`
J-Lens vector, averaged over workspace layers and over the literal argument
position plus every following prompt position. France loading is explanatory
data and is not an individual pass condition.

## Change Metrics

For each of the five swap cases, record the target answer's:

- clean rank and top-1 token;
- rank and top-1 token after `alpha=1`;
- rank and top-1 token after `alpha=2`;
- best intervened rank across the two strengths;
- whether either strength strictly improves the target rank;
- whether either strength places the target answer at rank 1.

The `alpha=1` top-1 result is the primary paper-style metric. The `alpha=2`
result is reported separately as the stronger rescue condition; combining them
is used only for the Qwen capability gate.

## Pass Conditions

The overall sanity experiment passes only when all of these conditions hold:

1. The pinned model and lens load and pass compatibility validation.
2. All five clean expected answers are top-1 in their next-token distributions.
3. The spider read satisfies the Qwen capability gate: best J-Lens rank is at
   most 5 and is better than the best logit-lens rank.
4. At least three of the five swaps strictly improve the swapped-in target's
   rank at `alpha=1` or `alpha=2` compared with the clean run.
5. At least one swapped-in target reaches rank 1 at one of the two strengths.

Individual swaps may fail without failing the entire run. This matches the
paper's observation that coordinate swaps are not universally successful while
still requiring clear evidence that this model and framework can both read and
causally alter J-space content.

Failure messages distinguish model/lens compatibility, clean baseline, read,
aggregate rank-improvement, and top-1 intervention failures. Complete results
must be serialized before the notebook raises its final summary error.

## Code Organization

### Experiment module

`src/jlens_reasoning/experiments/readout_sanity.py` remains the reusable core
and gains:

- explicit swap-case definitions;
- strict single-token swap-surface resolution;
- J-Lens vector construction from `J_l` and `W_U`;
- the pure tensor coordinate-swap operation;
- the exception-safe block-hook context manager;
- next-token rank summaries for clean, `alpha=1`, and `alpha=2` runs;
- workspace-loading and aggregate capability-gate calculations.

Model download, Colab initialization, and fixed artifact paths remain outside
the module.

### Colab notebook

`notebooks/01_jlens_readout_sanity.ipynb` continues to load the pinned model
and lens, run the experiment, save `result.json`, and render the spider and
France-capital HTML views. It removes the long-form generation/evaluator loop
and instead supplies the causal LM's unembedding weights and next-token forward
pass to the reusable experiment code.

### Documentation

The README describes the notebook as a read-and-change sanity experiment and
lists the clean, `alpha=1`, and `alpha=2` metrics. Existing unrelated README
changes in the worktree are preserved.

## Result Artifact

The JSON artifact retains model, lens, package, and project-commit provenance
and the existing detailed readout payload. It additionally contains:

- swap source/target surfaces and token IDs;
- intervention strengths and workspace layers;
- clean and intervened top tokens and target ranks;
- per-case improvement and top-1 flags;
- France workspace loading;
- read and change aggregate counts;
- separate paper-style metrics and Qwen capability checks;
- structured failure messages.

No tensors or evaluator dataclasses may remain in the serialized result.

## Testing

CPU-only tests use synthetic tensors, fake tokenizers, and tiny transformer
blocks. Test-driven implementation must cover:

- exact released prompts, clean answers, and swap targets;
- strict configured swap-token resolution;
- J-Lens vector construction with known matrices;
- `alpha=0`, `alpha=1`, and `alpha=2` coordinate behavior;
- preservation of activation components orthogonal to the swap span;
- patching every sequence position at every selected layer;
- tensor and tuple block outputs;
- hook cleanup after normal execution and exceptions;
- next-token answer ranks across token variants;
- spider read capability checks;
- three-of-five rank-improvement and any-top-1 aggregation;
- JSON serialization of the expanded schema;
- notebook use of next-token intervention inference without long-form answer
  generation.

The full unit suite and Ruff checks must pass locally. The GPU-backed model run
remains a Colab verification step because CI does not download the model or
lens.

## Deferred Scope

This milestone does not:

- reproduce all 192 flexible-generalization trials;
- claim numerical agreement with Claude 4.5;
- fit a new Jacobian lens;
- automatically infer Qwen's workspace boundaries;
- support multi-token coordinate concepts;
- tune a separate intervention strength for each prompt;
- require every individual swap to succeed.

## References

- Anthropic, *Verbalizable Representations Form a Global Workspace in Language
  Models*: <https://transformer-circuits.pub/2026/workspace/index.html>
- Released flexible-generalization protocol and prompts:
  <https://github.com/anthropics/jacobian-lens/tree/main/data/experiments>
