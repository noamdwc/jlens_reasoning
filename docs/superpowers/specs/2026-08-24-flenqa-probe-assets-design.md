# FLenQA Linear Probe Assets Design

## Goal

Add a single, readable notebook that creates a frozen problem-level FLenQA
split, trains one binary linear probe for each transformer layer from final
input-token residual states, and saves all assets needed by later probe
experiments without inspecting the test examples.

## Scope and non-goals

The notebook is `notebooks/flenqa_probe_assets.ipynb`. It will contain setup,
dataset loading, fixed split creation/loading, prompt selection, activation
extraction, probe fitting, asset serialization, and a compact save summary.
It will not contain J-Lens analysis, interventions, shuffled-label controls,
random-probe controls, long-context test evaluation, or any scientific result
about test failures.

## Existing abstractions and invariant identifier

The notebook will use the existing Colab bootstrap and artifact paths, the
existing FLenQA dataset loader and `normalize_rows(..., full=True)`,
`FlenqaRow.problem_id`, and the existing model constants and local-only
Transformers loading convention. `problem_id` is the normalized value of the
published `global_sample_id`; the dataset validator confirms 300 problems with
40 variants per problem, so it identifies an underlying reasoning problem and
not an individual prompt variant.

`prepare_prompts()` will render and deduplicate exact final prompts. The
notebook will retain the source-row metadata needed to associate each prepared
prompt with exactly one nominal `ctx_size`; it will assert that variants of a
problem receive the same split before selecting rows.

## Split and data policy

The split is stratified at the problem level by `(task, label)` using a fixed
seed and 60/20/20 train/validation/test proportions. The split asset stores
the seed, strategy, and sorted problem IDs in each partition, plus the
problem-level task and label used for stratification. If the split asset
already exists, it is loaded and its metadata and integrity assertions are
checked rather than regenerated.

Only rows with `ctx_size` 250 or 500 and problem IDs in train or validation
are used for probe work. Train rows fit the probes; validation rows select
regularization; test rows are not loaded into activation matrices, fit calls,
or metric calculations. The notebook asserts that all problem IDs are
partition-disjoint, that their union is the full problem set, and that every
source-row variant inherits its problem’s saved partition.

## Activation extraction

For each selected prompt, the notebook tokenizes the exact prompt with the
existing tokenizer and runs the causal language model with hidden states
returned. The feature for every transformer layer is the residual hidden state
at `input_ids.shape[1] - 1`, the final input token immediately before answer
generation. The resulting per-layer matrices have shape
`[num_examples, hidden_dim]`; extraction is performed in inference mode and
stored on CPU before fitting.

The model’s embedding/output hidden-state entry is handled explicitly by
using the transformer layer entries corresponding to
`model.config.num_hidden_layers`; the saved metadata records the layer indices
and hidden dimension. No activation normalization is applied during extraction.
Probe fitting may center each layer using its training mean, and if so that
mean is saved and the saved score is defined as `unit_weight @ (h -
training_mean) + bias`.

## Probe fitting and saved asset

Each layer uses binary L2-regularized logistic regression with labels converted
to `0=False` and `1=True`, and with the positive score convention explicitly
defined as `True`. A small fixed grid of inverse regularization strengths is
fit on train features and selected by validation log loss, breaking ties by
choosing the smaller value. No test data participates in selection.

The reusable asset is a single `torch.save` dictionary under
`context.checkpoints` containing:

- the fixed split and split metadata;
- layer-indexed `weight`, `bias`, and normalized `unit_weight` tensors;
- the per-layer training mean (or an explicit zero mean if centering is not
  used);
- selected regularization values;
- train and validation loss/accuracy metrics;
- model/tokenizer names and revision metadata, prompt/activation settings,
  seed, label convention, and source commit marker.

A small JSON sidecar records the same non-tensor metadata in a human-readable
form. The notebook creates parent directories through the existing artifact
path object and prints only train/validation summaries and the saved paths.

## Verification

The notebook itself contains data-integrity assertions and checks that every
layer has a nonzero probe direction and matching feature dimensions. Repository
verification will include notebook structural checks (no outputs or execution
counts, canonical loader cell, no credentials) and the project’s standard
format, lint, lock, and CPU-only test commands. Model-backed execution remains
a Colab/GPU responsibility and is not added to CI.
