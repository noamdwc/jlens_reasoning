# FLenQA probe × J-Lens

Primary notebook: `flenqa_probe_jlens.ipynb`. This is a descriptive experiment;
it contains no interventions. The existing probes predict the gold True/False
task label. Probe success establishes label decodability, not retention of all
task-relevant information.

## Run

1. Upload the current wheel with `scripts/upload_colab_wheel.sh --allow-dirty`
   when testing uncommitted changes, or omit that flag after committing.
2. Run `notebooks/flenqa_probe_assets.ipynb` to train **new chat-format probes**.
   Old raw-prompt probes are incompatible. The notebook reuses the original
   problem split and 250/500 training policy, and writes to a separate directory.
3. Ensure generated answers contain `input_sha256`. Legacy answers lack this
   evidence even if their metadata says `direct`: archive the old
   `runs/flenqa-full-run/model_outputs.parquet`, then rerun the setup and
   `save-model-outputs` cell of `notebooks/flenqa_full_run.ipynb` with the current
   wheel. Skip its `run-benchmark` cell; existing lens readout shards do not
   need to be regenerated for this alignment.
4. Run `notebooks/flenqa_probe_eval.ipynb` to export
   `runs/flenqa-probe-eval-chat-v2/{probe_results.parquet,auroc.parquet,manifest.json}`.
5. Run this experiment notebook in Colab. The CLI equivalent is
   `scripts/run_colab_notebook.sh experiments/flenqa_probe_jlens/flenqa_probe_jlens.ipynb`.

The probe checkpoint and metadata live under
`/content/drive/MyDrive/jlens-reasoning/checkpoints/flenqa-probe-assets-chat-v2/`.
The shared split remains at `checkpoints/flenqa-probe-assets/problem_split.json`;
do not delete or regenerate it during migration. Original raw probes and
evaluation/analysis directories are retained for historical comparison.
Saved generated answers live under
`/content/drive/MyDrive/jlens-reasoning/runs/flenqa-full-run/`.
Model and lens paths are the existing `MODEL_PATH` and `LENS_PATH` constants,
under `/content/drive/MyDrive/data/jlens-reasoning/assets/`.

## Input alignment and artifact compatibility

`prepare_chat_inputs` in `jlens_reasoning.inference` is shared by generation,
probe training, evaluation, and selected sensitivities. It supplies one user
message to the tokenizer's direct chat template with `add_generation_prompt=True`
and `enable_thinking=False`. There is no second tokenization, added BOS, padding,
or truncation. The 4096-token limit includes the wrapper. The probe and gradient
position is the final **wrapped** input token, immediately before generation.

Version 2 metadata fingerprints the tokenizer backend (including vocabulary,
normalizer, and special tokens) and chat template, and records the feature
boundary and input policy. Loaders reject incompatible contracts. Answer rows
record a SHA-256 of the actual input IDs and attention mask; evaluation checks
this before forwarding the prompt and carries the hash/token count into every
probe row. Sensitivity runs check it again, then reproduce both the saved probe
score and next-token margin. Manifests bind the checkpoint, answer file, probe
table, and AUROC table by hash. Do not relabel legacy artifacts as version 2 or
backfill hashes onto old answers: regenerate them to establish this provenance.

The descriptive margin column is now `output_margin` (and the gold-oriented
analysis column is `gold_output_margin`), replacing the old `raw_*` names.
New measurements must be reproduced before comparing them with the preliminary
raw/chat conclusions in `docs/probe_jlens_routing_framework.md`. That document
is retained unchanged as a historical research note, not a validated result of
the aligned pipeline.

## Quantities and limitations

- Probe score: `w @ (h - training_mean) + bias`. The threshold and probe are
  frozen. A gold-label sigmoid score of at least 0.8 defines the descriptive
  strong-score subset; it is not calibrated confidence under context shift.
- Static propagation: `v = J_bar @ (w / ||w||)`. `JacobianLens.transport` uses
  row batches, `h @ J_bar.T`. The matrix orientation is tested with a
  nonsymmetric matrix and against the existing `jlens_vector` pullback.
- Vocabulary scores: `W_U @ v`. Equivalently each score is the dot product of
  the probe direction with `J_bar.T @ W_U[token]`. These are token directions,
  not a separate semantic concept representation.
- Normalization: `jlens.from_hf(...).unembed` applies the final norm and then
  the output head. Therefore the linear vocabulary scores are not normalized
  lens logits or their local derivatives.
- Static estimator: the pinned fitter sums derivatives over valid downstream
  target positions and averages over valid source positions and prompts.
  The saved external lens lacks target-layer/estimator metadata; its fitting
  provenance cannot be recovered from a square tensor's dimensions.
- Feature boundary: existing probes use `hidden_states[layer + 1]`. The last
  entry is post-final-norm. That probe remains in performance/sensitivity
  tables but is excluded from projection through block-output J-Lens maps.
- Prompt sensitivity: `grad_h(True-minus-False margin) @ unit_probe`, at the
  final wrapped input token. Orienting both the margin and direction toward the
  gold label gives the same derivative for either label.
- Input formats: generated answers, probes, and gradients use the same direct
  chat input. The next-token margin remains a diagnostic, not a graded generated
  answer. This alignment does not establish causal use of the decoded feature.

Results are saved under `runs/flenqa-probe-jlens-chat-v2/`: three figures, per-layer
performance/failure tables, full strong-failure rows, static propagation,
vocabulary scores, selected pair IDs, prompt and paired sensitivities, and a
run manifest. The optional `notebooks/flenqa_probe_jlens_concepts.ipynb` now
views the primary notebook's vocabulary table.

Pairs retain all source provenance. Selection uses the validation-chosen
headline layer, seed 1729, and up to three distinct problems per label/cohort.
The existing matched lengths are 2000 and 3000; performance covers all five
lengths. Selected gradients are case studies, not independent population
estimates. A missing cohort is reported explicitly.

The next causal experiment must first reproduce measurements with the aligned
pipeline, then prespecify a layer and norm-controlled direction using development
data. Test on fresh held-out problems with identity and random-direction
controls. This branch does not select an intervention strength or run it.

## Reading the code

- `notebooks/flenqa_probe_assets.ipynb`: fixed problem split → select 250/500
  variants → extract final-token states → fit and save probes. Training still
  uses all selected source rows, including identical prompt variants.
- `notebooks/flenqa_probe_eval.ipynb`: load frozen probes → prepare unique test
  prompts → grade saved chat answers → verify input hashes → extract chat states → build one
  `probe_results` table. Performance, failure summaries, AUROC, and export all
  use that table.
- `flenqa_probe_jlens.ipynb`: load scores → define strong failures → inspect
  performance → project directions → select matched pairs → differentiate
  selected prompts → compare and export. Loading, tables, and plots have
  separate cells so intermediate results remain inspectable.
- `analysis.py` contains only three helpers: validate the saved problem split,
  match provenance conditions, and compute `J_bar @ unit_probe` followed by
  `W_U @ projected_probe`. The projection takes tensors directly.
- `inputs.py` defines the direct-chat input contract and rejects incompatible
  probe assets or prompt records; tokenization itself stays in shared inference.
- `notebooks/flenqa_probe_jlens_concepts.ipynb` remains a small optional viewer
  of the exported vocabulary table.

## Refactor and validation

The cleanup removes repeated split checks, upstream-guaranteed tensor/context
checks, repeated answer grading in the analysis notebook, duplicate score-table
assembly, and unused per-example metadata. New checkpoints omit the unused
`unit_weight`. The alignment changes the feature inputs, so weights, training
means, and biases must be refit. Split assignments, centering, validation-based
regularization/layer selection, and the zero decision threshold are preserved.

The split and pairing helpers remain because leakage and mismatched conditions
can invalidate an experiment without causing a tensor error. Hashes still bind
the evaluation manifest to its checkpoint and generated-answer file. Model
identity, matching chat inputs, the final-normalization boundary, finite probe
directions, token variants, and recomputed-vs-saved probe scores remain checked.
`ActivationRecorder` remains responsible for autograd setup and hook cleanup.
Shared dataset normalization, grading, and deterministic token ranking are
reused without redesigning those modules.

The CPU regression suite executes the notebook extraction and sensitivity cells
with a tiny randomly initialized model and a local tokenizer for all three
FLenQA task renderers. It checks generation token/mask equality, the final
wrapped position, training/evaluation feature equality, saved score/margin
reproduction, hook cleanup, and rejection of stale records and tokenizers.
These checks validate code wiring, not scientific outcomes on Qwen.

Scientific caveats remain: static maps cannot establish a context-length
effect; the external lens lacks fitting-provenance metadata;
the final probe is post-normalization; and label decodability is not evidence
of causal use. The evaluation notebook's broad "model wrong" summaries include
unparseable answers, while the primary strong-failure cohort requires a parsed
wrong answer. These existing distinctions are preserved. No real FLenQA result
is claimed by the synthetic validation, and no J-gain intervention is run here.
