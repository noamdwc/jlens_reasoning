# FLenQA probe × J-Lens

Primary notebook: `flenqa_probe_jlens.ipynb`. This is a descriptive experiment;
it contains no interventions. The existing probes predict the gold True/False
task label. Probe success establishes label decodability, not retention of all
task-relevant information.

## Run

1. Upload the current wheel with `scripts/upload_colab_wheel.sh --allow-dirty`
   when testing uncommitted changes, or omit that flag after committing.
2. Use existing `notebooks/flenqa_probe_assets.ipynb` assets, or run that notebook
   if they do not exist. It preserves the problem split and 250/500 training
   policy. Do not retrain merely to change this analysis.
3. Run `notebooks/flenqa_probe_eval.ipynb` once with the current wheel to export
   `runs/flenqa-probe-eval/{probe_results.parquet,auroc.parquet,manifest.json}`.
4. Run this experiment notebook in Colab. The CLI equivalent is
   `scripts/run_colab_notebook.sh experiments/flenqa_probe_jlens/flenqa_probe_jlens.ipynb`.

The probe checkpoint and split live under
`/content/drive/MyDrive/jlens-reasoning/checkpoints/flenqa-probe-assets/`.
Saved generated answers live under
`/content/drive/MyDrive/jlens-reasoning/runs/flenqa-full-run/`.
Model and lens paths are the existing `MODEL_PATH` and `LENS_PATH` constants,
under `/content/drive/MyDrive/data/jlens-reasoning/assets/`.

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
  final raw-input token. Orienting both the margin and direction toward the
  gold label gives the same derivative for either label.
- Input formats: saved generated answers use the direct chat template; probes
  and gradients use raw FLenQA text. Report cross-format failure associations
  separately from the same-input next-token margin. The latter is a diagnostic,
  not a graded generated answer.

Results are saved under `runs/flenqa-probe-jlens/`: three figures, per-layer
performance/failure tables, full strong-failure rows, static propagation,
vocabulary scores, selected pair IDs, prompt and paired sensitivities, and a
run manifest. The optional `notebooks/flenqa_probe_jlens_concepts.ipynb` now
views the primary notebook's vocabulary table.

Pairs retain all source provenance. Selection uses the validation-chosen
headline layer, seed 1729, and up to three distinct problems per label/cohort.
The existing matched lengths are 2000 and 3000; performance covers all five
lengths. Selected gradients are case studies, not independent population
estimates. A missing cohort is reported explicitly.

The next causal experiment must align probe and model-answer input formats
first, then prespecify a layer and norm-controlled direction using development
data. Test on fresh held-out problems with identity and random-direction
controls. This branch does not select an intervention strength or run it.

## Reading the code

- `notebooks/flenqa_probe_assets.ipynb`: fixed problem split → select 250/500
  variants → extract final-token states → fit and save probes. Training still
  uses all selected source rows, including identical prompt variants.
- `notebooks/flenqa_probe_eval.ipynb`: load frozen probes → prepare unique test
  prompts → grade saved chat answers → extract raw-prompt states → build one
  `probe_results` table. Performance, failure summaries, AUROC, and export all
  use that table.
- `flenqa_probe_jlens.ipynb`: load scores → define strong failures → inspect
  performance → project directions → select matched pairs → differentiate
  selected prompts → compare and export. Loading, tables, and plots have
  separate cells so intermediate results remain inspectable.
- `analysis.py` contains only three helpers: validate the saved problem split,
  match provenance conditions, and compute `J_bar @ unit_probe` followed by
  `W_U @ projected_probe`. The projection takes tensors directly.
- `notebooks/flenqa_probe_jlens_concepts.ipynb` remains a small optional viewer
  of the exported vocabulary table.

## Simplification notes

The cleanup removes repeated split checks, upstream-guaranteed tensor/context
checks, repeated answer grading in the analysis notebook, duplicate score-table
assembly, and unused per-example metadata. New checkpoints omit the unused
`unit_weight`; readers also accept existing checkpoints containing that field.
Weights, training means, biases, split assignments, fit policy, output columns,
and scientific definitions are unchanged.

The split and pairing helpers remain because leakage and mismatched conditions
can invalidate an experiment without causing a tensor error. Hashes still bind
the evaluation manifest to its checkpoint and generated-answer file. Model
identity, raw/chat input formats, the final-normalization boundary, finite probe
directions, token variants, and recomputed-vs-saved probe scores remain checked.
`ActivationRecorder` remains responsible for autograd setup and hook cleanup.
Shared dataset normalization, grading, and deterministic token ranking are
reused without redesigning those modules.

Compared with `experiment/flenqa-probe-jlens`, the three working notebooks plus
`analysis.py` contain 1,393 code lines instead of 1,580 (counting blank lines and
the unchanged loaders, excluding notebook JSON and markdown). The largest code
cell in the primary notebook shrank from 138 to 65 lines. No new framework or
classes were introduced. One redundant mixed-context pairing test was removed;
the shared prompt-preparation tests protect that invariant upstream.

Validation used the existing CPU-only suite (413 tests), notebook schema/code
compilation and import checks, scoped Ruff lint/format checks, and an offline
lockfile check. A temporary synthetic before/after run compared probe training,
evaluation/export tables, seeded pair selection, static projections, and actual
autograd sensitivities, including empty cohorts. It used no external data or
real model assets. The neighboring failure-concept notebook's loader was
restored to the canonical loader to fix a pre-existing test failure; its
experiment cells were untouched. Repository-wide Ruff checks still report
pre-existing style issues outside the probe code.

Scientific caveats are separate from this cleanup: raw-prompt probes and chat
answers use different input formats; static maps cannot establish a
context-length effect; the external lens lacks fitting-provenance metadata;
the final probe is post-normalization; and label decodability is not evidence
of causal use. The evaluation notebook's broad "model wrong" summaries include
unparseable answers, while the primary strong-failure cohort requires a parsed
wrong answer. These existing distinctions are preserved, not resolved by the
refactor. No real FLenQA result is claimed by the synthetic validation.
