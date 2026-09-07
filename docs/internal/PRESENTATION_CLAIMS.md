# Public presentation claim ledger

This ledger records the claims surfaced by `README.md` and keeps code-backed
workflow contracts separate from empirical findings. It is intentionally not a
results report. A future result release should update the relevant entries with
the exact committed artifact, model/lens revision, data revision, and analysis
commit.

## Status meanings

- **Verified** — directly supported by tracked code, tests, notebook structure,
  or an external source explicitly linked as attribution. For workflow claims,
  this means the repository implements the contract; it does not mean a
  model-backed run has been executed.
- **Conservative inference** — a bounded interpretation of the implementation
  or evidence boundary; it should not be read as a measured result.
- **Hypothesis** — a question or proposed causal interpretation that the
  experiments are designed to test.

## Claims

### Research framing

**CLAIM:** The repository investigates whether internal readout distributions
change as matched FLenQA prompts become longer, and whether candidate J-Lens
directions are related to answer behavior.

**EVIDENCE:** The drift and intervention notebook descriptions and code define
matched context sizes, position-aware readouts, drift summaries, and
coefficient interventions.

**FILE / ARTIFACT:**
[`experiments/flenqa_lens_drift/flenqa_lens_drift.ipynb`](../experiments/flenqa_lens_drift/flenqa_lens_drift.ipynb),
[`experiments/flenqa_lens_drift/flenqa_lens_intervention.ipynb`](../experiments/flenqa_lens_drift/flenqa_lens_intervention.ipynb)

**STATUS:** Hypothesis

**CLAIM:** The repository is currently stronger evidence of research
engineering than of public scientific outcomes.

**EVIDENCE:** Tracked notebooks have empty outputs and null execution counts;
the tracked tree contains no generated model-output, top-k, probe, or report
artifact.

**FILE / ARTIFACT:** `tests/test_notebooks.py`; tracked repository tree at the
presentation branch base.

**STATUS:** Verified

### FLenQA and prompt construction

**CLAIM:** The FLenQA implementation covers PIR, MonoRel, and Simplified
RuleTaker, with declared context sizes of 250, 500, 1000, 2000, and 3000.

**EVIDENCE:** The benchmark constants and task-specific prompt renderer declare
these task and context-size sets.

**FILE / ARTIFACT:** `src/jlens_reasoning/benchmarks/flenqa/dataset.py`

**STATUS:** Verified

**CLAIM:** The full-dataset validation contract expects 12,000 source rows,
300 underlying problems, and 40 source variants per problem.

**EVIDENCE:** `FULL_DATASET_ROW_COUNT` and `_validate_full_counts` assert these
published-dataset invariants; the probe notebooks reuse
`normalize_rows(..., full=True)`.

**FILE / ARTIFACT:**
`src/jlens_reasoning/benchmarks/flenqa/dataset.py`,
`notebooks/flenqa_probe_assets.ipynb`

**STATUS:** Verified as an implementation/data-integrity contract, not a
newly measured run result.

**CLAIM:** Exact prompt text is hashed, deduplicated in first-occurrence order,
and retains source-row condition provenance.

**EVIDENCE:** `compute_prompt_id`, `prepare_prompts`, `FlenqaPrompt.provenance`,
and `SourceProvenance` implement and serialize these fields.

**FILE / ARTIFACT:**
`src/jlens_reasoning/benchmarks/flenqa/dataset.py`,
`src/jlens_reasoning/benchmarks/flenqa/storage.py`

**STATUS:** Verified

**CLAIM:** The FLenQA runner is configured to produce 9,862 unique prompt
records from the full run.

**EVIDENCE:** The full-run and accuracy notebooks assert `len(records) ==
9_862` and the expected per-length unique counts before writing outputs.

**FILE / ARTIFACT:** `notebooks/flenqa_full_run.ipynb`,
`notebooks/flenqa_accuracy.ipynb`

**STATUS:** Verified as a workflow assertion; not an observed public run
result because the generated table is not committed.

**CLAIM:** Short and long prompts are prepared independently; the experiment
does not align different-length sequences with fake padding.

**EVIDENCE:** `prepare_prompt` resolves spans and positions in each prompt's
own tokenization, and the drift/intervention notebooks explicitly select
matched prompt IDs and retain each prompt's own activation sequence.

**FILE / ARTIFACT:**
`src/jlens_reasoning/benchmarks/flenqa/positions.py`,
`experiments/flenqa_lens_drift/flenqa_lens_intervention.ipynb`

**STATUS:** Verified

### Readouts, evaluation, and probes

**CLAIM:** The benchmark runner extracts both Jacobian Lens and Logit Lens
top-k values at explicit fact, question, padding, and final-prompt positions.

**EVIDENCE:** Position labels are defined in `positions.py`; `run_prompt` runs
both lens modes and writes typed `prompts`, `positions`, and `topk` batches.

**FILE / ARTIFACT:**
`src/jlens_reasoning/benchmarks/flenqa/positions.py`,
`src/jlens_reasoning/benchmarks/flenqa/lens.py`,
`src/jlens_reasoning/benchmarks/flenqa/runner.py`

**STATUS:** Verified

**CLAIM:** The paired lens path checks model-logit agreement and model/lens
compatibility rather than treating a successful lens call as sufficient.

**EVIDENCE:** `run_prompt` compares layer keys, vocabulary sizes, shapes, and
model logits; `validate_model_lens` checks residual width and fitted layer
bounds.

**FILE / ARTIFACT:**
`src/jlens_reasoning/benchmarks/flenqa/lens.py`,
`src/jlens_reasoning/experiments_utils/validation.py`

**STATUS:** Verified

**CLAIM:** The project separates raw generation capture from paper-compatible
FLenQA scoring and preserves inference/evaluation metadata.

**EVIDENCE:** `generate_chat` preserves token IDs/pieces, raw text, finish
reason, and inference settings; `evaluate_paper_binary` is called in the
separate accuracy notebook; the normative policy describes the boundary.

**FILE / ARTIFACT:**
`src/jlens_reasoning/inference.py`, `src/jlens_reasoning/evaluation.py`,
`docs/llm-answer-evaluation.md`, `notebooks/flenqa_accuracy.ipynb`

**STATUS:** Verified

**CLAIM:** The probe workflow freezes a 180/60/60 problem-level split and
trains per-layer probes using only 250/500-token train/validation examples.

**EVIDENCE:** The asset notebook splits 300 problem IDs into 60/20/20 train,
validation, and test partitions, extracts final-token residual states, and
fits validation-selected L2 logistic probes. The evaluation notebook asserts
the held-out partition sizes and applies the saved assets without refitting.

**FILE / ARTIFACT:** `notebooks/flenqa_probe_assets.ipynb`,
`notebooks/flenqa_probe_eval.ipynb`

**STATUS:** Verified as an implemented workflow; no saved checkpoint or
held-out output is committed.

**CLAIM:** A frozen probe prediction tests decodability under the probe
contract, not whether the model causally uses that information.

**EVIDENCE:** The evaluation notebook's interpretation explicitly separates
probe accuracy/AUROC from causal intervention, and the probe is fit from
stored residual features without intervening on the model.

**FILE / ARTIFACT:** `notebooks/flenqa_probe_eval.ipynb`

**STATUS:** Conservative inference

### Sanity and interventions

**CLAIM:** The J-Lens sanity experiment uses Qwen3.5-4B and a released
Qwen3.5-4B `qwen-n1000` lens checkpoint, with `spider` → `ant` and
`France` → `China` cases.

**EVIDENCE:** The constants and notebook define the model/lens asset names,
the five cases, alpha 1/2 conditions, and open-model capability gate.

**FILE / ARTIFACT:**
`experiments/jlens_readout_sanity/constants.py`,
`experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`

**STATUS:** Verified as experiment configuration; no current public result
artifact is available.

**CLAIM:** Coordinate interventions are implemented as explicit residual-space
swaps or patches with configurable direction, layer, position, and alpha.

**EVIDENCE:** The intervention utilities construct J-Lens directions from
unembedding rows, compute pseudoinverse coordinates, and install removable
forward hooks; the notebooks configure matched coefficient movement and
concept selection.

**FILE / ARTIFACT:**
`src/jlens_reasoning/experiments_utils/interventions.py`,
`experiments/flenqa_lens_drift/flenqa_lens_intervention.ipynb`,
`experiments/flenqa_lens_drift/flenqa_failure_concept_intervention.ipynb`

**STATUS:** Verified

**CLAIM:** The sanity experiment defines negative controls for identity,
matched-norm random vectors, wrong concepts, and random targets.

**EVIDENCE:** The control suite implements those four control paths and
records deterministic seeds, tolerances, and a non-statistical percentile
interpretation.

**FILE / ARTIFACT:**
`experiments/jlens_readout_sanity/constants.py`,
`experiments/jlens_readout_sanity/controls.py`

**STATUS:** Verified as control implementation; no control result is claimed.

### Deliberately unclaimed findings

**CLAIM:** This repository currently does not establish long-context accuracy
degradation, layer-specific J-Lens drift, probe AUROC trends, drift/error
associations, intervention recovery, or a causal mechanism.

**EVIDENCE:** No committed model-generated artifacts, reports, or executed
notebook outputs exist in the public tree. The notebooks define analyses and
assertions but are saved without outputs.

**FILE / ARTIFACT:** `README.md`, `tests/test_notebooks.py`, tracked repository
tree at the presentation branch base.

**STATUS:** Verified evidence boundary

**CLAIM:** This Qwen setup is not presented as a replication of the
Anthropic/Claude results.

**EVIDENCE:** The repository uses a different open model and lens checkpoint;
the sanity notebook applies an explicit capability gate and reports paper
targets only as diagnostic context. The README states the replication boundary.

**FILE / ARTIFACT:**
`experiments/jlens_readout_sanity/constants.py`,
`experiments/jlens_readout_sanity/reporting.py`, `README.md`

**STATUS:** Verified

**CLAIM:** End-to-end model-backed reproduction cannot be performed from Git
alone.

**EVIDENCE:** Model, lens, FLenQA data, Colab runtime, and generated artifacts
are external to the repository; the scripts and notebooks record paths and
commit markers but do not vendor those assets.

**FILE / ARTIFACT:** `docs/REPRODUCING.md`,
`src/jlens_reasoning/environments/colab.py`, `.gitignore`

**STATUS:** Conservative inference
