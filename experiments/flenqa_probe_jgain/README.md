# Probe J-gain pilot

**Status:** Development completed: 12 problems, 24 matched prompts and 480
condition measurements. The sensitivity intervention works mechanically;
generated answers and accuracy were unchanged at all tested strengths.

## First run: memory limit

The September 27, 2026 run used commit `82f7d8e`, float32 eager attention and an
NVIDIA A100-SXM4-40GB. Asset loading and selection of 12 matched development
problems succeeded. The first prompt was the 3,000-token variant of MonoRel
problem 5399 (`same`, `last`). Its J-gain calculation failed inside the
reverse-over-reverse Jacobian-vector product: the process occupied 38.88 GiB
and could not allocate another 628 MiB.

Colab CLI 0.6.0 continued into the cohort cell after this error; that computation
was interrupted. Notebook gates now prevent later measurement, summary and
freeze cells from proceeding after a failed prerequisite.

The executed failure record is
`artifacts/colab/probe-jgain-82f7d8e/flenqa_probe_jgain_out.ipynb`. The three uploaded files were
retrieved into `artifacts/colab/probe-jgain-82f7d8e/`: `run_metadata.json`,
`selected_pairs.parquet` and `coverage.parquet`. The manifest remains `running`,
so this is an incomplete run, with no saved condition or derivative-check table.
The runtime was terminated. These outputs provide memory-feasibility evidence,
not evidence for or against the intervention hypothesis.

## Memory fix verified

The final-token hidden map now caches the fixed causal prefix and differentiates
a single-token continuation. Its clean target is checked against full prefill;
small-model tests also compare edited targets and derivatives with full-prompt
replay. Earlier source positions retain the original full-prompt path.

On September 27, the same failing prompt (3,203 wrapped tokens) passed on an
A100 40 GB with unchanged inputs, float32 and eager attention. Peak PyTorch GPU
allocation was **18.48 GiB**; map construction, J-gain and both finite-difference
checks took **15.26 seconds**. Sensitivity was 3.21456 and gain norm was 7.21982.
Both checks passed at the existing relative steps 0.001 and 0.003. This verifies
the memory fix on one prompt, not intervention effects or complete cohort coverage.

The executed diagnostic notebook is `artifacts/colab/jgain_memory_check_out.ipynb`;
its four output files are under
`runs/flenqa-probe-jgain/development/memory-check/` in R2 and
`artifacts/colab/probe-jgain-memory-check/` locally. The recorded source bundle
SHA-256 is `dedea1ea4e14f80ea35152e78def208c69b0e355a1a2fa2472022c627d3b140b`.
The full development run below uses the same implementation.

## Completed development pilot

The September 27 A100 run completed with all 480 rows marked `ok`, all 48
finite-difference checks passing, and a peak PyTorch GPU allocation of
**18.63 GiB**. Model, dataset and probe hashes match the memory check, and the
source fingerprint is the same `dedea1ea...` value recorded above. All six
result-file hashes in the completed manifest were verified after retrieval.

At relative strengths 0.001, 0.003 and 0.01:

- `+g` increased sensitivity in all 72 prompt/strength combinations; `-g`
  decreased it in all 72. Each change's magnitude exceeded all three matched
  random-control changes.
- All 360 gain/random measurements preserved probe scores within the configured
  tolerance. Maximum absolute drift was `9.54e-7`; identity controls reproduced
  the clean token sequences.
- No condition changed a generated token sequence. Clean accuracy was **12/12
  short prompts** and **9/12 long prompts**, and remained identical under gain,
  amplification and random interventions. There were no repairs or damage.

This supports mechanical control of the measured sensitivity, with no observed
behavioral effect in this small development sample. It does not establish that
such interventions have no effect generally. The all-zero empirical bootstrap
contrasts reflect unchanged correctness on these 12 problems, not certainty
about a population effect. No strength was frozen, and evaluation has not run.

The executed notebook and verified artifacts are in
`artifacts/colab/probe-jgain-development-20260927/`. R2 outputs remain under
`runs/flenqa-probe-jgain/development/`, with `run_metadata.json` marked
`complete`. The GPU runtime was terminated after upload.

## Question

Can we change how strongly a probe feature affects the model's final hidden
state, while keeping its probe score fixed? Does that change improve or damage
the model's answers?

The probe predicts the True/False task label. Preserving its score does not mean
we have preserved all task information.

## What we change

Start at layer 19 (the existing validation-selected layer, using zero-based
indexing), at the final wrapped prompt token. The target is the final normalized
hidden state at that same token. Keep the direct-chat input contract.

Let `P` project onto the unit probe axis and `h` be the selected hidden state.
`J(h)` is the prompt-specific Jacobian from that state to the target:

```text
d = P @ (h - probe_training_mean)       # existing probe component
s = norm(J(h) @ unit(d))               # final-state sensitivity
g = (I - P) @ gradient_h(0.5 * s**2)    # second-order J-gain direction
```

Freeze `unit(d)` from the clean state when differentiating and when measuring
the intervention. Take one small step along `unit(g)` or its negative. This
preserves the linear probe projection while locally increasing or decreasing
sensitivity. Neither direction uses the gold answer. Report zero directions
explicitly; do not silently drop those cases.

This `s` differs from our existing True-minus-False output-margin derivative.
See the [framework](../../docs/probe_jlens_routing_framework.md) for derivation.

## Small experiment

1. Reuse the frozen chat-v2 probes, model, dataset, and original problem split.
   No probe retraining or lens-readout rerun is required for this plan.
2. Use about 12 development problems to check derivatives against finite
   differences, memory cost, and a few small intervention strengths. Match
   500- and 3,000-token variants, keeping task and rendering settings fixed.
3. Freeze settings before evaluating about 60 held-out problems. Cover all
   three tasks and both labels where available; include baseline successes and
   failures. Keep all variants of a problem in the same split. Report coverage
   and unavailable pairs. Previously inspected cases are debugging examples;
   without an untouched evaluation cohort, call the result exploratory.
4. Apply the intervention once during prompt processing, then generate answers
   with the existing direct-answer settings.

Compare clean/zero intervention, `+g`, `-g`, random directions orthogonal to the
probe (several seeds), and amplification along `unit(d)`. Match perturbation
norms across directions. Amplification follows the existing component, even
when it points toward the wrong label. Choose strengths on development data for
stable sensitivity changes and small perturbations, not maximum answer repair.

## How we measure success

Save one row per prompt, condition, strength, and random seed: probe score,
signed probe projection, sensitivity, perturbation/hidden-state norms, generated
answer, and correctness. Keep problem IDs and input/model/probe provenance.

- **Mechanical success:** `+g` raises sensitivity and `-g` lowers it beyond
  numerical noise and random controls, while the probe score stays fixed within
  a tolerance established during development. Check several small strengths.
- **Behavioral success:** `+g` improves long-context accuracy over clean and
  random controls; wrong-to-correct changes outweigh correct-to-wrong changes.
  Damage under `-g` would strengthen the evidence.
- **Long-context evidence:** effects are stronger on long prompts than their
  matched short versions. Compute paired uncertainty across underlying
  problems, not across variants or random seeds. A small pilot may be inconclusive.

If sensitivity changes but answers do not, our norm may not measure useful
task influence. If random directions work equally well, the effect may be
generic perturbation. If only probe amplification helps, strengthening the
feature works better than this sensitivity intervention. None of these outcomes
alone proves or rules out a general routing mechanism.

## Run the notebook

Use [`flenqa_probe_jgain.ipynb`](flenqa_probe_jgain.ipynb) with the existing
Colab/R2 setup. Commit or stage new source paths before launching: the runner
does not send untracked files.

```bash
COLAB_GPU=A100 ./scripts/run_colab_notebook.sh experiments/flenqa_probe_jgain/flenqa_probe_jgain.ipynb
```

The default `PHASE = "development"` uses 12 validation problems where available,
one seeded matched rendering per problem. The notebook reports every missing
pair and uses all eligible test problems in evaluation. It loads the existing
chat-v2 probes and generates fresh clean answers in float32 with eager attention
for numerical checks. It does not assume these answers match earlier bfloat16
results. The cached-prefix implementation passes memory and derivative checks
across the completed development cohort.
Failures block later measurement and result publication rather than changing
the experiment.

Inspect the first-prompt checks, the full derivative table, probe-score drift,
and random controls. Strengths and tolerances are provisional development
settings. Set `FREEZE_STRENGTH` to one tested strength and run the freeze cell
only after reviewing the mechanical results. Upload those artifacts, then set
`PHASE = "evaluation"` and `FREEZE_STRENGTH = None` in a fresh run. Evaluation
loads the frozen settings and verifies the development files, inputs, and code.
The existing test split has prior descriptive analysis, so results are labeled
exploratory rather than a fresh confirmatory test.

Results live under `runs/flenqa-probe-jgain/{development,evaluation}/`:
selected pairs, population coverage, per-condition answers/measurements,
finite-difference checks, mechanical/behavioral summaries, paired accuracy
intervals, and a run manifest. Partial results are saved after every prompt;
only a manifest with `status: complete` establishes completion. Development can
also write `frozen_settings.json`. Reruns replace files within their phase.
