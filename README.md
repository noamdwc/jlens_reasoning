# J-Lens Reasoning

Experimental research code for studying how long-context reasoning changes in
model representations, using Jacobian Lens readouts and controlled residual
interventions.

The project asks: when the same reasoning problem is placed in a longer
context, do internal readouts change—and do candidate changes relate to answer
behavior? The benchmark is [FLenQA](https://aclanthology.org/2024.acl-long.818/).

> Status: research prototype. The implementation and analysis workflows are
> public; model-backed result artifacts are not yet committed. This README does
> not claim that long-context failure has been explained.

## What is built

- Typed FLenQA loading with exact task templates, schema/count checks, prompt
  IDs, deduplication, and source-row provenance.
- Independently tokenized position assets for facts, questions, sampled context,
  and final-prompt positions.
- Paired Jacobian/Logit Lens readouts with typed Parquet schemas and logit,
  layer, vocabulary, and tokenization checks.
- Behavioral scoring, frozen per-layer probes, and residual-space intervention
  hooks with explicit controls.
- Model-free CPU tests and a Colab workflow for model-backed experiments.

## Experiments

| Experiment | What it tests | Current status |
| --- | --- | --- |
| [J-Lens sanity](experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb) | Qwen3.5-4B readouts and controlled concept/fact swaps | Gates and controls implemented; result artifact not committed |
| [FLenQA behavior](notebooks/flenqa_full_run.ipynb) | Behavior across nominal lengths 250–3000 with provenance-preserving scoring | Runner and scorer implemented; no accuracy curve published |
| [Readout drift](experiments/flenqa_lens_drift/flenqa_lens_drift.ipynb) | Position-aware Jacobian/Logit Lens distribution changes | Analysis implemented; no drift result published |
| [Frozen probes](notebooks/flenqa_probe_assets.ipynb) / [evaluation](notebooks/flenqa_probe_eval.ipynb) | Whether answer directions are decodable on held-out problems | Workflow implemented; no probe report published |
| [Intervention pilots](experiments/flenqa_lens_drift/flenqa_lens_intervention.ipynb) / [failure concepts](experiments/flenqa_lens_drift/flenqa_failure_concept_intervention.ipynb) | Whether candidate directions selectively change answer behavior | Mechanics and controls implemented; no causal result claimed |

The intended text-only sanity artifact is:

```text
runs/jlens-readout-sanity/
└── result.json
```

The `Qwen sanity threshold` is an experiment gate, not a result. The paper gap
is explicit: this open-model setup is not a numerical replication of the
Anthropic/Claude experiments.

## Evidence boundary

The repository contains implementation and workflow contracts, not a result
release. In particular, it does not currently establish:

- long-context accuracy degradation;
- a causal layer, token, or representation explaining failure;
- reasoning recovery from an intervention; or
- probe decodability as evidence of causal model use.

Every tracked notebook is saved without outputs, and generated model outputs,
weights, lens checkpoints, and reports remain outside Git.

## Quick start

```bash
uv sync --locked --extra experiment
uv run pytest
uv run ruff format --check .
uv run ruff check .
```

For Colab setup, asset download, run order, artifact contracts, and the
reproducibility boundary, see [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

## Code map

- `src/jlens_reasoning/` — reusable data, inference, evaluation, lens, and
  intervention code.
- [`src/jlens_reasoning/probing/`](docs/probing.md) — shared probe extraction,
  fitting, scoring, and artifacts.
- [`src/jlens_reasoning/probe_jlens.py`](docs/probe_jlens.md) — combined probe/J-Lens
  transport, sensitivities, and future routing-framework implementation.
- `experiments/` — J-Lens sanity, drift, and intervention notebooks.
- `notebooks/` — FLenQA drivers, scoring, and frozen probe workflows.
- `tests/` — CPU-only unit and notebook-structure tests.

## Attribution

- [Anthropic's Jacobian Lens implementation](https://github.com/anthropics/jacobian-lens)
  provides the upstream method and package.
- [FLenQA / “Same Task, More Tokens”](https://aclanthology.org/2024.acl-long.818/)
  provides the benchmark and research basis.
- The [Qwen3.5-4B model](https://huggingface.co/Qwen/Qwen3.5-4B) and released
  lens checkpoint are external assets.
