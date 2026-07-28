# CLAUDE.md

Research tooling for applying the [Jacobian Lens](https://github.com/anthropics/jacobian-lens)
(`jlens`, pinned to a git rev in `pyproject.toml`) to reasoning benchmarks.

## Commands

```bash
uv sync --locked --extra experiment   # install; CI uses the same flags
uv run pytest                         # all tests (CPU-only, model-free)
uv run pytest tests/experiments/jlens_readout_sanity/test_experiment.py -v
uv run ruff format .                  # CI checks with --check
uv run ruff check .
uv lock --check                       # CI verifies the lockfile is current
```

Run all four (format, lint, tests, lock check) before committing — CI runs them
on ubuntu 3.11/3.12 and macOS 3.11.

## Execution model

Three environments, deliberately separated:

- **Mac** — development, tests, small CPU/MPS work.
- **Colab (GPU)** — every model-backed experiment. Colab is intentionally *not*
  part of CI: no workflow ever launches a runtime, and nothing in `tests/`
  imports `google.colab`. It *is* driven from `scripts/` (see below) — the
  README's blanket "not part of scripts or CI" wording predates
  `experiment_colab_run.sh` and is stale.
- **GitHub Actions** — secret-free CPU tests. Never touches HF, W&B, or Drive.

Colab does not install from git. `./scripts/upload_colab_wheel.sh` builds the
project wheel plus exported locked requirements and a commit marker, and uploads
them to `data/jlens-reasoning/wheels` on the rclone remote; the notebook loader
cell installs from there. **Re-run the uploader after any code or dependency
change**, otherwise Colab silently runs stale code.
`./scripts/experiment_colab_run.sh` chains upload + notebook run.

## Layout

```text
src/jlens_reasoning/          # reusable library
  config.py                   # ArtifactPaths, JLENS_REAS_ARTIFACT_ROOT
  runtime.py                  # device selection
  environments/               # initialize_colab, RuntimeContext
  evaluation.py               # answer grading state machine (see policy below)
  evaluation_utils.py         # extraction / normalization / rank primitives
  experiments_utils/          # generic mechanics: tokens, interventions,
                              # controls, artifacts, validation
experiments/<name>/           # one self-contained package + its notebook
notebooks/                    # shared bootstrap + environment check only
docs/superpowers/{specs,plans}/  # design docs and implementation plans
```

`experiments/` is a top-level importable package (setuptools discovers both
`src` and `.`). Adding an experiment means: new package, new notebook, mirrored
tests under `tests/experiments/<name>/`. There is **no registry, plugin
protocol, or import-time discovery** — nothing in `src/` or `experiments/` needs
editing to add one. Do not introduce a registry.

The one place a new experiment must touch shared code is
`tests/test_notebooks.py`. It *collects* notebooks by globbing
`experiments/*/*.ipynb`, but `test_experiment_notebooks_are_discovered_without_a_registry`
then asserts that glob equals an exact hardcoded list — so adding a notebook
fails that test until the list is updated. That assertion is a deliberate
tripwire (it also guards against the old `notebooks/01_*.ipynb` layout coming
back), not an oversight. Update it; don't delete it, and don't add a registry to
avoid it.

Split of responsibility: generic, reusable mechanics go in
`jlens_reasoning.experiments_utils`; experiment policy, thresholds, result
assembly, and reporting stay local to the owning experiment package
(`constants.py`, `experiment.py`, `reporting.py`, `utils.py` facade).

## Conventions

- Imports go at the top of the file. Function-local imports are allowed only for
  the two reasons already present in the codebase, and each one carries the
  narrowest possible scope:
  - **Optional / environment-only dependencies** that must not be imported on
    Mac or in CI — `import wandb` in `tracking.py:23`, `from google.colab
    import ...` in `environments/colab.py:20,26`.
  - **Breaking a genuine import cycle** — `experiment.py:497` imports from
    `controls.py`, which imports from `experiment.py` at module level.

  Anything else goes at the top. If a new local import is neither of these,
  restructure instead.
- New modules start with `from __future__ import annotations`.
- ruff: line-length 88, rules `["B", "E", "F", "I", "UP", "W"]` (E501 ignored).
- Prefer `@dataclass(frozen=True, slots=True)` for results and configs, and
  `StrEnum` for statuses. Results are serialized via
  `experiments_utils/artifacts.py::write_results` (stable, sorted JSON).
- Tests are CPU-only and model-free — use the fake-tokenizer pattern in
  `tests/experiments_utils/test_tokens.py`. No credentials of any kind.
- Notebooks are committed with **no outputs and no execution counts**; the Drive
  loader cell must stay byte-identical across notebooks (`test_notebooks.py`
  enforces both). Keep the workflow visible in cells rather than hiding it
  behind one opaque call; experiment cases are defined in the notebook itself.
- Artifacts and data are never committed. Everything writes under
  `JLENS_REAS_ARTIFACT_ROOT` (default `./artifacts`, `/content/drive/MyDrive/jlens-reasoning`
  on Colab), laid out as `datasets/ cache/huggingface/ lenses/ checkpoints/ runs/`.
- W&B is on by default in Colab and fails loudly; pass `enable_wandb=False` for
  experiments that don't track.

## Answer evaluation

`docs/llm-answer-evaluation.md` is **normative policy**, and
`src/jlens_reasoning/evaluation.py` + `evaluation_utils.py` are its code
implementation. **All LLM answer evaluation goes through these two modules** —
never write ad-hoc grading inside an experiment package or notebook. Extend the
shared modules (and the policy doc) instead.

- `evaluation_utils.py` — the separable primitives the policy requires:
  `parse_think_tags` / `no_reasoning`, `extract_answer` (gold-blind),
  `normalize_text`, `match_reference`, `safe_truncated_text`, plus rank helpers
  (`best_token_rank`, `top_token_values`, `log_rank_gain`,
  `answer_token_variants`).
- `evaluation.py` — the typed contracts over them: `ModelOutput`,
  `EvaluationResult`, `GenerationStatus` / `AnswerStatus` / `ReasoningStatus`,
  the `FactualEvaluator` protocol with `SimpleFactualEvaluator`, and the
  `evaluate` / `evaluate_next_token` / `compare_token_ranks` entry points.

Core rules the code enforces: preserve the raw output; isolate the visible
answer via declared reasoning delimiters only; extract the front-loaded answer
*without* passing it the references; then compare against predefined references
under a fixed normalization. Searching output for the gold answer is
prohibited. Generation, reasoning, and answer statuses are recorded separately;
errors produce `not_graded`, not `incorrect`. Every real artifact failure
becomes a regression test in `tests/test_evaluation.py` before behavior changes.

The policy is versioned and currently scoped to short factual sanity answers.
New answer shapes (FLenQA, multiple choice, numeric, open-ended, model judges)
need a new policy version, not an inferred extension of v1.

## Working practice

Features are designed in `docs/superpowers/specs/` and implemented from a
task-by-task plan in `docs/superpowers/plans/` (dated filenames). Read the
matching spec before implementing a plan. Work happens on feature branches —
never commit to `main`.
