# FLenQA Probe Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a short Colab notebook that freezes a problem-level FLenQA split and saves per-layer logistic probe assets trained from final-token residual states.

**Architecture:** Keep the workflow in one linear notebook and reuse the existing Colab bootstrap, FLenQA normalization, prompt rendering, artifact paths, and model constants. Persist one split JSON plus one torch checkpoint and JSON metadata sidecar under the configured checkpoints directory; never materialize test activations or calculate test metrics.

**Tech Stack:** Jupyter notebook JSON, Python, PyTorch, Transformers, scikit-learn, pandas, datasets, joblib-free JSON/`torch.save` serialization.

**Spec:** `docs/superpowers/specs/2026-08-24-flenqa-probe-assets-design.md`

## Global Constraints

- Use `global_sample_id` normalized as `FlenqaRow.problem_id` as the underlying problem identifier.
- Split at problem level with fixed seed, 60% train, 20% validation, 20% test, stratified by `(task, label)`.
- Fit only on 250- and 500-token rows from train problems; select regularization only on corresponding validation rows.
- Extract residual states at the final input token immediately before answer generation.
- Use one independent binary L2 logistic probe per transformer layer with positive score meaning `True`.
- Do not use test rows for fitting, model selection, exploratory evaluation, or reported metrics.
- Do not add J-Lens analysis, interventions, shuffled labels, random probes, or test scientific results.
- Keep the canonical first Colab loader cell byte-identical and commit notebooks with empty outputs and null execution counts.
- Do not add model-backed execution to CPU CI.

### Task 1: Create the linear probe asset notebook

**Files:**
- Create: `notebooks/flenqa_probe_assets.ipynb`

**Interfaces:**
- Consumes: `initialize_colab`, `context.datasets_dir`, `context.checkpoints_dir`, `experiments.jlens_readout_sanity.constants.MODEL_NAME`, `MODEL_PATH`, `normalize_rows`, `FlenqaRow.problem_id`, and the local Transformers model/tokenizer convention.
- Produces: `context.checkpoints / "flenqa-probe-assets" / "problem_split.json"`, `probes.pt`, and `metadata.json`.

- [ ] **Step 1: Copy the canonical loader and write the setup cells**

  Use the exact first cell from `notebooks/flenqa_full_run.ipynb`. Add a setup cell that calls `initialize_colab(enable_wandb=False, require_cuda=True)` and imports `json`, `math`, `random`, `Counter`, `Path`, NumPy, pandas, PyTorch, Transformers, `load_from_disk`, `LogisticRegression`, `log_loss`, `accuracy_score`, and `tqdm`. Import `MODEL_NAME` and `MODEL_PATH` from the existing constants module and `normalize_rows` from the FLenQA dataset module.

- [ ] **Step 2: Load and validate FLenQA and construct the problem table**

  Load `context.datasets_dir / "flenqa"`, select `dataset["eval"]` when the object has keys, and call `normalize_rows(raw_rows, full=True)`. Build one record per `problem_id` from the normalized rows and assert that each problem has one task, one label, and all 40 source variants. Confirm the full problem count is 300.

- [ ] **Step 3: Create or load the fixed problem-level split**

  Define `SPLIT_SEED = 1729`, `SPLIT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}`, and `SPLIT_PATH = context.checkpoints / "flenqa-probe-assets" / "problem_split.json"`. Use `sklearn.model_selection.train_test_split` twice on problem records with `stratify=[(task, label)]`, first reserving 40% and then splitting that into equal validation/test halves. Sort IDs before saving. When loading an existing asset, assert matching seed/fractions and reuse its partitions.

  Assert pairwise disjoint partitions, exact union with all normalized problem IDs, one saved partition per problem, and consistency of task/label metadata. Build a `problem_to_split` mapping and assert every normalized source row resolves to the saved partition.

- [ ] **Step 4: Select only train/validation 250+500 prompt rows**

  Filter normalized rows by `ctx_size_declared in {250, 500}` and split membership in `{"train", "validation"}`. Assert that no selected row belongs to test, that both context sizes occur in both fitting partitions, and that each selected problem retains only one task and label. Keep each row as a distinct prompt example so both nominal lengths contribute.

- [ ] **Step 5: Load the model and extract final-token hidden states**

  Load `AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, local_files_only=True).to(context.device)` and its tokenizer, then set evaluation mode. For each selected row, render its exact prompt through `build_prompt_text` using the normalized row fields, tokenize with `return_tensors="pt"`, `truncation=False`, and `max_length=4096`, and assert the final sequence length is positive. Run the model with `output_hidden_states=True` and `use_cache=False` under `torch.inference_mode()`. For each transformer layer index `0..num_hidden_layers-1`, take `outputs.hidden_states[layer_index + 1][0, -1, :]` and move it to CPU float32. Record labels as 0/1 and prompt metadata; assert each layer’s feature matrix is `[num_examples, hidden_dim]`.

- [ ] **Step 6: Fit and validate one centered L2 logistic probe per layer**

  Split the extracted matrices by saved partition, compute the training mean per layer, and center train/validation features by that mean. For each layer and fixed grid `C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)`, fit `LogisticRegression(penalty="l2", solver="lbfgs", C=C, max_iter=2000, random_state=SPLIT_SEED)`. Select by validation `log_loss`, tie-breaking on smaller `C`; calculate train/validation loss and accuracy for the selected model. Convert the learned weight and intercept to float32 tensors, calculate `unit_weight = weight / ||weight||`, assert the norm is positive, and save the selected `C`, mean, weight, bias, unit direction, and metrics. Document that positive score is `True` and negative score is `False`.

- [ ] **Step 7: Save the frozen probe checkpoint and metadata**

  Create `ASSET_DIR = context.checkpoints / "flenqa-probe-assets"`. Save a dictionary with `format_version`, split, layer assets, label convention, context sizes, feature extraction position, model metadata, project commit, and counts to `probes.pt` using `torch.save`. Save JSON-safe split/metrics/config metadata to `metadata.json`. Assert all three files exist and print a concise summary with paths and train/validation counts only.

- [ ] **Step 8: Strip notebook outputs and verify structural requirements**

  Ensure every cell has `execution_count: null` and an empty `outputs` list. Check the notebook contains no J-Lens imports or analysis/intervention calls, no test activation or test metric code, and includes the data-integrity assertions and save paths.

- [ ] **Step 9: Commit the notebook**

  Run `git add notebooks/flenqa_probe_assets.ipynb` and commit with:

  ```bash
  git commit -m "feat: save FLenQA linear probe assets"
  ```

### Task 2: Verify the isolated change

**Files:**
- Test: `tests/test_notebooks.py` (read-only unless the existing notebook discovery contract requires an explicit update)

**Interfaces:**
- Consumes: the committed notebook and existing repository test conventions.
- Produces: verified notebook structure and clean repository checks.

- [ ] **Step 1: Run notebook-focused tests**

  Run `uv run pytest tests/test_notebooks.py -q`. Expected: all existing notebook tests pass; the new notebook is not added to `ALL_NOTEBOOKS` unless the test contract is intentionally expanded for shared notebooks.

- [ ] **Step 2: Run formatting and lint checks**

  Run `uv run ruff format --check .` and `uv run ruff check .`. Expected: no formatting or lint failures.

- [ ] **Step 3: Run lock and CPU test checks**

  Run `uv lock --check` and `uv run pytest -q`. Expected: the lockfile is current and all model-free tests pass.

- [ ] **Step 4: Review the final worktree**

  Run `git status --short --branch` and `git log -2 --oneline`. Expected: only the intended feature-branch commits and no changes in `/Users/noamc/repos/jlens_reasoning`.
