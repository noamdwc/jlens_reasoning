import math
import re
import string
from dataclasses import asdict
from pathlib import Path

import nbformat
import pandas as pd
import pytest

SHARED_NOTEBOOKS = [
    Path("notebooks/_template.ipynb"),
    Path("notebooks/00_environment_check.ipynb"),
]
FLENQA_BENCHMARK_NOTEBOOKS = [
    Path("notebooks/flenqa_smoke.ipynb"),
    Path("notebooks/flenqa_full_run.ipynb"),
]
FLENQA_ACCURACY_NOTEBOOK = Path("notebooks/flenqa_accuracy.ipynb")
FLENQA_PROBE_JLENS_NOTEBOOK = Path(
    "experiments/flenqa_probe_jlens/flenqa_probe_jlens.ipynb"
)
FLENQA_LENS_DRIFT_NOTEBOOK = Path(
    "experiments/flenqa_lens_drift/flenqa_lens_drift.ipynb"
)
FLENQA_LENS_INTERVENTION_NOTEBOOK = Path(
    "experiments/flenqa_lens_drift/flenqa_lens_intervention.ipynb"
)
FLENQA_NOTEBOOKS = [
    *FLENQA_BENCHMARK_NOTEBOOKS,
    FLENQA_ACCURACY_NOTEBOOK,
]
EXPERIMENT_NOTEBOOKS = sorted(Path("experiments").glob("*/*.ipynb"))
NOTEBOOKS = [*SHARED_NOTEBOOKS, *FLENQA_NOTEBOOKS, *EXPERIMENT_NOTEBOOKS]
ASSET_NOTEBOOK = Path("notebooks/01_download_assets.ipynb")
ALL_NOTEBOOKS = [*NOTEBOOKS, ASSET_NOTEBOOK]


def load_notebook(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(path, as_version=4)


def notebook_cells_by_id(path: Path) -> dict[str, str]:
    notebook = load_notebook(path)
    return {cell.id: cell.source for cell in notebook.cells}


def test_flenqa_full_run_requests_top_250_tokens() -> None:
    source = notebook_cells_by_id(Path("notebooks/flenqa_full_run.ipynb"))[
        "run-benchmark"
    ]

    assert "top_k=250" in source


def execute_notebook_case_cell() -> dict[str, object]:
    path = Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb")
    source = notebook_cells_by_id(path)["define-cases"]
    namespace: dict[str, object] = {}
    exec(compile(source, f"{path}:define-cases", "exec"), namespace)
    return namespace


def test_notebooks_have_no_saved_outputs_or_execution_counts() -> None:
    for path in ALL_NOTEBOOKS:
        notebook = load_notebook(path)
        for cell in notebook.cells:
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []


def test_notebooks_share_one_canonical_drive_loader_cell() -> None:
    recurring_paths = [
        Path("notebooks/_template.ipynb"),
        *FLENQA_NOTEBOOKS,
        *EXPERIMENT_NOTEBOOKS,
    ]
    recurring_loaders = [
        load_notebook(path).cells[0].source for path in recurring_paths
    ]
    loader = recurring_loaders[0]
    environment_check_loader = (
        load_notebook(Path("notebooks/00_environment_check.ipynb")).cells[0].source
    )

    assert all(candidate == loader for candidate in recurring_loaders)
    assert environment_check_loader == loader.replace(
        "%pip install -qq ", "%pip install "
    )
    assert 'drive.mount("/content/drive")' in loader
    assert "/content/drive/MyDrive/data/jlens-reasoning/wheels" in loader
    assert "requirements-colab.txt" in loader
    assert "project-commit.txt" in loader
    assert "project-dirty.txt" in loader
    assert "PROJECT_COMMIT" in loader
    assert "PROJECT_WORKING_TREE_DIRTY" in loader
    assert 'glob("jlens_reasoning-*.whl")' in loader
    assert loader.count("%pip install -qq") == 2
    assert environment_check_loader.count("%pip install") == 2
    assert "%pip install -q" not in environment_check_loader
    assert "--requirement" in loader
    assert "--no-deps" in loader
    assert "subprocess.run" not in loader
    assert "sys.executable" not in loader
    assert "GITHUB_TOKEN_JLENS_REAS" not in loader
    assert "scripts/colab_bootstrap.py" not in loader
    assert "PROJECT_REF" not in loader
    assert not Path("scripts/colab_bootstrap.py").exists()


def test_asset_notebook_installs_dependencies_very_quietly() -> None:
    source = "\n".join(cell.source for cell in load_notebook(ASSET_NOTEBOOK).cells)

    assert source.count("%pip install -qq") == 1


def test_notebooks_do_not_contain_credentials() -> None:
    forbidden_fragments = ("github_pat_", "ghp_", "wandb-secret")

    for path in ALL_NOTEBOOKS:
        source = path.read_text(encoding="utf-8")
        assert not any(fragment in source for fragment in forbidden_fragments)
        assert re.search(r"hf_[A-Za-z0-9]{20,}", source) is None


def test_notebooks_use_the_colab_environment_module() -> None:
    for path in NOTEBOOKS:
        notebook = load_notebook(path)
        source = "\n".join(cell.source for cell in notebook.cells)

        assert (
            "from jlens_reasoning.environments.colab import initialize_colab" in source
        )
        assert "context = initialize_colab(" in source
        assert "PROJECT_DIR" not in source
        assert "rev-parse" not in source


def test_experiment_notebooks_exclude_flenqa_benchmark_drivers() -> None:
    assert EXPERIMENT_NOTEBOOKS == [
        FLENQA_PROBE_JLENS_NOTEBOOK,
        FLENQA_LENS_DRIFT_NOTEBOOK,
        FLENQA_LENS_INTERVENTION_NOTEBOOK,
        Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb"),
    ]
    assert not Path("notebooks/01_jlens_readout_sanity.ipynb").exists()


def test_flenqa_probe_jlens_notebook_connects_probe_scores_to_jlens() -> None:
    source = "\n".join(
        cell.source for cell in load_notebook(FLENQA_PROBE_JLENS_NOTEBOOK).cells
    )

    for required in (
        "flenqa-probe-assets",
        "flenqa-full-run",
        "JacobianLens.from_pretrained",
        "gold_probe_score",
        "propagation_norm",
        "answer_effect",
        "model_correct",
        "2000",
        "3000",
        "axhline(0",
        "groupby([\"layer\", \"model_correct\"]",
    ):
        assert required in source
    assert "LensCoordinatePatcher" not in source
    assert "coordinate_patch(" not in source
    assert "causal_lm.generate(" not in source


def test_flenqa_notebooks_are_benchmark_drivers() -> None:
    forbidden = (
        "run_preflight(",
        "score_binary_answer(",
        "select_summary_positions(",
        "reduce_readout(",
        "ParquetWriter",
        "TABLE_SCHEMAS",
    )

    for path in FLENQA_BENCHMARK_NOTEBOOKS:
        source = "\n".join(cell.source for cell in load_notebook(path).cells)
        assert "from jlens_reasoning.benchmarks.flenqa.lens import" in source
        assert "from jlens_reasoning.benchmarks.flenqa.runner import" in source
        assert "run_benchmark(" in source
        assert "model_name=" not in source
        assert "lens_revision=" not in source
        assert "tokenizer_name=" not in source
        assert "code_revision=" not in source
        assert not any(fragment in source for fragment in forbidden)


def test_flenqa_benchmark_notebooks_select_the_published_eval_split() -> None:
    for path in FLENQA_BENCHMARK_NOTEBOOKS:
        source = "\n".join(cell.source for cell in load_notebook(path).cells)

        assert 'dataset["eval"]' in source
        assert 'dataset["train"]' not in source


def test_flenqa_full_run_saves_model_outputs_for_accuracy_scoring() -> None:
    notebook = load_notebook(Path("notebooks/flenqa_full_run.ipynb"))
    cells = notebook_cells_by_id(Path("notebooks/flenqa_full_run.ipynb"))
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "prepare_prompts(rows)" in cells["save-model-outputs"]
    assert "generate_chat(" in cells["save-model-outputs"]
    assert '"model_outputs.parquet"' in cells["save-model-outputs"]
    assert "generated_token_ids" in cells["save-model-outputs"]
    assert "generated_text" in cells["save-model-outputs"]
    assert "pq.write_table" in cells["save-model-outputs"]
    assert "evaluate_paper_binary" not in source


def test_flenqa_accuracy_notebook_scores_saved_model_outputs() -> None:
    notebook = load_notebook(FLENQA_ACCURACY_NOTEBOOK)
    cells = notebook_cells_by_id(FLENQA_ACCURACY_NOTEBOOK)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "initialize_colab(enable_wandb=False, require_cuda=True)" in source
    assert '"flenqa-full-run" / "model_outputs.parquet"' in source
    assert "pq.read_table(MODEL_OUTPUT_PATH)" in cells["load-model-outputs"]
    assert "evaluate_paper_binary" in cells["run-accuracy"]
    assert "generate_chat(" not in source
    assert "transformers" not in source
    assert "load_from_disk" not in source
    assert "causal_lm.generate(" not in source
    assert "generated_text" in cells["load-model-outputs"]
    assert "append_column" in cells["run-accuracy"]
    assert "pq.write_table" in cells["save-results"]
    assert '"results.parquet"' in cells["save-results"]
    assert "weighted_correct" in cells["paper-curve"]
    assert ".groupby(" in cells["paper-curve"]
    assert ".groupby(" in cells["unique-curve"]
    assert "run_accuracy(" not in source
    assert "load_accuracy_results(" not in source
    assert "run-manifest" not in source
    assert "shard" not in source.casefold()
    assert "run_benchmark(" not in source
    assert "JacobianLens" not in source


def test_flenqa_lens_drift_summarizes_final_position_token_prominence() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)["summarize-tokens"]
    namespace = {"pd": pd}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:summarize-tokens", "exec"),
        namespace,
    )

    prompt_info = pd.DataFrame(
        {
            "prompt_id": ["short", "long"],
            "ctx_size": [250, 1000],
            "final_position": [9, 19],
        }
    )
    topk = pd.DataFrame(
        {
            "prompt_id": ["short", "short", "short", "long", "long"],
            "lens_kind": ["jacobian"] * 5,
            "layer": [0] * 5,
            "position": [9, 9, 3, 19, 19],
            "rank": [1, 2, 1, 1, 2],
            "token_id": [10, 20, 99, 30, 10],
            "logit": [5.0, 4.0, 100.0, 6.0, 3.0],
        }
    )

    summary, seen_prompt_ids = namespace["summarize_tokens"](topk, prompt_info)
    actual = {
        (row.ctx_size, row.token_id): (
            row.appearances,
            row.rank_score,
            row.logit_sum,
        )
        for row in summary.itertuples()
    }
    assert actual == {
        (250, 10): (1, 1.0, 5.0),
        (250, 20): (1, 0.5, 4.0),
        (1000, 10): (1, 0.5, 3.0),
        (1000, 30): (1, 1.0, 6.0),
    }
    assert seen_prompt_ids == {"short", "long"}


def test_flenqa_lens_drift_rejects_unknown_topk_prompt_ids() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)["summarize-tokens"]
    namespace = {"pd": pd}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:summarize-tokens", "exec"),
        namespace,
    )

    prompt_info = pd.DataFrame(
        {"prompt_id": ["known"], "ctx_size": [250], "final_position": [9]}
    )
    topk = pd.DataFrame(
        {
            "prompt_id": ["unknown"],
            "lens_kind": ["jacobian"],
            "layer": [0],
            "position": [9],
            "rank": [1],
            "token_id": [10],
            "logit": [5.0],
        }
    )

    with pytest.raises(ValueError, match="top-k rows reference unknown prompt IDs"):
        namespace["summarize_tokens"](topk, prompt_info)


def test_flenqa_lens_drift_summarizes_each_labeled_position_by_problem() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)[
        "summarize-position-tokens"
    ]
    namespace = {"pd": pd}
    exec(
        compile(
            source,
            f"{FLENQA_LENS_DRIFT_NOTEBOOK}:summarize-position-tokens",
            "exec",
        ),
        namespace,
    )

    position_info = pd.DataFrame(
        {
            "prompt_id": ["short", "short", "long"],
            "problem_id": [7, 7, 7],
            "ctx_size": [250, 250, 1000],
            "position": [3, 9, 19],
            "position_label": ["fact_a_end", "final_prompt", "final_prompt"],
        }
    )
    topk = pd.DataFrame(
        {
            "prompt_id": ["short", "short", "long"],
            "lens_kind": ["jacobian"] * 3,
            "layer": [2] * 3,
            "position": [3, 9, 19],
            "rank": [1, 2, 1],
            "token_id": [10, 20, 30],
            "logit": [6.0, 4.0, 8.0],
        }
    )

    summary, seen_prompt_ids = namespace["summarize_position_tokens"](
        topk, position_info
    )
    actual = {
        (row.ctx_size, row.position_label, row.token_id): (
            row.problem_id,
            row.appearances,
            row.rank_score,
            row.logit_sum,
        )
        for row in summary.itertuples(index=False)
    }
    assert actual == {
        (250, "fact_a_end", 10): (7, 1, 1.0, 6.0),
        (250, "final_prompt", 20): (7, 1, 0.5, 4.0),
        (1000, "final_prompt", 30): (7, 1, 1.0, 8.0),
    }
    assert seen_prompt_ids == {"short", "long"}

    prompt_summary, _ = namespace["summarize_position_tokens"](
        topk, position_info, retain_prompt=True
    )
    assert set(prompt_summary["prompt_id"]) == {"short", "long"}


def test_flenqa_lens_drift_measures_token_distribution_change() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)[
        "measure-distribution-drift"
    ]
    namespace = {"pd": pd}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:measure-drift", "exec"),
        namespace,
    )

    token_stats = pd.DataFrame(
        {
            "lens_kind": ["jacobian"] * 5,
            "layer": [4] * 5,
            "ctx_size": [250, 250, 1000, 1000, 1000],
            "token_id": [10, 20, 10, 20, 30],
            "rank_score": [3.0, 1.0, 1.0, 1.0, 2.0],
        }
    )

    drift = namespace["measure_distribution_drift"](token_stats)
    actual = {
        row.ctx_size: row.total_variation for row in drift.itertuples(index=False)
    }
    assert actual == {250: 0.0, 1000: 0.5}


def test_flenqa_lens_drift_measures_problem_matched_position_drift() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)["measure-grouped-drift"]
    namespace = {"pd": pd}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:grouped-drift", "exec"),
        namespace,
    )

    token_stats = pd.DataFrame(
        {
            "lens_kind": ["jacobian"] * 4,
            "layer": [4] * 4,
            "position_label": ["question_end"] * 4,
            "problem_id": [7, 7, 8, 8],
            "ctx_size": [250, 1000, 250, 1000],
            "token_id": [10, 10, 10, 20],
            "rank_score": [1.0, 1.0, 1.0, 1.0],
        }
    )

    drift = namespace["measure_grouped_distribution_drift"](
        token_stats,
        group_keys=["lens_kind", "layer", "position_label", "problem_id"],
    )
    actual = {
        (row.problem_id, row.ctx_size): row.total_variation
        for row in drift.itertuples(index=False)
    }
    assert actual == {
        (7, 250): 0.0,
        (7, 1000): 0.0,
        (8, 250): 0.0,
        (8, 1000): 1.0,
    }

    incomplete = token_stats[
        ~((token_stats["problem_id"] == 8) & (token_stats["ctx_size"] == 1000))
    ]
    with pytest.raises(ValueError, match="every drift group must contain every length"):
        namespace["measure_grouped_distribution_drift"](
            incomplete,
            group_keys=["lens_kind", "layer", "position_label", "problem_id"],
            expected_ctx_sizes=[250, 1000],
        )


def test_flenqa_lens_drift_reports_sparse_position_groups() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)["measure-grouped-drift"]
    namespace = {"pd": pd}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:sparse-grouped-drift", "exec"),
        namespace,
    )

    token_stats = pd.DataFrame(
        {
            "lens_kind": ["jacobian"] * 4,
            "layer": [4] * 4,
            "position_label": [
                "question_end",
                "question_end",
                "sampled_padding",
                "sampled_padding",
            ],
            "ctx_size": [250, 1000, 1000, 2000],
            "token_id": [10, 20, 30, 40],
            "rank_score": [1.0, 1.0, 1.0, 1.0],
        }
    )

    drift = namespace["measure_grouped_distribution_drift"](
        token_stats,
        group_keys=["lens_kind", "layer", "position_label"],
        expected_ctx_sizes=[250, 1000, 2000],
        require_complete=False,
    )
    actual = {
        (row.position_label, row.ctx_size): (
            row.baseline_ctx_size,
            row.total_variation,
        )
        for row in drift.itertuples(index=False)
    }
    assert actual == {
        ("question_end", 250): (250, 0.0),
        ("question_end", 1000): (250, 1.0),
        ("sampled_padding", 1000): (1000, 0.0),
        ("sampled_padding", 2000): (1000, 1.0),
    }


def test_flenqa_lens_drift_measures_each_prompt_against_problem_baseline() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)["measure-prompt-drift"]
    namespace = {"pd": pd}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:prompt-drift", "exec"),
        namespace,
    )

    baseline = pd.DataFrame(
        {
            "problem_id": [7],
            "lens_kind": ["jacobian"],
            "layer": [4],
            "position_label": ["question_end"],
            "token_id": [10],
            "rank_score": [1.0],
        }
    )
    prompts = pd.DataFrame(
        {
            "prompt_id": ["same", "changed"],
            "problem_id": [7, 7],
            "ctx_size": [1000, 1000],
            "lens_kind": ["jacobian", "jacobian"],
            "layer": [4, 4],
            "position_label": ["question_end", "question_end"],
            "token_id": [10, 20],
            "rank_score": [1.0, 1.0],
        }
    )

    drift = namespace["measure_prompt_distribution_drift"](prompts, baseline)
    actual = {
        row.prompt_id: row.total_variation for row in drift.itertuples(index=False)
    }
    assert actual == {"changed": 1.0, "same": 0.0}

    with pytest.raises(ValueError, match="prompt components are missing baselines"):
        namespace["measure_prompt_distribution_drift"](
            prompts.assign(problem_id=8), baseline
        )

    mixed_prompts = pd.concat(
        [prompts.iloc[[0]], prompts.iloc[[1]].assign(problem_id=8)],
        ignore_index=True,
    )
    covered_only = namespace["measure_prompt_distribution_drift"](
        mixed_prompts,
        baseline,
        require_complete=False,
    )
    assert covered_only["prompt_id"].tolist() == ["same"]
    assert covered_only["total_variation"].tolist() == [0.0]


def test_flenqa_lens_drift_identifies_only_answer_interface_tokens() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)["surface-token-policy"]
    namespace = {"string": string}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:surface-policy", "exec"),
        namespace,
    )

    is_answer_surface_token = namespace["is_answer_surface_token"]
    assert is_answer_surface_token(" True")
    assert is_answer_surface_token("FALSE")
    assert is_answer_surface_token(",True")
    assert is_answer_surface_token("<think>")
    assert is_answer_surface_token("**")
    assert is_answer_surface_token("   ")
    assert not is_answer_surface_token("Rule")
    assert not is_answer_surface_token("evidence")
    assert not is_answer_surface_token("123")


def test_flenqa_lens_drift_persistent_score_uses_middle_semantic_positions() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)[
        "persistent-semantic-score"
    ]
    namespace = {"math": math}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:persistent-score", "exec"),
        namespace,
    )

    problem_drift = pd.DataFrame(
        {
            "lens_kind": ["jacobian"] * 7,
            "problem_id": [7] * 7,
            "ctx_size": [250, 250, 1000, 1000, 1000, 1000, 1000],
            "layer": [1, 2, 0, 1, 1, 2, 3],
            "position_label": [
                "fact_a_end",
                "question_end",
                "fact_a_end",
                "fact_a_end",
                "final_prompt",
                "question_end",
                "question_end",
            ],
            "total_variation": [0.0, 0.0, 0.9, 0.2, 1.0, 0.4, 0.9],
        }
    )

    persistent = namespace["summarize_persistent_semantic_drift"](problem_drift)
    actual = persistent.set_index("ctx_size")
    assert actual.loc[250, "persistent_drift"] == 0.0
    assert actual.loc[1000, "persistent_drift"] == pytest.approx(0.3)
    assert actual["component_count"].to_dict() == {250: 2, 1000: 2}
    assert persistent["layer_min"].unique().tolist() == [1]
    assert persistent["layer_max"].unique().tolist() == [2]

    layer_two_only = namespace["summarize_persistent_semantic_drift"](
        problem_drift, layer_min=2, layer_max=2
    ).set_index("ctx_size")
    assert layer_two_only.loc[250, "persistent_drift"] == 0.0
    assert layer_two_only.loc[1000, "persistent_drift"] == pytest.approx(0.4)


def test_flenqa_lens_drift_associates_matched_drift_with_error_rate() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)["drift-error-association"]
    namespace = {"pd": pd}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:drift-error", "exec"),
        namespace,
    )

    persistent = pd.DataFrame(
        {
            "lens_kind": ["jacobian"] * 6,
            "problem_id": [1, 2, 3, 1, 2, 3],
            "ctx_size": [250, 250, 250, 1000, 1000, 1000],
            "persistent_drift": [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
        }
    )
    problem_accuracy = pd.DataFrame(
        {
            "problem_id": [1, 2, 3, 1, 2, 3],
            "ctx_size": [250, 250, 250, 1000, 1000, 1000],
            "accuracy": [1.0, 1.0, 1.0, 1.0, 0.5, 0.0],
        }
    )

    association = namespace["measure_drift_error_association"](
        persistent, problem_accuracy
    )
    row = association[association["ctx_size"] == 1000].iloc[0]
    assert row.lens_kind == "jacobian"
    assert row.ctx_size == 1000
    assert row.problem_count == 3
    assert row.spearman_drift_vs_error == pytest.approx(1.0)
    assert row.spearman_drift_vs_accuracy_loss == pytest.approx(1.0)


def test_flenqa_lens_drift_rejects_mismatched_prompt_assets() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)["check-prompt-ids"]
    namespace: dict[str, object] = {}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:check-prompt-ids", "exec"),
        namespace,
    )

    with pytest.raises(
        ValueError, match="accuracy has 1 missing and 1 extra prompt IDs"
    ):
        namespace["require_same_prompt_ids"](
            ["a", "b"],
            accuracy=["b", "c"],
        )


def test_flenqa_lens_drift_rejects_missing_labeled_positions() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)["check-prompt-ids"]
    namespace: dict[str, object] = {}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:check-position-keys", "exec"),
        namespace,
    )

    with pytest.raises(
        ValueError, match="top-k positions has 1 missing and 1 extra labeled positions"
    ):
        namespace["require_same_position_keys"](
            [("a", 3), ("b", 7)],
            [("b", 7), ("c", 9)],
            name="top-k positions",
        )

    with pytest.raises(
        ValueError, match="surface ablation has 1 missing and 0 extra prompt components"
    ):
        namespace["require_same_position_keys"](
            [("prompt", "question")],
            [],
            name="surface ablation",
            unit="prompt components",
        )


def test_flenqa_lens_drift_combines_streamed_token_summaries() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)["combine-token-summaries"]
    namespace = {"pd": pd}
    exec(compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:combine", "exec"), namespace)

    first = pd.DataFrame(
        {
            "lens_kind": ["jacobian"],
            "layer": [2],
            "ctx_size": [250],
            "token_id": [10],
            "appearances": [2],
            "rank_score": [1.5],
            "logit_sum": [7.0],
        }
    )
    second = first.assign(appearances=3, rank_score=2.0, logit_sum=11.0)

    combined = namespace["combine_token_summaries"]([first, second])
    row = combined.iloc[0]
    assert (row.appearances, row.rank_score, row.logit_sum) == (5, 3.5, 18.0)


def test_flenqa_lens_drift_combines_position_summaries_by_problem() -> None:
    source = notebook_cells_by_id(FLENQA_LENS_DRIFT_NOTEBOOK)[
        "combine-position-summaries"
    ]
    namespace = {"pd": pd}
    exec(
        compile(source, f"{FLENQA_LENS_DRIFT_NOTEBOOK}:combine-positions", "exec"),
        namespace,
    )

    first = pd.DataFrame(
        {
            "lens_kind": ["jacobian", "jacobian"],
            "layer": [2, 2],
            "position_label": ["question_end", "question_end"],
            "ctx_size": [250, 250],
            "problem_id": [7, 8],
            "token_id": [10, 10],
            "appearances": [2, 5],
            "rank_score": [1.5, 4.0],
            "logit_sum": [7.0, 20.0],
        }
    )
    second = first.iloc[[0]].assign(appearances=3, rank_score=2.0, logit_sum=11.0)

    combined = namespace["combine_position_summaries"]([first, second])
    actual = {
        row.problem_id: (row.appearances, row.rank_score, row.logit_sum)
        for row in combined.itertuples(index=False)
    }
    assert actual == {7: (5, 3.5, 18.0), 8: (5, 4.0, 20.0)}

    aggregate = namespace["combine_position_summaries"](
        [first, second], retain_problem=False
    )
    row = aggregate.iloc[0]
    assert (row.appearances, row.rank_score, row.logit_sum) == (10, 7.5, 38.0)

    prompt_first = first.assign(prompt_id=["a", "b"])
    prompt_second = second.assign(prompt_id="a")
    by_prompt = namespace["combine_position_summaries"](
        [prompt_first, prompt_second], retain_prompt=True
    )
    assert set(by_prompt["prompt_id"]) == {"a", "b"}
    prompt_a = by_prompt[by_prompt["prompt_id"] == "a"].iloc[0]
    assert (prompt_a.appearances, prompt_a.rank_score, prompt_a.logit_sum) == (
        5,
        3.5,
        18.0,
    )


def test_readout_sanity_notebook_has_pinned_gpu_workflow() -> None:
    path = Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb")
    notebook = load_notebook(path)
    cells_by_id = notebook_cells_by_id(path)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "initialize_colab(enable_wandb=False, require_cuda=True)" in source
    assert 'MODEL_NAME = "Qwen/Qwen3.5-4B"' not in source
    assert "from experiments.jlens_readout_sanity.constants import" in source
    assert "from experiments.jlens_readout_sanity.utils import" in source
    assert "Path(MODEL_PATH)" in source
    assert "Path(LENS_PATH)" in source
    assert "local_files_only=True" in source
    assert "JacobianLens.from_pretrained" in source
    assert "LENS_REPO" not in source
    assert "run_experiment" in source
    assert "write_results" in source
    assert "render_sanity_report" in source
    assert "jlens.vis" not in source
    assert "compute_slice" not in source
    assert "build_page" not in source
    assert "notebook_iframe" not in source
    assert ".html" not in source
    assert "write_text(page" not in source
    assert "raise RuntimeError" in source
    assert "forward_next_token" in source
    assert "get_output_embeddings().weight" in source
    assert "causal_lm.generate" in source
    assert "ModelOutput" in source
    assert "GenerationStatus" in source
    assert "max_new_tokens=64" in source
    assert "READOUT_CASES" not in cells_by_id["load-model-and-lens"]
    assert "CASES = (" in cells_by_id["define-cases"]
    assert "READOUT_CASES" not in source
    assert "SWAP_CASES" not in source
    assert "generate_output" in cells_by_id["run-experiment"]
    assert "cases=CASES" in cells_by_id["run-experiment"]


def test_readout_cases_are_defined_visibly_in_the_notebook() -> None:
    namespace = execute_notebook_case_cell()

    cases = namespace["CASES"]
    assert [asdict(case) for case in cases] == [
        {
            "key": "spider",
            "prompt": "The number of legs on the animal that spins webs is",
            "expected_answers": ("8", "eight"),
            "readout": {
                "concepts": ("spider",),
                "literal_argument": None,
                "require_capability_gate": True,
            },
            "intervention": {
                "source_surface": " spider",
                "target_surface": " ant",
                "target_answers": ("6", "six"),
                "alphas": (1.0, 2.0),
            },
        },
        {
            "key": "france_capital",
            "prompt": "The capital of France is the city of",
            "expected_answers": ("Paris",),
            "readout": {
                "concepts": ("France",),
                "literal_argument": "France",
                "require_capability_gate": False,
            },
            "intervention": {
                "source_surface": " France",
                "target_surface": " China",
                "target_answers": ("Beijing",),
                "alphas": (1.0, 2.0),
            },
        },
        {
            "key": "france_language",
            "prompt": "Most people in France speak",
            "expected_answers": ("French",),
            "readout": {
                "concepts": ("France",),
                "literal_argument": "France",
                "require_capability_gate": False,
            },
            "intervention": {
                "source_surface": " France",
                "target_surface": " China",
                "target_answers": ("Chinese",),
                "alphas": (1.0, 2.0),
            },
        },
        {
            "key": "france_continent",
            "prompt": "France is a country on the continent of",
            "expected_answers": ("Europe",),
            "readout": {
                "concepts": ("France",),
                "literal_argument": "France",
                "require_capability_gate": False,
            },
            "intervention": {
                "source_surface": " France",
                "target_surface": " China",
                "target_answers": ("Asia",),
                "alphas": (1.0, 2.0),
            },
        },
        {
            "key": "france_currency",
            "prompt": (
                "The single-word name for the currency now used in France is the"
            ),
            "expected_answers": ("Euro",),
            "readout": {
                "concepts": ("France",),
                "literal_argument": "France",
                "require_capability_gate": False,
            },
            "intervention": {
                "source_surface": " France",
                "target_surface": " China",
                "target_answers": ("Yuan",),
                "alphas": (1.0, 2.0),
            },
        },
    ]


def test_readout_execution_saving_and_reporting_are_separate_cells() -> None:
    notebook = load_notebook(
        Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb")
    )
    cells_by_id = {cell.id: cell.source for cell in notebook.cells}

    run_source = cells_by_id["run-experiment"]
    assert "forward_next_token" in run_source
    assert "run_experiment" in run_source
    assert "write_results" not in run_source
    assert 'result["provenance"]' not in run_source

    save_source = cells_by_id["save-result"]
    assert "result.provenance" in save_source
    assert '"working_tree_dirty": PROJECT_WORKING_TREE_DIRTY' in save_source
    assert "write_results" in save_source
    assert "run_experiment" not in save_source
    assert "result.cases" not in save_source

    report_source = cells_by_id["report-results"]
    assert "write_results" not in report_source
    assert "run_experiment" not in report_source
    assert "print(render_sanity_report(result))" in report_source
    assert "for case in" not in report_source
    assert "for swap in" not in report_source
    assert "render-slices" not in cells_by_id


def test_readout_sanity_documents_text_only_result_artifact() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    assert "runs/jlens-readout-sanity/\n└── result.json" in readme
    assert "spider.html" not in readme
    assert "france_capital.html" not in readme
    assert "Qwen sanity threshold" in normalized
    assert "paper gap" in normalized


def test_asset_notebook_downloads_the_two_pinned_assets_to_drive() -> None:
    notebook = load_notebook(ASSET_NOTEBOOK)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert 'drive.mount("/content/drive")' in source
    assert 'userdata.get("HF_TOKEN")' in source
    assert "/content/drive/MyDrive/data/jlens-reasoning" in source
    assert "Qwen/Qwen3.5-4B" in source
    assert "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" in source
    assert "neuronpedia/jacobian-lens" in source
    assert "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a" in source
    assert "rclone" not in source
