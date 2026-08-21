import re
from dataclasses import asdict
from pathlib import Path

import nbformat
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
FLENQA_SANITY_NOTEBOOK = Path("notebooks/flenqa_output_sanity.ipynb")
FLENQA_NOTEBOOKS = [*FLENQA_BENCHMARK_NOTEBOOKS, FLENQA_ACCURACY_NOTEBOOK]
EXPERIMENT_NOTEBOOKS = sorted(Path("experiments").glob("*/*.ipynb"))
NOTEBOOKS = [
    *SHARED_NOTEBOOKS,
    *FLENQA_NOTEBOOKS,
    FLENQA_SANITY_NOTEBOOK,
    *EXPERIMENT_NOTEBOOKS,
]
ASSET_NOTEBOOK = Path("notebooks/01_download_assets.ipynb")
ALL_NOTEBOOKS = [*NOTEBOOKS, ASSET_NOTEBOOK]


def load_notebook(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(path, as_version=4)


def notebook_cells_by_id(path: Path) -> dict[str, str]:
    notebook = load_notebook(path)
    return {cell.id: cell.source for cell in notebook.cells}


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
        FLENQA_SANITY_NOTEBOOK,
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
        Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb"),
    ]
    assert not Path("notebooks/01_jlens_readout_sanity.ipynb").exists()


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


def test_flenqa_accuracy_notebook_reuses_full_run_prompts() -> None:
    notebook = load_notebook(FLENQA_ACCURACY_NOTEBOOK)
    cells = notebook_cells_by_id(FLENQA_ACCURACY_NOTEBOOK)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "initialize_colab(enable_wandb=False, require_cuda=True)" in source
    assert 'context.runs_dir / "flenqa-full-run" / "prompts"' in cells["load-prompts"]
    assert "ds.dataset(" in cells["load-prompts"]
    assert 'sort_by("canonical_index")' in cells["load-prompts"]
    assert "len(prompts) == 9_862" in cells["load-prompts"]
    assert "load_from_disk" not in source
    assert "normalize_rows" not in source
    assert "prepare_prompts" not in source
    assert "for prompt in tqdm(prompts" in cells["run-accuracy"]
    assert "evaluate_paper_binary" in cells["run-accuracy"]
    assert "from jlens_reasoning.inference import" in source
    assert "InferenceConfig.direct(" in source
    assert "max_input_tokens=4096" in source
    assert "generate_chat(" in cells["run-accuracy"]
    assert '"canonical_index": prompt["canonical_index"]' in cells["run-accuracy"]
    assert "causal_lm.generate(" not in source
    assert "generated_text" in cells["run-accuracy"]
    assert "reasoning_text" in cells["run-accuracy"]
    assert "answer_text" in cells["run-accuracy"]
    assert "reasoning_status" in cells["run-accuracy"]
    assert "inference_mode" in cells["run-accuracy"]
    assert "max_new_tokens" in cells["run-accuracy"]
    assert "paper_weight" in cells["run-accuracy"]
    assert "pa.Table.from_pylist" in cells["save-results"]
    assert (
        'pa.field("canonical_index", pa.int32(), nullable=False)'
        in cells["save-results"]
    )
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


def test_flenqa_output_sanity_notebook_has_a_lightweight_read_only_workflow() -> None:
    cells = notebook_cells_by_id(FLENQA_SANITY_NOTEBOOK)
    source = "\n".join(cells.values())

    assert "initialize_colab(enable_wandb=False, require_cuda=False)" in source
    assert 'context.runs_dir / "flenqa-full-run"' in source
    assert "TABLE_SCHEMAS" in source
    assert "metadata.num_rows" in source
    assert "9_862" in source
    assert 'row_counts["topk"] == len(execution_positions)' in source
    assert "check_topk_sample(" in source
    assert "AutoTokenizer.from_pretrained" in source
    assert "AutoModel" not in source
    assert "JacobianLens" not in source
    assert "write_table" not in source
    assert "ParquetWriter" not in source


def test_flenqa_output_sanity_sample_check_rejects_broken_rank_groups() -> None:
    source = notebook_cells_by_id(FLENQA_SANITY_NOTEBOOK)["define-sample-check"]
    namespace: dict[str, object] = {}
    exec(
        compile(source, f"{FLENQA_SANITY_NOTEBOOK}:define-sample-check", "exec"),
        namespace,
    )
    check_topk_sample = namespace["check_topk_sample"]
    rows = [
        {
            "prompt_id": "p",
            "lens_kind": lens_kind,
            "layer": 2,
            "position": 4,
            "rank": rank,
            "token_id": rank,
            "logit": float(3 - rank),
        }
        for lens_kind in ("jacobian", "logit")
        for rank in (1, 2)
    ]

    assert check_topk_sample(rows, {("p", 4)}, top_k=2) == 1
    rows[-1]["rank"] = 3
    with pytest.raises(AssertionError):
        check_topk_sample(rows, {("p", 4)}, top_k=2)


def test_flenqa_output_sanity_sample_check_rejects_extra_lens_kinds() -> None:
    source = notebook_cells_by_id(FLENQA_SANITY_NOTEBOOK)["define-sample-check"]
    namespace: dict[str, object] = {}
    exec(
        compile(source, f"{FLENQA_SANITY_NOTEBOOK}:define-sample-check", "exec"),
        namespace,
    )
    check_topk_sample = namespace["check_topk_sample"]
    rows = [
        {
            "prompt_id": "p",
            "lens_kind": lens_kind,
            "layer": 2,
            "position": 4,
            "rank": rank,
            "token_id": rank,
            "logit": float(3 - rank),
        }
        for lens_kind in ("jacobian", "logit", "unexpected")
        for rank in (1, 2)
    ]

    with pytest.raises(AssertionError):
        check_topk_sample(rows, {("p", 4)}, top_k=2)


def test_flenqa_output_sanity_sample_check_rejects_position_layer_gaps() -> None:
    source = notebook_cells_by_id(FLENQA_SANITY_NOTEBOOK)["define-sample-check"]
    namespace: dict[str, object] = {}
    exec(
        compile(source, f"{FLENQA_SANITY_NOTEBOOK}:define-sample-check", "exec"),
        namespace,
    )
    check_topk_sample = namespace["check_topk_sample"]
    rows = [
        {
            "prompt_id": "p",
            "lens_kind": lens_kind,
            "layer": layer,
            "position": position,
            "rank": rank,
            "token_id": rank,
            "logit": float(3 - rank),
        }
        for lens_kind in ("jacobian", "logit")
        for position, layers in ((4, (1, 2)), (5, (1,)))
        for layer in layers
        for rank in (1, 2)
    ]

    with pytest.raises(AssertionError):
        check_topk_sample(rows, {("p", 4), ("p", 5)}, top_k=2)


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
