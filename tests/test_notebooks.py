import re
from dataclasses import asdict
from pathlib import Path

import nbformat

SHARED_NOTEBOOKS = [
    Path("notebooks/_template.ipynb"),
    Path("notebooks/00_environment_check.ipynb"),
]
FLENQA_NOTEBOOKS = [
    Path("notebooks/flenqa_smoke.ipynb"),
    Path("notebooks/flenqa_full_run.ipynb"),
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

    for path in FLENQA_NOTEBOOKS:
        source = "\n".join(cell.source for cell in load_notebook(path).cells)
        assert "from jlens_reasoning.benchmarks.flenqa.runner import" in source
        assert "run_benchmark(" in source
        assert not any(fragment in source for fragment in forbidden)


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
