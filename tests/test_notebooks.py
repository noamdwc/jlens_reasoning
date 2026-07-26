from dataclasses import asdict
from pathlib import Path

import nbformat

SHARED_NOTEBOOKS = [
    Path("notebooks/_template.ipynb"),
    Path("notebooks/00_environment_check.ipynb"),
]
EXPERIMENT_NOTEBOOKS = sorted(Path("experiments").glob("*/*.ipynb"))
NOTEBOOKS = [*SHARED_NOTEBOOKS, *EXPERIMENT_NOTEBOOKS]


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
    for path in NOTEBOOKS:
        notebook = load_notebook(path)
        for cell in notebook.cells:
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []


def test_notebooks_share_one_canonical_loader_cell() -> None:
    loader_cells = [load_notebook(path).cells[0].source for path in NOTEBOOKS]

    assert loader_cells[0] == loader_cells[1]
    assert "GITHUB_TOKEN_JLENS_REAS" in loader_cells[0]
    assert "scripts/colab_bootstrap.py" in loader_cells[0]
    assert "PROJECT_REF" in loader_cells[0]


def test_notebooks_do_not_contain_credentials() -> None:
    forbidden_fragments = ("github_pat_", "ghp_", "hf_", "wandb-secret")

    for path in NOTEBOOKS:
        source = path.read_text(encoding="utf-8")
        assert not any(fragment in source for fragment in forbidden_fragments)


def test_notebooks_use_the_colab_environment_module() -> None:
    for path in NOTEBOOKS:
        notebook = load_notebook(path)
        source = "\n".join(cell.source for cell in notebook.cells)

        assert (
            "from jlens_reasoning.environments.colab import initialize_colab" in source
        )
        assert "context = initialize_colab(" in source


def test_experiment_notebooks_are_discovered_without_a_registry() -> None:
    assert EXPERIMENT_NOTEBOOKS == [
        Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb")
    ]
    assert not Path("notebooks/01_jlens_readout_sanity.ipynb").exists()


def test_readout_sanity_notebook_has_pinned_gpu_workflow() -> None:
    path = Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb")
    notebook = load_notebook(path)
    cells_by_id = notebook_cells_by_id(path)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "initialize_colab(enable_wandb=False, require_cuda=True)" in source
    assert 'MODEL_NAME = "Qwen/Qwen3.5-4B"' not in source
    assert "from experiments.jlens_readout_sanity.constants import" in source
    assert "from experiments.jlens_readout_sanity.utils import" in source
    assert "JacobianLens.from_pretrained" in source
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
