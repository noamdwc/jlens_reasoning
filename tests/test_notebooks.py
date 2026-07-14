from pathlib import Path

import nbformat

NOTEBOOKS = [
    Path("notebooks/_template.ipynb"),
    Path("notebooks/00_environment_check.ipynb"),
    Path("notebooks/01_jlens_readout_sanity.ipynb"),
]


def load_notebook(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(path, as_version=4)


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


def test_readout_sanity_notebook_has_pinned_gpu_workflow() -> None:
    notebook = load_notebook(Path("notebooks/01_jlens_readout_sanity.ipynb"))
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "initialize_colab(enable_wandb=False, require_cuda=True)" in source
    assert 'MODEL_NAME = "Qwen/Qwen3.5-4B"' not in source
    assert "from jlens_reasoning.experiments.readout_sanity import" in source
    assert "JacobianLens.from_pretrained" in source
    assert "run_readout_sanity" in source
    assert "write_results" in source
    assert "compute_slice" in source
    assert 'mode="embed"' in source
    assert "raise RuntimeError" in source
