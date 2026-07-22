import ast
from pathlib import Path

from experiments.jlens_readout_sanity import (
    controls,
    experiment,
    utils,
)

DOCUMENTED_MODULES = (
    Path("experiments/jlens_readout_sanity/experiment.py"),
    Path("experiments/jlens_readout_sanity/controls.py"),
    Path("experiments/jlens_readout_sanity/reporting.py"),
    Path("src/jlens_reasoning/evaluation.py"),
    Path("src/jlens_reasoning/evaluation_utils.py"),
    Path("src/jlens_reasoning/experiments_utils/interventions.py"),
)


def test_utils_is_the_small_notebook_facade() -> None:
    assert utils.run_experiment is experiment.run_experiment
    assert utils.write_results.__module__ == (
        "jlens_reasoning.experiments_utils.artifacts"
    )
    assert utils.validate_model_lens.__module__ == (
        "jlens_reasoning.experiments_utils.validation"
    )
    assert utils.render_sanity_report.__module__ == (
        "experiments.jlens_readout_sanity.reporting"
    )


def test_control_orchestration_remains_experiment_local() -> None:
    assert controls.run_control_suite.__module__ == (
        "experiments.jlens_readout_sanity.controls"
    )


def test_public_and_large_functions_have_responsibility_docstrings() -> None:
    missing: list[str] = []
    for path in DOCUMENTED_MODULES:
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body_start = node.body[0].lineno
            body_lines = node.end_lineno - body_start + 1
            requires_doc = not node.name.startswith("_") or body_lines >= 20
            if requires_doc and ast.get_docstring(node, clean=False) is None:
                missing.append(f"{path}:{node.lineno}:{node.name}")

    assert missing == []
