import importlib
import importlib.util
from pathlib import Path

EXPECTED_MODULES = (
    "experiments.jlens_readout_sanity.constants",
    "experiments.jlens_readout_sanity.controls",
    "experiments.jlens_readout_sanity.runner",
    "experiments.jlens_readout_sanity.utils",
    "jlens_reasoning.experiments_utils.artifacts",
    "jlens_reasoning.experiments_utils.controls",
    "jlens_reasoning.experiments_utils.interventions",
    "jlens_reasoning.experiments_utils.tokens",
    "jlens_reasoning.experiments_utils.validation",
)


def test_project_and_jacobian_lens_import() -> None:
    import jlens

    import jlens_reasoning

    assert jlens_reasoning.__version__ == "0.1.0"
    assert jlens is not None


def test_new_experiment_package_roots_import() -> None:
    import experiments
    import experiments.jlens_readout_sanity
    import jlens_reasoning.experiments_utils

    assert experiments.__name__ == "experiments"
    assert experiments.jlens_readout_sanity.__name__ == (
        "experiments.jlens_readout_sanity"
    )
    assert jlens_reasoning.experiments_utils.__name__ == (
        "jlens_reasoning.experiments_utils"
    )


def test_all_focused_experiment_modules_import() -> None:
    for module_name in EXPECTED_MODULES:
        assert importlib.import_module(module_name).__name__ == module_name


def test_old_experiments_package_is_removed() -> None:
    assert not Path("src/jlens_reasoning/experiments").exists()
    assert importlib.util.find_spec("jlens_reasoning.experiments") is None
