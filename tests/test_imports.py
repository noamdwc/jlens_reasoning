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
