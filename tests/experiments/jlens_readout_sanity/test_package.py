from experiments.jlens_readout_sanity import controls, runner, utils


def test_utils_is_the_small_notebook_facade() -> None:
    assert utils.run_readout_sanity is runner.run_readout_sanity
    assert utils.write_results.__module__ == (
        "jlens_reasoning.experiments_utils.artifacts"
    )
    assert utils.validate_model_lens.__module__ == (
        "jlens_reasoning.experiments_utils.validation"
    )


def test_control_orchestration_remains_experiment_local() -> None:
    assert controls.run_negative_controls.__module__ == (
        "experiments.jlens_readout_sanity.controls"
    )
