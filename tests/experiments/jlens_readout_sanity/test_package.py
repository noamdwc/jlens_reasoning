from experiments.jlens_readout_sanity import (
    control_analysis,
    control_execution,
    controls,
    runner,
    utils,
)


def test_utils_is_the_small_notebook_facade() -> None:
    assert utils.run_readout_sanity is runner.run_readout_sanity
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
    assert controls.run_negative_controls.__module__ == (
        "experiments.jlens_readout_sanity.controls"
    )


def test_control_analysis_owns_pure_control_logic() -> None:
    assert control_analysis._control_metadata.__module__ == (
        "experiments.jlens_readout_sanity.control_analysis"
    )
    assert control_analysis.summarize_wrong_concept.__module__ == (
        "experiments.jlens_readout_sanity.control_analysis"
    )
    assert control_analysis.aggregate_all_checks.__module__ == (
        "experiments.jlens_readout_sanity.control_analysis"
    )


def test_controls_facade_reexports_analysis_api() -> None:
    assert controls.summarize_wrong_concept is control_analysis.summarize_wrong_concept
    assert controls.require_exact_cases is control_analysis.require_exact_cases
    assert controls.controls_passed is control_analysis.controls_passed
    assert controls.aggregate_all_checks is control_analysis.aggregate_all_checks


def test_control_execution_owns_model_backed_logic() -> None:
    assert control_execution.analyze_identity_case.__module__ == (
        "experiments.jlens_readout_sanity.control_execution"
    )
    assert control_execution.run_identity_control.__module__ == (
        "experiments.jlens_readout_sanity.control_execution"
    )
    assert control_execution.run_matched_random_vector_control.__module__ == (
        "experiments.jlens_readout_sanity.control_execution"
    )
    assert control_execution.run_wrong_concept_control.__module__ == (
        "experiments.jlens_readout_sanity.control_execution"
    )
    assert control_execution.run_random_target_control.__module__ == (
        "experiments.jlens_readout_sanity.control_execution"
    )


def test_controls_facade_reexports_identity_analyzer() -> None:
    assert controls.analyze_identity_case is control_execution.analyze_identity_case
