from pathlib import Path

from jlens_reasoning.experiments import (
    intervention_utils,
    readout_cases,
    readout_controls,
    readout_sanity,
    readout_utils,
)


def test_case_definitions_live_in_focused_module_and_are_reexported() -> None:
    assert readout_sanity.ReadoutCase is readout_cases.ReadoutCase
    assert readout_sanity.SwapCase is readout_cases.SwapCase
    assert readout_sanity.TokenVariant is readout_cases.TokenVariant
    assert readout_sanity.READOUT_CASES is readout_cases.READOUT_CASES
    assert readout_sanity.SWAP_CASES is readout_cases.SWAP_CASES
    assert readout_sanity.single_token_surface is readout_cases.single_token_surface
    assert readout_sanity.concept_token_variants is readout_cases.concept_token_variants


def test_stateless_utilities_are_reexported_from_facade() -> None:
    exported_names = (
        "find_last_subsequence",
        "positions_after_literal",
        "best_target_rank",
        "top_tokens",
        "prepare_scoring_input",
        "aggregate_capability_checks",
        "workspace_loading",
        "workspace_layers",
        "write_results",
        "validate_model_lens",
    )
    for name in exported_names:
        assert getattr(readout_sanity, name) is getattr(readout_utils, name)


def test_intervention_mechanics_are_reexported_from_facade() -> None:
    exported_names = (
        "LensCoordinateSwapper",
        "jlens_vector",
        "coordinate_swap",
        "execute_intervention",
        "analyze_identity_case",
        "summarize_swap_logits",
        "analyze_swap_case",
        "_token_vectors_by_layer",
    )
    for name in exported_names:
        assert getattr(readout_sanity, name) is getattr(intervention_utils, name)


def test_negative_control_orchestration_has_a_focused_module() -> None:
    assert callable(readout_controls.run_negative_controls)
    assert (
        readout_sanity._run_negative_controls is readout_controls.run_negative_controls
    )


def test_readout_sanity_is_a_small_stable_facade() -> None:
    facade = Path(readout_sanity.__file__)
    assert len(facade.read_text(encoding="utf-8").splitlines()) < 500
    assert readout_sanity.run_readout_sanity.__module__ == (
        "jlens_reasoning.experiments.readout_sanity"
    )
