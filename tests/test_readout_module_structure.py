from jlens_reasoning.experiments import readout_cases, readout_sanity, readout_utils


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
