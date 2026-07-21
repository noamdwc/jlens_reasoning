from jlens_reasoning.experiments import readout_cases
from jlens_reasoning.experiments import readout_sanity


def test_case_definitions_live_in_focused_module_and_are_reexported() -> None:
    assert readout_sanity.ReadoutCase is readout_cases.ReadoutCase
    assert readout_sanity.SwapCase is readout_cases.SwapCase
    assert readout_sanity.TokenVariant is readout_cases.TokenVariant
    assert readout_sanity.READOUT_CASES is readout_cases.READOUT_CASES
    assert readout_sanity.SWAP_CASES is readout_cases.SWAP_CASES
    assert readout_sanity.single_token_surface is readout_cases.single_token_surface
    assert readout_sanity.concept_token_variants is readout_cases.concept_token_variants
