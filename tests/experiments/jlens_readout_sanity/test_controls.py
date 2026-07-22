import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.jlens_readout_sanity.constants import (
    CONTROL_SEEDS,
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    PERCENTILE_INTERPRETATION,
    RANDOM_TARGET_NAMESPACE,
    RANDOM_VECTOR_NAMESPACE,
)
from experiments.jlens_readout_sanity.controls import (
    _control_metadata,
    _wrong_reference_contexts,
    aggregate_all_checks,
    controls_passed,
    require_exact_cases,
    summarize_wrong_concept,
)
from jlens_reasoning.experiments_utils.artifacts import write_results
from jlens_reasoning.experiments_utils.controls import (
    build_random_target_exclusions,
    log_rank_gain,
    percentile,
    percentile_label,
)
from jlens_reasoning.experiments_utils.controls import (
    derive_subseed as _derive_subseed,
)
from jlens_reasoning.experiments_utils.controls import (
    matched_random_vectors as _matched_random_vectors,
)
from jlens_reasoning.experiments_utils.controls import (
    select_random_targets as _select_random_targets,
)
from jlens_reasoning.experiments_utils.controls import (
    strict_percentile_gate as _strict_percentile_gate,
)
from jlens_reasoning.experiments_utils.tokens import concept_surfaces


def derive_subseed(base_seed, layer_index, role):
    return _derive_subseed(
        base_seed,
        layer_index,
        role,
        namespace=RANDOM_VECTOR_NAMESPACE,
    )


def matched_random_vectors(real_vectors, *, base_seed):
    return _matched_random_vectors(
        real_vectors,
        base_seed=base_seed,
        namespace=RANDOM_VECTOR_NAMESPACE,
        norm_atol=1e-6,
        norm_rtol=1e-5,
        low_precision_norm_atol=1e-2,
        low_precision_norm_rtol=1e-2,
        max_attempts=1024,
    )


def strict_percentile_gate(real_score, control_scores, *, quantile=0.95):
    return _strict_percentile_gate(
        real_score,
        control_scores,
        quantile=quantile,
        interpretation=PERCENTILE_INTERPRETATION,
    )


def select_random_targets(tokenizer, **kwargs):
    return _select_random_targets(
        tokenizer,
        namespace=RANDOM_TARGET_NAMESPACE,
        **kwargs,
    )


EXPECTED_CASE_KEYS = (
    "spider",
    "france_capital",
    "france_language",
    "france_continent",
    "france_currency",
)


def _direction_context(key: str, source_id: int, target_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        resolved=SimpleNamespace(
            case=SimpleNamespace(key=key),
            source=SimpleNamespace(token_id=source_id),
            target=SimpleNamespace(token_id=target_id),
        )
    )


def _metadata_context(
    key: str,
    source_id: int,
    target_id: int,
    *,
    source_surface: str,
    target_surface: str,
    clean_answers: tuple[str, ...],
    target_answers: tuple[str, ...],
    formatting_token_id: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        resolved=SimpleNamespace(
            case=SimpleNamespace(
                key=key,
                source_surface=source_surface,
                target_surface=target_surface,
                target_answers=target_answers,
            ),
            read_case=SimpleNamespace(expected_answers=clean_answers),
            source=SimpleNamespace(token_id=source_id),
            target=SimpleNamespace(token_id=target_id),
        ),
        formatting_prefix=[{"token_id": formatting_token_id}],
    )


def test_control_metadata_derives_from_renamed_contexts() -> None:
    first = _metadata_context(
        "alpha",
        1,
        2,
        source_surface=" Wolf",
        target_surface=" Bear",
        clean_answers=("Pup",),
        target_answers=("Cub",),
        formatting_token_id=7,
    )
    second = _metadata_context(
        "beta",
        3,
        4,
        source_surface=" Oak",
        target_surface=" Pine",
        clean_answers=("Leaf",),
        target_answers=("Needle",),
        formatting_token_id=8,
    )

    metadata = _control_metadata((first, second))

    assert metadata.expected_keys == ("alpha", "beta")
    assert metadata.wrong_references == (second, first)
    assert metadata.source_surfaces == (
        *concept_surfaces("Wolf"),
        *concept_surfaces("Oak"),
    )
    assert metadata.target_surfaces == (
        *concept_surfaces("Bear"),
        *concept_surfaces("Pine"),
    )
    assert metadata.clean_answer_surfaces == (
        *concept_surfaces("Pup"),
        *concept_surfaces("Leaf"),
    )
    assert metadata.intended_answer_surfaces == (
        *concept_surfaces("Cub"),
        *concept_surfaces("Needle"),
    )
    assert metadata.formatting_token_ids == (7, 8)


def test_wrong_concept_references_derive_from_directions_not_case_names() -> None:
    first = _direction_context("alpha", 1, 2)
    second = _direction_context("beta", 3, 4)
    third = _direction_context("gamma", 3, 4)

    references = _wrong_reference_contexts((first, second, third))

    assert references == (second, first, first)


def test_wrong_concept_requires_two_distinct_directions() -> None:
    contexts = (
        _direction_context("alpha", 1, 2),
        _direction_context("beta", 1, 2),
    )

    with pytest.raises(ValueError, match="two distinct swap directions"):
        _wrong_reference_contexts(contexts)


def test_shared_control_definitions_are_fixed() -> None:
    assert CONTROL_SEEDS == (
        11,
        29,
        47,
        71,
        101,
        131,
        167,
        199,
        239,
        281,
        331,
        379,
        431,
        487,
        547,
        607,
    )
    assert IDENTITY_ATOL == 1e-6
    assert IDENTITY_RTOL == 1e-5


def test_log_rank_gain_uses_natural_logarithms() -> None:
    assert log_rank_gain(100, 10) == pytest.approx(math.log(10.0))
    assert log_rank_gain(10, 100) == pytest.approx(-math.log(10.0))
    assert log_rank_gain(7, 7) == 0.0


@pytest.mark.parametrize(("clean_rank", "intervened_rank"), [(0, 1), (1, 0)])
def test_log_rank_gain_requires_positive_ranks(
    clean_rank: int, intervened_rank: int
) -> None:
    with pytest.raises(ValueError, match="positive"):
        log_rank_gain(clean_rank, intervened_rank)


def test_percentile_uses_documented_linear_interpolation() -> None:
    values = [float(value) for value in range(16)]
    assert percentile(values, 0.95) == pytest.approx(14.25)


def test_percentile_label_is_derived_from_quantile() -> None:
    assert percentile_label(0.95) == "95th-percentile"
    assert percentile_label(0.9) == "90th-percentile"


def test_percentile_gate_is_strict_and_not_significance_claim() -> None:
    values = [0.0] * 15 + [4.0]
    threshold = percentile(values, 0.95)
    equal = strict_percentile_gate(threshold, values, quantile=0.95)
    greater = strict_percentile_gate(threshold + 1e-12, values, quantile=0.95)

    assert equal["passed"] is False
    assert greater["passed"] is True
    assert equal["interpretation"] == (
        "deterministic sanity check; not statistical significance"
    )


def test_subseeds_are_stable_layer_and_role_specific() -> None:
    assert derive_subseed(11, 7, "source") == 3688398498245801101
    assert derive_subseed(11, 7, "source") != derive_subseed(11, 7, "target")
    assert derive_subseed(11, 7, "source") != derive_subseed(11, 8, "source")


def test_subseed_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="Role"):
        derive_subseed(11, 7, "other")


def test_random_vectors_are_deterministic_for_seed_and_order_independent() -> None:
    real = {
        7: (torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])),
        9: (torch.tensor([1.0, 0.0]), torch.tensor([5.0, 12.0])),
    }

    first, _ = matched_random_vectors(real, base_seed=CONTROL_SEEDS[0])
    second, _ = matched_random_vectors(
        dict(reversed(list(real.items()))), base_seed=CONTROL_SEEDS[0]
    )
    different, _ = matched_random_vectors(real, base_seed=CONTROL_SEEDS[1])

    for layer in real:
        assert torch.equal(first[layer][0], second[layer][0])
        assert torch.equal(first[layer][1], second[layer][1])
    assert not torch.equal(first[7][0], different[7][0])


def test_random_vectors_match_each_per_layer_role_norm() -> None:
    real = {
        2: (torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])),
        3: (torch.zeros(2), torch.tensor([5.0, 12.0])),
    }

    generated, norms = matched_random_vectors(real, base_seed=CONTROL_SEEDS[1])

    for layer in sorted(real):
        for role_index, role in enumerate(("source", "target")):
            real_vector = real[layer][role_index]
            random_vector = generated[layer][role_index]
            assert torch.linalg.vector_norm(random_vector).item() == pytest.approx(
                torch.linalg.vector_norm(real_vector).item(), abs=1e-6, rel=1e-5
            )
            assert norms[str(layer)][role]["matched"] is True
    assert torch.equal(generated[3][0], torch.zeros(2))


def test_random_vectors_restore_real_device_and_dtype() -> None:
    real = {
        2: (
            torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
            torch.tensor([0.0, 2.0], dtype=torch.bfloat16),
        )
    }

    generated, norms = matched_random_vectors(real, base_seed=CONTROL_SEEDS[2])

    for role_index, role in enumerate(("source", "target")):
        assert generated[2][role_index].device == real[2][role_index].device
        assert generated[2][role_index].dtype == real[2][role_index].dtype
        assert norms["2"][role]["device_matches"] is True
        assert norms["2"][role]["dtype_matches"] is True


def test_random_vectors_remain_finite_and_matched_after_low_precision_cast() -> None:
    real = {
        0: (
            torch.tensor([60000.0, 60000.0], dtype=torch.float16),
            torch.tensor([1.0, 2.0], dtype=torch.float16),
        )
    }

    generated, norms = matched_random_vectors(real, base_seed=29)

    assert torch.isfinite(generated[0][0]).all()
    assert norms["0"]["source"]["matched"] is True


class VocabularyTokenizer:
    def __init__(self) -> None:
        self.vocab = {f"token-{token_id}": token_id for token_id in range(40)}
        self.pieces = {
            "spider": [1],
            "France": [2],
            "ant": [3],
            "China": [4],
            "8": [5],
            "Paris": [6],
            "six": [7],
            "New Yuan": [10, 11],
        }
        self.all_special_ids = [0, 13, 14]
        self.added_tokens_decoder = {
            15: SimpleNamespace(special=True),
            16: SimpleNamespace(special=False),
        }
        self.decode_calls: list[tuple[tuple[int, ...], bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.pieces[text]

    def decode(
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        self.decode_calls.append((tuple(token_ids), clean_up_tokenization_spaces))
        token_id = token_ids[0]
        if token_id == 8:
            return " "
        if token_id == 12:
            return "\n"
        return f"token-{token_id}"

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)


def test_random_target_exclusions_are_token_id_based_and_complete() -> None:
    tokenizer = VocabularyTokenizer()

    exclusions = build_random_target_exclusions(
        tokenizer,
        source_surfaces=("spider", "France"),
        target_surfaces=("ant", "China"),
        clean_answer_surfaces=("8", "Paris"),
        intended_answer_surfaces=("six", "New Yuan"),
        formatting_token_ids=(8,),
        existing_excluded_ids=(9,),
    )

    assert exclusions == {
        "sources": [1, 2],
        "targets": [3, 4],
        "clean_answers": [5, 6],
        "intended_answers": [7, 10, 11],
        "formatting": [8],
        "reserved_special": [0, 13, 14, 15, 16],
        "decoded_formatting": [8, 12],
        "existing_filter": [9],
        "all": list(range(17)),
    }
    assert all(cleanup is False for _, cleanup in tokenizer.decode_calls)


def test_random_target_selection_is_exact_deterministic_unique_and_bounded() -> None:
    tokenizer = VocabularyTokenizer()
    excluded_ids = range(16)

    first = select_random_targets(
        tokenizer,
        excluded_ids=excluded_ids,
        seeds=CONTROL_SEEDS,
        output_vocab_size=32,
    )
    second = select_random_targets(
        tokenizer,
        excluded_ids=reversed(list(excluded_ids)),
        seeds=CONTROL_SEEDS,
        output_vocab_size=32,
    )

    assert [item["token_id"] for item in first] == [
        20,
        26,
        21,
        18,
        19,
        28,
        30,
        27,
        22,
        17,
        25,
        16,
        24,
        29,
        31,
        23,
    ]
    assert first == second
    assert len({item["token_id"] for item in first}) == 16
    assert all(0 <= item["token_id"] < 32 for item in first)
    assert all(item["token_id"] != 35 for item in first)
    assert all(item["token"] == f"token-{item['token_id']}" for item in first)


def test_random_target_selection_rejects_empty_eligible_vocabulary() -> None:
    with pytest.raises(ValueError, match="eligible"):
        select_random_targets(
            VocabularyTokenizer(),
            excluded_ids=range(32),
            seeds=CONTROL_SEEDS,
            output_vocab_size=32,
        )


def _gain_cases(gains: list[float]) -> list[dict[str, float | str]]:
    return [
        {"key": key, "log_rank_gain": gain}
        for key, gain in zip(EXPECTED_CASE_KEYS, gains, strict=True)
    ]


def test_exact_case_validation_rejects_missing_duplicate_extra_and_order() -> None:
    complete = _gain_cases([1.0] * 5)
    require_exact_cases(complete, expected_keys=EXPECTED_CASE_KEYS)

    malformed = (
        complete[:-1],
        [*complete[:-1], complete[0]],
        [*complete, {"key": "extra", "log_rank_gain": 1.0}],
        list(reversed(complete)),
    )
    for cases in malformed:
        with pytest.raises(ValueError, match="exact case keys"):
            require_exact_cases(cases, expected_keys=EXPECTED_CASE_KEYS)


def test_wrong_concept_requires_aggregate_and_four_strict_case_wins() -> None:
    result = summarize_wrong_concept(
        _gain_cases([1.0, 1.0, 1.0, 1.0, 0.0]),
        _gain_cases([0.0, 0.0, 0.0, 0.0, 0.0]),
        expected_keys=EXPECTED_CASE_KEYS,
    )

    assert result["matched_mean_log_rank_gain"] == pytest.approx(0.8)
    assert result["mismatched_mean_log_rank_gain"] == 0.0
    assert result["matched_winning_case_count"] == 4
    assert result["aggregate_condition"] is True
    assert result["case_condition"] is True
    assert result["passed"] is True


def test_wrong_concept_ties_do_not_count_and_three_wins_fail() -> None:
    result = summarize_wrong_concept(
        _gain_cases([1.0, 1.0, 1.0, 0.0, 0.0]),
        _gain_cases([0.0, 0.0, 0.0, 0.0, -1.0]),
        expected_keys=EXPECTED_CASE_KEYS,
    )

    assert [case["matched_wins"] for case in result["cases"]] == [
        True,
        True,
        True,
        False,
        True,
    ]
    assert result["matched_winning_case_count"] == 4

    three_wins = summarize_wrong_concept(
        _gain_cases([1.0, 1.0, 1.0, 0.0, 0.0]),
        _gain_cases([0.0, 0.0, 0.0, 0.0, 0.0]),
        expected_keys=EXPECTED_CASE_KEYS,
    )
    assert three_wins["matched_winning_case_count"] == 3
    assert three_wins["passed"] is False


def _passing_controls() -> dict[str, dict[str, object]]:
    return {
        "identity": {
            "passed": True,
            "passed_case_count": 5,
            "required_case_count": 5,
            "maximum_absolute_logit_difference": 0.0,
        },
        "matched_random_vector": {
            "passed": True,
            "real_mean_log_rank_gain": 1.0,
            "percentile_95_threshold": 0.5,
        },
        "wrong_concept": {
            "passed": True,
            "matched_mean_log_rank_gain": 1.0,
            "mismatched_mean_log_rank_gain": 0.0,
            "matched_winning_case_count": 5,
            "required_winning_case_count": 4,
        },
        "random_target": {
            "passed": True,
            "real_mean_log_rank_gain": 1.0,
            "percentile_95_threshold": 0.5,
        },
    }


def test_all_four_controls_are_integrated_into_global_checks() -> None:
    controls = _passing_controls()

    checks, failures, passed = aggregate_all_checks(
        {"clean_baselines": True}, [], controls
    )

    assert checks == {
        "clean_baselines": True,
        "identity_control": True,
        "matched_random_vector_control": True,
        "wrong_concept_control": True,
        "random_target_control": True,
    }
    assert failures == []
    assert controls_passed(controls) is True
    assert passed is True


def test_control_and_global_pass_scopes_are_separate() -> None:
    controls = _passing_controls()

    _, failures, global_passed = aggregate_all_checks(
        {"clean_baselines": False}, ["existing failure"], controls
    )

    assert controls_passed(controls) is True
    assert global_passed is False
    assert failures == ["existing failure"]


@pytest.mark.parametrize(
    "control_name",
    ["identity", "matched_random_vector", "wrong_concept", "random_target"],
)
def test_each_failed_control_forces_global_failure_and_actionable_failure(
    control_name: str,
) -> None:
    controls = _passing_controls()
    controls[control_name]["passed"] = False

    checks, failures, passed = aggregate_all_checks(
        {"clean_baselines": True}, [], controls
    )

    assert checks[f"{control_name}_control"] is False
    assert controls_passed(controls) is False
    assert passed is False
    assert len(failures) == 1
    assert control_name.replace("_", " ") in failures[0]
    assert "required" in failures[0]


def test_missing_control_payload_is_rejected() -> None:
    controls = _passing_controls()
    del controls["identity"]

    with pytest.raises(KeyError, match="identity"):
        aggregate_all_checks({"clean_baselines": True}, [], controls)


def test_control_schema_serializes_stably_with_separate_pass_scopes(
    tmp_path: Path,
) -> None:
    controls = {
        "seeds": list(CONTROL_SEEDS),
        "definitions": {
            "percentile_interpretation": (
                "deterministic sanity check; not statistical significance"
            )
        },
        "thresholds": {"percentile_quantile": 0.95},
        "tolerances": {"identity_logits": {"atol": 1e-6, "rtol": 1e-5}},
        **_passing_controls(),
        "passed": True,
    }
    result = {
        "controls": controls,
        "checks": {"clean_baselines": False, "identity_control": True},
        "failures": ["existing failure"],
        "passed": False,
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_results(first, result)
    write_results(second, result)

    assert first.read_bytes() == second.read_bytes()
    serialized = json.loads(first.read_text(encoding="utf-8"))
    assert serialized["controls"]["seeds"] == list(CONTROL_SEEDS)
    assert serialized["controls"]["thresholds"] == {"percentile_quantile": 0.95}
    assert serialized["controls"]["identity"]["passed"] is True
    assert serialized["controls"]["passed"] is True
    assert serialized["passed"] is False
