import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import experiments.jlens_readout_sanity.control_execution as control_execution_module
import experiments.jlens_readout_sanity.runner as runner_module
import jlens_reasoning.experiments_utils.interventions as intervention_utils_module
from experiments.jlens_readout_sanity.constants import (
    CONTROL_SEEDS,
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    WORKSPACE_LAYER_LOWER_FRACTION,
    WORKSPACE_LAYER_UPPER_FRACTION,
)
from experiments.jlens_readout_sanity.runner import (
    _validate_case_configuration,
    aggregate_capability_checks,
    analyze_case,
    analyze_swap_case,
    run_readout_sanity,
    summarize_swap_logits,
)
from experiments.jlens_readout_sanity.types import ReadoutCase, SwapCase
from jlens_reasoning.experiments_utils.artifacts import write_results
from jlens_reasoning.experiments_utils.controls import log_rank_gain
from jlens_reasoning.experiments_utils.interventions import (
    LensCoordinateSwapper,
    coordinate_swap,
    execute_intervention,
    jlens_vector,
)
from jlens_reasoning.experiments_utils.interventions import (
    token_vectors_by_layer as _token_vectors_by_layer,
)
from jlens_reasoning.experiments_utils.tokens import (
    TokenVariant,
    best_target_rank,
    concept_token_variants,
    find_last_subsequence,
    positions_after_literal,
    single_token_surface,
    top_tokens,
)
from jlens_reasoning.experiments_utils.tokens import (
    prepare_scoring_input as _prepare_scoring_input,
)
from jlens_reasoning.experiments_utils.validation import (
    validate_model_lens,
    workspace_loading,
)
from jlens_reasoning.experiments_utils.validation import (
    workspace_layers as _workspace_layers,
)
from tests.experiments.jlens_readout_sanity.case_fixtures import (
    READOUT_CASES,
    SWAP_CASES,
)


def workspace_layers(n_layers, source_layers):
    return _workspace_layers(
        n_layers,
        source_layers,
        lower_fraction=WORKSPACE_LAYER_LOWER_FRACTION,
        upper_fraction=WORKSPACE_LAYER_UPPER_FRACTION,
    )


def prepare_scoring_input(*args, **kwargs):
    kwargs.setdefault("max_formatting_tokens", 2)
    return _prepare_scoring_input(*args, **kwargs)


def test_run_requires_explicit_case_tuples() -> None:
    parameters = inspect.signature(run_readout_sanity).parameters
    assert parameters["cases"].default is inspect.Parameter.empty
    assert parameters["swap_cases"].default is inspect.Parameter.empty


def test_case_configuration_requires_five_ordered_matching_keys() -> None:
    valid_reads = tuple(READOUT_CASES)
    valid_swaps = tuple(SWAP_CASES)

    _validate_case_configuration(valid_reads, valid_swaps)

    malformed = (
        (valid_reads[:-1], valid_swaps[:-1]),
        (valid_reads, tuple(reversed(valid_swaps))),
        (valid_reads, (*valid_swaps[:-1], valid_swaps[0])),
    )
    for read_cases, swap_cases in malformed:
        with pytest.raises(
            ValueError,
            match="five|unique|same keys in the same order",
        ):
            _validate_case_configuration(read_cases, swap_cases)


def test_released_artifact_coordinates_match_upstream_walkthrough() -> None:
    assert MODEL_NAME == "Qwen/Qwen3.5-4B"
    assert LENS_REPO == "neuronpedia/jacobian-lens"
    assert LENS_REVISION == "qwen-n1000"
    assert LENS_FILE.endswith("Qwen3.5-4B_jacobian_lens_n1000.pt")


def test_cases_cover_released_read_and_swap_examples() -> None:
    read_cases = {case.key: case for case in READOUT_CASES}
    swap_cases = {case.key: case for case in SWAP_CASES}

    assert read_cases["spider"].prompt == (
        "The number of legs on the animal that spins webs is"
    )
    assert read_cases["spider"].expected_answers == ("8", "eight")
    assert read_cases["spider"].target_concepts == ("spider",)

    assert [(case.key, case.target_answers[0]) for case in SWAP_CASES] == [
        ("spider", "6"),
        ("france_capital", "Beijing"),
        ("france_language", "Chinese"),
        ("france_continent", "Asia"),
        ("france_currency", "Yuan"),
    ]
    assert swap_cases["spider"].source_surface == " spider"
    assert swap_cases["spider"].target_surface == " ant"
    france_swaps = [case for case in SWAP_CASES if case.key.startswith("france_")]
    assert all(case.source_surface == " France" for case in france_swaps)
    assert all(case.target_surface == " China" for case in france_swaps)


def test_single_token_surface_is_strict() -> None:
    tokenizer = FakeTokenizer()

    assert single_token_surface(tokenizer, " France") == TokenVariant(
        token_id=17,
        surface=" France",
    )
    with pytest.raises(ValueError, match="exactly one token"):
        single_token_surface(tokenizer, " FRANCE")


class FakeTokenizer:
    def __init__(self) -> None:
        self.pieces = {
            "France": [7],
            " France": [17],
            "france": [8],
            " france": [18],
            "FRANCE": [9, 10],
            " FRANCE": [19, 20],
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.pieces.get(text, [99, 100])

    def decode(
        self, token_ids: list[int], *, clean_up_tokenization_spaces: bool = False
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return f"token-{token_ids[0]}"


def test_concept_variants_keep_single_tokens_and_deduplicate() -> None:
    variants = concept_token_variants(FakeTokenizer(), ("France", "france"))

    assert [(variant.token_id, variant.surface) for variant in variants] == [
        (7, "France"),
        (17, " France"),
        (8, "france"),
        (18, " france"),
    ]


def test_find_last_subsequence_and_positions_after_literal() -> None:
    assert find_last_subsequence([1, 17, 2, 17, 3], ([7], [17])) == (3, 4)
    assert positions_after_literal(
        FakeTokenizer(), torch.tensor([[1, 17, 2, 3]]), "France"
    ) == [2, 3]


def test_rank_is_one_based_best_variant_and_stable_for_ties() -> None:
    logits = torch.tensor([0.0, 3.0, 3.0, 1.0])

    assert best_target_rank(logits, (2, 3)) == 2
    assert best_target_rank(logits, (3,)) == 3


def test_jlens_vector_composes_jacobian_and_unembedding() -> None:
    lens = SimpleNamespace(jacobians={1: torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
    unembedding = torch.tensor([[0.0, 0.0], [5.0, 6.0]])

    assert torch.equal(
        jlens_vector(lens, unembedding, layer=1, token_id=1),
        torch.tensor([23.0, 34.0]),
    )


def test_sampled_target_uses_the_real_jlens_vector_construction_path() -> None:
    lens = SimpleNamespace(jacobians={2: torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
    unembedding = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    vectors = _token_vectors_by_layer(
        lens=lens,
        unembedding_weight=unembedding,
        layers=[2],
        source_token_id=1,
        target_token_id=3,
    )

    assert torch.equal(
        vectors[2][1],
        jlens_vector(lens, unembedding, layer=2, token_id=3),
    )
    assert not torch.equal(vectors[2][1], unembedding[3])


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        (0.0, [1.0, 0.0, 7.0]),
        (1.0, [0.0, 1.0, 7.0]),
        (2.0, [-1.0, 2.0, 7.0]),
    ],
)
def test_coordinate_swap_strength_and_orthogonal_component(
    alpha: float,
    expected: list[float],
) -> None:
    hidden = torch.tensor([1.0, 0.0, 7.0])
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])

    actual = coordinate_swap(hidden, source, target, alpha=alpha)

    assert torch.allclose(actual, torch.tensor(expected))


def test_coordinate_swap_preserves_shape_and_dtype() -> None:
    hidden = torch.tensor(
        [[[1.0, 0.0], [0.5, 0.25]]],
        dtype=torch.bfloat16,
    )

    actual = coordinate_swap(
        hidden,
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        alpha=1.0,
    )

    assert actual.shape == hidden.shape
    assert actual.dtype == hidden.dtype


class TensorBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TupleBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, str]:
        return hidden, "cache"


def test_swapper_patches_all_positions_and_preserves_tuple_members() -> None:
    blocks = nn.ModuleList([TensorBlock(), TupleBlock()])
    vectors = {
        0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
        1: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
    }
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    with LensCoordinateSwapper(blocks, vectors, alpha=1.0):
        first = blocks[0](hidden)
        second, cache = blocks[1](hidden)

    assert torch.equal(first, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    assert torch.equal(second, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    assert cache == "cache"
    assert all(not block._forward_hooks for block in blocks)


def test_swapper_removes_hooks_after_exception() -> None:
    blocks = nn.ModuleList([TensorBlock()])
    vectors = {
        0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
    }

    with pytest.raises(RuntimeError, match="stop"):
        with LensCoordinateSwapper(blocks, vectors, alpha=1.0):
            raise RuntimeError("stop")

    assert not blocks[0]._forward_hooks


def test_intervention_executor_removes_hooks_when_forward_raises() -> None:
    model = SimpleNamespace(layers=nn.ModuleList([TensorBlock(), TensorBlock()]))
    vectors = {
        0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
        1: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
    }

    with pytest.raises(RuntimeError, match="forward failed"):
        execute_intervention(
            model=model,
            forward_next_token=lambda input_ids: (_ for _ in ()).throw(
                RuntimeError("forward failed")
            ),
            scoring_input=torch.tensor([[1, 2]]),
            vectors_by_layer=vectors,
            alpha=1.0,
        )

    assert all(not block._forward_hooks for block in model.layers)


def test_top_tokens_preserve_token_ids_and_logits() -> None:
    assert top_tokens(torch.tensor([0.0, 2.0, 1.0]), FakeTokenizer(), k=2) == [
        {"token_id": 1, "token": "token-1", "logit": 2.0},
        {"token_id": 2, "token": "token-2", "logit": 1.0},
    ]


def test_workspace_layers_use_inclusive_ceil_and_floor_bounds() -> None:
    assert workspace_layers(20, range(20)) == list(range(7, 17))
    assert workspace_layers(20, [0, 6, 7, 12, 16, 17, 19]) == [7, 12, 16]


def test_workspace_loading_averages_layers_and_positions() -> None:
    activations = {
        2: torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        3: torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    }
    vectors = {
        2: torch.tensor([1.0, 0.0]),
        3: torch.tensor([1.0, 0.0]),
    }

    assert workspace_loading(activations, vectors, positions=[0, 1]) == pytest.approx(
        0.75
    )


def test_results_round_trip_without_tensors(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    write_results(output, {"rank": torch.tensor(3), "layers": torch.tensor([7, 8])})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "layers": [7, 8],
        "rank": 3,
    }


class RunnerTokenizer(FakeTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.pieces.update(
            {
                "spider": [2],
                " spider": [2],
                "Spider": [2],
                " Spider": [2],
                "SPIDER": [2],
                " SPIDER": [2],
                "8": [4],
                " 8": [4],
                "eight": [4],
                " eight": [4],
                "Eight": [4],
                " Eight": [4],
                "EIGHT": [4],
                " EIGHT": [4],
            }
        )


def test_swap_summary_uses_best_strength_and_clean_rank() -> None:
    tokenizer = RunnerTokenizer()
    tokenizer.pieces.update(
        {
            "6": [5],
            " 6": [5],
            "six": [5],
            " six": [5],
            "Six": [5],
            " Six": [5],
            "SIX": [5],
            " SIX": [5],
        }
    )
    clean = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
    alpha_1 = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, 3.5])
    alpha_2 = torch.tensor([1.0, 0.0, -1.0, -2.0, -3.0, 5.0])

    result = summarize_swap_logits(
        clean,
        {1.0: alpha_1, 2.0: alpha_2},
        clean_answers=("8", "eight"),
        target_answers=("6", "six"),
        tokenizer=tokenizer,
        top_k=3,
    )

    assert result["clean"]["expected_rank"] == 5
    assert result["clean"]["expected_top1"] is False
    assert result["clean"]["target_rank"] == 6
    assert result["interventions"]["1.0"]["target_rank"] == 2
    assert result["interventions"]["2.0"]["target_rank"] == 1
    assert result["best_intervened_rank"] == 1
    assert result["improved"] is True
    assert result["target_top1"] is True


class FormattingTokenizer(RunnerTokenizer):
    def decode(
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return " " if token_ids[0] == 0 else f"token-{token_ids[0]}"


class SwapTokenizer(RunnerTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.pieces.update(
            {
                " ant": [3],
                "6": [5],
                " 6": [5],
                "six": [5],
                " six": [5],
                "Six": [5],
                " Six": [5],
                "SIX": [5],
                " SIX": [5],
            }
        )


class FormattingSwapTokenizer(SwapTokenizer):
    def decode(
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return " " if token_ids[0] == 0 else f"token-{token_ids[0]}"


class TinySwapModel:
    n_layers = 4
    d_model = 2

    def __init__(self) -> None:
        self.layers = nn.ModuleList(
            [TensorBlock(), TensorBlock(), TensorBlock(), TensorBlock()]
        )

    def encode(self, prompt: str, *, max_length: int = 512) -> torch.Tensor:
        del prompt, max_length
        return torch.tensor([[0, 1]])


def test_analyze_swap_case_runs_clean_and_both_strengths() -> None:
    model = TinySwapModel()
    lens = SimpleNamespace(
        jacobians={2: torch.eye(2)},
        source_layers=[2],
        d_model=2,
    )
    unembedding = torch.zeros(6, 2)
    unembedding[2] = torch.tensor([1.0, 0.0])
    unembedding[3] = torch.tensor([0.0, 1.0])

    def forward_next_token(input_ids: torch.Tensor) -> torch.Tensor:
        del input_ids
        hidden = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        for block in model.layers:
            hidden = block(hidden)
        logits = torch.zeros(6)
        logits[4] = hidden[0, -1, 0]
        logits[5] = hidden[0, -1, 1]
        return logits

    result = analyze_swap_case(
        SwapCase("spider", " spider", " ant", ("6", "six")),
        read_case=ReadoutCase(
            key="spider",
            prompt="prompt",
            expected_answers=("8", "eight"),
            target_concepts=("spider",),
        ),
        model=model,
        lens=lens,
        tokenizer=SwapTokenizer(),
        unembedding_weight=unembedding,
        forward_next_token=forward_next_token,
        layers=[2],
        alphas=(1.0, 2.0),
        top_k=3,
    )

    assert result["source"] == {"surface": " spider", "token_id": 2}
    assert result["target"] == {"surface": " ant", "token_id": 3}
    assert set(result["interventions"]) == {"1.0", "2.0"}
    assert result["improved"] is True
    assert result["target_top1"] is True


def test_scoring_input_appends_only_bounded_clean_formatting_tokens() -> None:
    calls: list[list[int]] = []

    def forward_next_token(input_ids: torch.Tensor) -> torch.Tensor:
        calls.append(input_ids[0].tolist())
        logits = torch.zeros(6)
        logits[0 if input_ids.shape[1] == 1 else 4] = 5.0
        return logits

    scoring_input, prefix = prepare_scoring_input(
        torch.tensor([[9]]),
        forward_next_token=forward_next_token,
        tokenizer=FormattingTokenizer(),
        max_formatting_tokens=2,
    )

    assert scoring_input.tolist() == [[9, 0]]
    assert prefix == [{"token_id": 0, "token": " "}]
    assert calls == [[9], [9, 0]]


def test_capability_gate_requires_three_improvements_and_one_top1() -> None:
    read_results = [
        {"key": "spider", "checks": {"baseline_top1": True, "read_capability": True}},
        {"key": "france_capital", "checks": {"baseline_top1": True}},
        {"key": "france_language", "checks": {"baseline_top1": True}},
        {"key": "france_continent", "checks": {"baseline_top1": True}},
        {"key": "france_currency", "checks": {"baseline_top1": True}},
    ]
    swap_results = [
        {"improved": True, "target_top1": True},
        {"improved": True, "target_top1": False},
        {"improved": True, "target_top1": False},
        {"improved": False, "target_top1": False},
        {"improved": False, "target_top1": False},
    ]

    checks, failures = aggregate_capability_checks(read_results, swap_results)

    assert checks == {
        "clean_baselines": True,
        "spider_read": True,
        "swap_rank_improvements": True,
        "swap_target_top1": True,
    }
    assert failures == []


def test_capability_gate_reports_aggregate_swap_failures() -> None:
    read_results = [
        {"key": "spider", "checks": {"baseline_top1": True, "read_capability": True}},
    ]
    swap_results = [
        {"improved": True, "target_top1": False},
        {"improved": True, "target_top1": False},
        {"improved": False, "target_top1": False},
    ]

    checks, failures = aggregate_capability_checks(read_results, swap_results)

    assert checks["swap_rank_improvements"] is False
    assert checks["swap_target_top1"] is False
    assert failures == [
        "coordinate swaps improved 2/3 target ranks; need at least 3",
        "no coordinate swap placed its target answer at top-1",
    ]


class FakeLens:
    d_model = 4
    source_layers = [0, 1, 2, 3]
    n_prompts = 1000

    def apply(self, model, prompt, *, use_jacobian=True, **kwargs):
        del model, prompt, kwargs
        input_ids = torch.tensor([[0, 1, 3]])
        model_logits = torch.zeros(3, 6)
        model_logits[-1, 4] = 9.0
        layer_logits = {}
        for layer in self.source_layers:
            logits = torch.zeros(3, 6)
            logits[:, 2] = 8.0 if use_jacobian and layer == 2 else -1.0
            layer_logits[layer] = logits
        return layer_logits, model_logits, input_ids


def test_validate_model_lens_rejects_width_and_layer_mismatches() -> None:
    with pytest.raises(ValueError, match="residual width"):
        validate_model_lens(SimpleNamespace(n_layers=4, d_model=5), FakeLens())

    lens = FakeLens()
    lens.source_layers = [0, 4]
    with pytest.raises(ValueError, match="fitted layers"):
        validate_model_lens(SimpleNamespace(n_layers=4, d_model=4), lens)


def test_analyze_case_grades_baseline_and_spider_readout() -> None:
    case = ReadoutCase(
        key="spider",
        prompt="prompt",
        expected_answers=("8", "eight"),
        target_concepts=("spider",),
    )

    result = analyze_case(
        case,
        model=SimpleNamespace(n_layers=4, d_model=4),
        lens=FakeLens(),
        tokenizer=RunnerTokenizer(),
        top_k=3,
    )

    assert result["checks"] == {"read_capability": True}
    assert result["diagnostics"] == {"paper_top1_hit": True}
    assert result["summary"]["jacobian_lens"]["best_rank"] == 1
    assert result["summary"]["jacobian_lens"]["layer"] == 2
    assert result["summary"]["logit_lens"]["best_rank"] > 1
    assert result["passed"] is True


def test_spider_paper_top1_is_diagnostic_not_a_capability_requirement() -> None:
    class RankTwoLens(FakeLens):
        def apply(self, model, prompt, *, use_jacobian=True, **kwargs):
            layer_logits, model_logits, input_ids = super().apply(
                model,
                prompt,
                use_jacobian=use_jacobian,
                **kwargs,
            )
            if use_jacobian:
                layer_logits[2][:, 0] = 9.0
            return layer_logits, model_logits, input_ids

    result = analyze_case(
        ReadoutCase("spider", "prompt", ("8", "eight"), ("spider",)),
        model=SimpleNamespace(n_layers=4, d_model=4),
        lens=RankTwoLens(),
        tokenizer=RunnerTokenizer(),
        top_k=3,
    )

    assert result["summary"]["jacobian_lens"]["best_rank"] == 2
    assert result["diagnostics"]["paper_top1_hit"] is False
    assert result["checks"]["read_capability"] is True
    assert result["passed"] is True


class TinyCompleteLens:
    d_model = 2
    source_layers = [2]
    n_prompts = 1000

    def __init__(self) -> None:
        self.jacobians = {2: torch.eye(2)}

    def apply(self, model, prompt, *, use_jacobian=True, **kwargs):
        del prompt, kwargs
        input_ids = model.encode("prompt")
        model_logits = torch.zeros(input_ids.shape[1], 6)
        model_logits[-1, 0] = 5.0
        readout = torch.zeros(input_ids.shape[1], 6)
        readout[:, 2] = 4.0 if use_jacobian else -1.0
        return {2: readout}, model_logits, input_ids


class FiveCaseTokenizer:
    def __init__(self) -> None:
        self.pieces: dict[str, list[int]] = {}
        self._register_variants("spider", 2)
        self.pieces[" spider"] = [2]
        self.pieces[" ant"] = [3]
        self._register_variants("8", 4)
        self._register_variants("eight", 4)
        self._register_variants("6", 5)
        self._register_variants("six", 5)
        self._register_variants("France", 17)
        self.pieces[" France"] = [17]
        self.pieces[" China"] = [18]
        for surface, token_id in {
            "Paris": 20,
            "French": 21,
            "Europe": 22,
            "Euro": 23,
            "Beijing": 24,
            "Chinese": 25,
            "Asia": 26,
            "Yuan": 27,
        }.items():
            self._register_variants(surface, token_id)
        self.all_special_ids = [60, 61, 62, 63]
        self.added_tokens_decoder: dict[int, object] = {}

    def _register_variants(self, surface: str, token_id: int) -> None:
        for variant in {
            surface,
            surface.lower(),
            surface.capitalize(),
            surface.upper(),
        }:
            self.pieces[variant] = [token_id]
            self.pieces[f" {variant}"] = [token_id]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.pieces.get(text, [58, 59])

    def decode(
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return " " if token_ids[0] == 0 else f"token-{token_ids[0]}"

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{token_id}": token_id for token_id in range(64)}


class FiveCaseModel:
    n_layers = 4
    d_model = 4

    def __init__(self) -> None:
        self.layers = nn.ModuleList([TensorBlock() for _ in range(self.n_layers)])
        self.case_ids = {
            case.prompt: 10 + index for index, case in enumerate(READOUT_CASES)
        }

    def encode(self, prompt: str, *, max_length: int = 512) -> torch.Tensor:
        del max_length
        case_id = self.case_ids[prompt]
        if prompt == READOUT_CASES[0].prompt:
            return torch.tensor([[case_id, 1]])
        return torch.tensor([[case_id, 17, 1]])


class FiveCaseLens:
    d_model = 4
    source_layers = [2]
    n_prompts = 1000

    def __init__(self) -> None:
        self.jacobians = {2: torch.eye(4)}
        self.clean_answer_ids = {
            case.prompt: token_id
            for case, token_id in zip(
                READOUT_CASES,
                (4, 20, 21, 22, 23),
                strict=True,
            )
        }

    def apply(self, model, prompt, *, use_jacobian=True, **kwargs):
        del kwargs
        input_ids = model.encode(prompt)
        model_logits = torch.zeros(input_ids.shape[1], 64)
        model_logits[-1, self.clean_answer_ids[prompt]] = 5.0
        readout = torch.zeros(input_ids.shape[1], 64)
        concept_id = 2 if prompt == READOUT_CASES[0].prompt else 17
        readout[:, concept_id] = 4.0 if use_jacobian else -1.0
        return {2: readout}, model_logits, input_ids


@pytest.mark.parametrize("alphas", [(1.0, 2.0), (1, 2)])
def test_run_readout_sanity_integrates_all_controls_without_storing_logits(
    monkeypatch: pytest.MonkeyPatch,
    alphas: tuple[float, float],
) -> None:
    model = FiveCaseModel()
    lens = FiveCaseLens()
    tokenizer = FiveCaseTokenizer()
    unembedding = torch.zeros(64, 4)
    unembedding[2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    unembedding[3] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    unembedding[17] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    unembedding[18] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    key_by_id = {value: key for key, value in model.case_ids.items()}
    clean_ids = dict(zip(key_by_id.values(), (4, 20, 21, 22, 23), strict=True))
    target_ids = dict(zip(key_by_id.values(), (5, 24, 25, 26, 27), strict=True))
    execution_calls: list[dict[str, object]] = []
    real_execute_intervention = intervention_utils_module.execute_intervention
    vector_calls: list[dict[str, int]] = []
    real_jlens_vector = intervention_utils_module.jlens_vector

    def recording_jlens_vector(*args, **kwargs):
        vector_calls.append(
            {
                "layer": kwargs["layer"],
                "token_id": kwargs["token_id"],
            }
        )
        return real_jlens_vector(*args, **kwargs)

    monkeypatch.setattr(
        intervention_utils_module,
        "jlens_vector",
        recording_jlens_vector,
    )

    def recording_execute_intervention(**kwargs):
        execution_calls.append(
            {
                "alpha": kwargs["alpha"],
                "vectors": {
                    layer: (
                        pair[0].detach().clone(),
                        pair[1].detach().clone(),
                    )
                    for layer, pair in kwargs["vectors_by_layer"].items()
                },
            }
        )
        return real_execute_intervention(**kwargs)

    monkeypatch.setattr(
        intervention_utils_module,
        "execute_intervention",
        recording_execute_intervention,
    )
    monkeypatch.setattr(
        runner_module,
        "execute_intervention",
        recording_execute_intervention,
    )
    monkeypatch.setattr(
        control_execution_module,
        "execute_intervention",
        recording_execute_intervention,
    )

    def forward_next_token(input_ids: torch.Tensor) -> torch.Tensor:
        prompt = key_by_id[int(input_ids[0, 0])]
        base_length = 2 if prompt == READOUT_CASES[0].prompt else 3
        source = (
            torch.tensor([1.0, 0.0, 0.0, 0.0])
            if prompt == READOUT_CASES[0].prompt
            else torch.tensor([0.0, 0.0, 1.0, 0.0])
        )
        hidden = source.view(1, 1, 4).expand(1, input_ids.shape[1], 4)
        for block in model.layers:
            hidden = block(hidden)
        logits = torch.zeros(64)
        if input_ids.shape[1] == base_length:
            logits[0] = 10.0
            return logits
        logits[clean_ids[prompt]] = 5.0
        target_coordinate = 1 if prompt == READOUT_CASES[0].prompt else 3
        logits[target_ids[prompt]] = hidden[0, -1, target_coordinate] * 10.0
        return logits

    result = run_readout_sanity(
        model=model,
        lens=lens,
        tokenizer=tokenizer,
        unembedding_weight=unembedding,
        forward_next_token=forward_next_token,
        cases=READOUT_CASES,
        swap_cases=SWAP_CASES,
        alphas=alphas,
        top_k=3,
    )

    assert result["policy"] == {
        "clean_baselines": {"required_count": 5},
        "spider_read": {
            "maximum_rank": 5,
            "requires_better_than_logit_lens": True,
            "paper_target_rank": 1,
        },
        "swap_rank_improvements": {
            "required_count": 3,
            "case_count": 5,
        },
        "swap_target_top1": {
            "required_count": 1,
            "case_count": 5,
            "paper_primary_alpha": 1.0,
            "paper_target_rank": 1,
        },
    }

    controls = result["controls"]
    selected_target_ids = {
        target["token_id"] for target in controls["random_target"]["targets"]
    }
    assert len(selected_target_ids) == len(CONTROL_SEEDS)
    assert len(vector_calls) == 10 + len(CONTROL_SEEDS)
    assert all(
        sum(call["token_id"] == token_id for call in vector_calls) == 1
        for token_id in selected_target_ids
    )
    assert controls["seeds"] == list(CONTROL_SEEDS)
    assert set(controls) >= {
        "identity",
        "matched_random_vector",
        "wrong_concept",
        "random_target",
        "passed",
    }
    assert controls["identity"]["passed"] is True
    assert "seeds" not in controls["matched_random_vector"]["configuration"]
    assert "seeds" not in controls["random_target"]["configuration"]
    assert len(controls["matched_random_vector"]["seeds"]) == 16
    assert len(controls["random_target"]["targets"]) == 16
    assert all(
        len(item["cases"]) == 5 for item in controls["matched_random_vector"]["seeds"]
    )
    assert all(len(item["cases"]) == 5 for item in controls["random_target"]["targets"])
    assert all(
        item["token_id"] < unembedding.shape[0]
        for item in controls["random_target"]["targets"]
    )
    expected_keys = [case.key for case in SWAP_CASES]
    expected_intended_ids = dict(zip(expected_keys, (5, 24, 25, 26, 27), strict=True))
    for seed_result in controls["matched_random_vector"]["seeds"]:
        assert [case["key"] for case in seed_result["cases"]] == expected_keys
        assert all(
            case["intended_target_ids"] == [expected_intended_ids[case["key"]]]
            for case in seed_result["cases"]
        )
    for target_result in controls["random_target"]["targets"]:
        assert [case["key"] for case in target_result["cases"]] == expected_keys
        assert all(
            case["intended_target_ids"] == [expected_intended_ids[case["key"]]]
            for case in target_result["cases"]
        )
    assert [case["key"] for case in controls["wrong_concept"]["cases"]] == (
        expected_keys
    )
    assert controls["wrong_concept"]["configuration"]["mismatches"] == [
        {
            "key": "spider",
            "source": {"surface": " France", "token_id": 17},
            "target": {"surface": " China", "token_id": 18},
        },
        *[
            {
                "key": key,
                "source": {"surface": " spider", "token_id": 2},
                "target": {"surface": " ant", "token_id": 3},
            }
            for key in expected_keys[1:]
        ],
    ]
    real_gains = []
    for swap in result["swaps"]:
        alpha_one = next(
            payload
            for alpha, payload in swap["interventions"].items()
            if float(alpha) == 1.0
        )
        real_gains.append(
            log_rank_gain(
                swap["clean"]["target_rank"],
                alpha_one["target_rank"],
            )
        )
    assert controls["matched_random_vector"][
        "real_mean_log_rank_gain"
    ] == pytest.approx(sum(real_gains) / 5)
    for name in ("matched_random_vector", "random_target"):
        control = controls[name]
        assert control["passed"] is (
            control["real_mean_log_rank_gain"] > control["percentile_95_threshold"]
        )

    assert [call["alpha"] for call in execution_calls[:10]] == [1.0, 2.0] * 5
    assert all(call["alpha"] == 1.0 for call in execution_calls[10:])
    assert len(execution_calls) == 180
    for call in execution_calls[10:15]:
        source_vector, target_vector = call["vectors"][2]
        assert torch.equal(source_vector, target_vector)
    wrong_targets = [torch.tensor([0.0, 0.0, 0.0, 1.0])] + [
        torch.tensor([0.0, 1.0, 0.0, 0.0]) for _ in range(4)
    ]
    for call, expected_target in zip(
        execution_calls[95:100], wrong_targets, strict=True
    ):
        assert torch.equal(call["vectors"][2][1], expected_target)
    first_random_target_id = controls["random_target"]["targets"][0]["token_id"]
    expected_random_vectors = [
        _token_vectors_by_layer(
            lens=lens,
            unembedding_weight=unembedding,
            layers=[2],
            source_token_id=source_id,
            target_token_id=first_random_target_id,
        )[2][1]
        for source_id in (2, 17, 17, 17, 17)
    ]
    for call, expected_vector in zip(
        execution_calls[100:105], expected_random_vectors, strict=True
    ):
        assert torch.equal(call["vectors"][2][1], expected_vector)
    assert set(result["checks"]) >= {
        "identity_control",
        "matched_random_vector_control",
        "wrong_concept_control",
        "random_target_control",
    }

    def contains_tensor(value: object) -> bool:
        if torch.is_tensor(value):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_tensor(item) for item in value)
        return False

    assert contains_tensor(controls) is False


def test_run_readout_sanity_rejects_missing_control_cases() -> None:
    model = TinySwapModel()
    lens = TinyCompleteLens()
    tokenizer = FormattingSwapTokenizer()
    unembedding = torch.zeros(6, 2)
    unembedding[2] = torch.tensor([1.0, 0.0])
    unembedding[3] = torch.tensor([0.0, 1.0])

    def forward_next_token(input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] == 2:
            logits = torch.zeros(6)
            logits[0] = 10.0
            return logits
        hidden = torch.tensor([[[1.0, 0.0]]]).expand(1, input_ids.shape[1], 2)
        for block in model.layers:
            hidden = block(hidden)
        logits = torch.zeros(6)
        logits[4] = hidden[0, -1, 0]
        logits[5] = hidden[0, -1, 1]
        return logits

    with pytest.raises(ValueError, match="exact five"):
        run_readout_sanity(
            model=model,
            lens=lens,
            tokenizer=tokenizer,
            unembedding_weight=unembedding,
            forward_next_token=forward_next_token,
            cases=(
                ReadoutCase(
                    key="spider",
                    prompt="prompt",
                    expected_answers=("8", "eight"),
                    target_concepts=("spider",),
                ),
            ),
            swap_cases=(SwapCase("spider", " spider", " ant", ("6", "six")),),
            minimum_improvements=1,
            top_k=3,
        )


def test_run_requires_configured_control_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "CONTROL_ALPHA", 2.0, raising=False)
    monkeypatch.setattr(
        runner_module,
        "resolve_swap_cases",
        lambda *args, **kwargs: (),
    )
    model = SimpleNamespace(d_model=2, n_layers=4)
    lens = SimpleNamespace(d_model=2, source_layers=(2,))

    with pytest.raises(ValueError, match="alpha=2"):
        runner_module.run_readout_sanity(
            model=model,
            lens=lens,
            tokenizer=SimpleNamespace(),
            unembedding_weight=torch.zeros(6, 2),
            forward_next_token=lambda input_ids: torch.zeros(6),
            cases=READOUT_CASES,
            swap_cases=SWAP_CASES,
            alphas=(1.0,),
        )


def test_run_validates_swap_surfaces_before_lens_forwards() -> None:
    class CountingLens(TinyCompleteLens):
        def __init__(self) -> None:
            super().__init__()
            self.apply_calls = 0

        def apply(self, *args, **kwargs):
            self.apply_calls += 1
            return super().apply(*args, **kwargs)

    lens = CountingLens()

    with pytest.raises(ValueError, match="exactly one token"):
        run_readout_sanity(
            model=TinySwapModel(),
            lens=lens,
            tokenizer=SwapTokenizer(),
            unembedding_weight=torch.zeros(6, 2),
            forward_next_token=lambda input_ids: torch.zeros(6),
            cases=READOUT_CASES,
            swap_cases=(
                SwapCase("spider", " spider", " bad surface", ("6",)),
                *SWAP_CASES[1:],
            ),
            minimum_improvements=1,
        )

    assert lens.apply_calls == 0
