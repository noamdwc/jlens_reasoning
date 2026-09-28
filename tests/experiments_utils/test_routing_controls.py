import pytest
import torch
from torch import nn

from jlens_reasoning.experiments_utils.controls import random_orthogonal_direction
from jlens_reasoning.experiments_utils.interventions import add_prompt_delta
from jlens_reasoning.experiments_utils.statistics import paired_problem_bootstrap
from jlens_reasoning.inference import (
    InferenceConfig,
    generate_chat,
    prepare_chat_inputs,
)


def test_random_control_is_orthogonal_matched_and_reproducible():
    axis = torch.tensor([0.3, -0.7, 2.1], dtype=torch.float64)
    direction = random_orthogonal_direction(axis, norm=0.17, seed=42)
    assert float(direction @ axis) == pytest.approx(0, abs=1e-14)
    assert direction.norm().item() == pytest.approx(0.17)
    torch.testing.assert_close(
        direction, random_orthogonal_direction(axis, norm=0.17, seed=42)
    )
    assert not torch.equal(
        direction, random_orthogonal_direction(axis, norm=0.17, seed=43)
    )
    assert random_orthogonal_direction(axis.bfloat16(), seed=42).dtype == torch.float32
    assert random_orthogonal_direction(axis, norm=0, seed=42).norm() == 0
    with pytest.raises(ValueError, match="nonzero"):
        random_orthogonal_direction(torch.zeros(3), seed=42)
    with pytest.raises(ValueError, match="at least two"):
        random_orthogonal_direction(torch.ones(1), seed=42)


@pytest.mark.parametrize("as_tuple", [False, True])
def test_delta_edits_one_prompt_token_once_and_cleans_up(as_tuple):
    class Block(nn.Module):
        def forward(self, hidden):
            return (hidden, "cache") if as_tuple else hidden

    block = Block()
    hidden = torch.zeros(1, 4, 3)
    delta = torch.tensor([1.0, 2.0, 3.0])
    with add_prompt_delta(block, delta, prompt_length=4, token_position=2):
        output = block(hidden)
        patched = output[0] if as_tuple else output
        assert torch.equal(patched[0, 2], delta)
        assert torch.count_nonzero(patched[0, [0, 1, 3]]) == 0
        assert torch.count_nonzero(hidden) == 0
        # Both cached decoding and a repeated full forward stay untouched.
        for next_hidden in (torch.zeros(1, 1, 3), hidden):
            output = block(next_hidden)
            assert torch.equal(output[0] if as_tuple else output, next_hidden)
    assert not block._forward_hooks


def test_delta_rejects_wrong_prompt_and_cleans_up_on_error():
    block = nn.Identity()
    with pytest.raises(ValueError, match="full prompt"):
        with add_prompt_delta(block, torch.ones(3), prompt_length=4):
            block(torch.zeros(1, 1, 3))
    assert not block._forward_hooks
    with pytest.raises(RuntimeError, match="never called"):
        with add_prompt_delta(block, torch.ones(3), prompt_length=4):
            pass
    assert not block._forward_hooks
    with pytest.raises(RuntimeError, match="generation failed"):
        with add_prompt_delta(block, torch.ones(3), prompt_length=4):
            raise RuntimeError("generation failed")
    assert not block._forward_hooks


def test_prompt_delta_integrates_with_generation(model, tokenizer):
    config = InferenceConfig.direct(max_new_tokens=3)
    clean = generate_chat(model, tokenizer, "cat", config=config)
    length = prepare_chat_inputs(tokenizer, "cat", config=config)["input_ids"].shape[1]
    with add_prompt_delta(model.model.layers[0], torch.zeros(8), prompt_length=length):
        identity = generate_chat(model, tokenizer, "cat", config=config)
    assert identity == clean
    # Keep generating to check that the hook ignores decode calls.
    model.generation_config.eos_token_id = None
    seen = []
    with add_prompt_delta(model.model.layers[0], torch.ones(8), prompt_length=length):
        handle = model.model.layers[0].register_forward_hook(
            lambda module, args, output: seen.append(output.detach().clone())
        )
        try:
            changed = generate_chat(model, tokenizer, "cat", config=config)
        finally:
            handle.remove()
    assert changed.generated_token_count == 3
    assert [state.shape[1] for state in seen] == [length, 1, 1]
    assert not model.model.layers[0]._forward_hooks


def test_bootstrap_weights_problems_equally_not_variants():
    result = paired_problem_bootstrap(["a", "a", "a", "b"], [1, 1, 1, -1], seed=4)
    balanced = paired_problem_bootstrap(["a", "b"], [1, -1], seed=4)
    assert result == balanced
    assert result.mean == 0
    assert result.n_problems == 2
    assert (result.low, result.high) == (-1, 1)
    identical = paired_problem_bootstrap([1, 2, 3], [0.25, 0.25, 0.25], seed=4)
    assert identical.mean == identical.low == identical.high == 0.25


def test_bootstrap_rejects_invalid_sampling_units():
    with pytest.raises(ValueError, match="two distinct"):
        paired_problem_bootstrap([1, 1], [0, 1], seed=4)
    with pytest.raises(ValueError, match="one finite"):
        paired_problem_bootstrap([1, 2], [0, float("nan")], seed=4)
