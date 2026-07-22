import torch
from torch import nn

from experiments.jlens_readout_sanity.control_execution import analyze_identity_case


class TensorBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TinySwapModel:
    def __init__(self) -> None:
        self.layers = nn.ModuleList([TensorBlock() for _ in range(4)])


def test_identity_case_uses_all_workspace_hooks_and_preserves_outputs() -> None:
    model = TinySwapModel()
    scoring_input = torch.tensor([[0, 1]])
    vectors = {
        1: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
        2: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
    }
    observed_hook_layers: list[tuple[int, ...]] = []

    def forward_next_token(input_ids: torch.Tensor) -> torch.Tensor:
        del input_ids
        hidden = torch.tensor([[[1.0, 0.0], [0.25, 0.75]]])
        active_layers = tuple(
            index for index, block in enumerate(model.layers) if block._forward_hooks
        )
        observed_hook_layers.append(active_layers)
        for block in model.layers:
            hidden = block(hidden)
        logits = torch.zeros(6)
        logits[4] = hidden[0, -1, 0]
        logits[5] = hidden[0, -1, 1]
        return logits

    clean_logits = forward_next_token(scoring_input)
    result = analyze_identity_case(
        key="spider",
        clean_logits=clean_logits,
        scoring_input=scoring_input,
        model=model,
        forward_next_token=forward_next_token,
        real_vectors_by_layer=vectors,
        target_ids=(5,),
    )

    assert observed_hook_layers == [(), (1, 2)]
    assert result["workspace_layers"] == [1, 2]
    assert result["top1_unchanged"] is True
    assert result["target_rank_unchanged"] is True
    assert result["logits_close"] is True
    assert result["maximum_absolute_logit_difference"] == 0.0
    assert result["atol"] == 1e-6
    assert result["rtol"] == 1e-5
    assert result["passed"] is True
    assert all(not block._forward_hooks for block in model.layers)


def test_identity_case_fails_when_logits_exceed_tolerance() -> None:
    model = TinySwapModel()

    def forward_next_token(input_ids: torch.Tensor) -> torch.Tensor:
        del input_ids
        logits = torch.tensor([1.0, 0.5, 0.0])
        if any(block._forward_hooks for block in model.layers):
            logits[1] += 1e-4
        return logits

    result = analyze_identity_case(
        key="spider",
        clean_logits=forward_next_token(torch.tensor([[0, 1]])),
        scoring_input=torch.tensor([[0, 1]]),
        model=model,
        forward_next_token=forward_next_token,
        real_vectors_by_layer={2: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))},
        target_ids=(1,),
    )

    assert result["top1_unchanged"] is True
    assert result["target_rank_unchanged"] is True
    assert result["logits_close"] is False
    assert result["passed"] is False
