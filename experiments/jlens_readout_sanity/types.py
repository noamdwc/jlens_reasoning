"""Data contracts for the J-Lens readout sanity experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from jlens_reasoning.experiments_utils.tokens import TokenVariant


@dataclass(frozen=True, slots=True)
class ReadoutCase:
    key: str
    prompt: str
    expected_answers: tuple[str, ...]
    target_concepts: tuple[str, ...]
    literal_argument: str | None = None


@dataclass(frozen=True, slots=True)
class SwapCase:
    key: str
    source_surface: str
    target_surface: str
    target_answers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolvedSwapCase:
    case: SwapCase
    read_case: ReadoutCase
    source: TokenVariant
    target: TokenVariant


@dataclass(slots=True)
class InterventionContext:
    resolved: ResolvedSwapCase
    input_ids: torch.Tensor
    scoring_input: torch.Tensor
    formatting_prefix: list[dict[str, Any]]
    clean_logits: torch.Tensor
    target_ids: tuple[int, ...]
    real_vectors_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]]
    workspace_loading: float | None
