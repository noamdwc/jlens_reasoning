"""Fixed readout cases and their token-resolution helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


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


READOUT_CASES = (
    ReadoutCase(
        key="spider",
        prompt="The number of legs on the animal that spins webs is",
        expected_answers=("8", "eight"),
        target_concepts=("spider",),
    ),
    ReadoutCase(
        key="france_capital",
        prompt="The capital of France is the city of",
        expected_answers=("Paris",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_language",
        prompt="Most people in France speak",
        expected_answers=("French",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_continent",
        prompt="France is a country on the continent of",
        expected_answers=("Europe",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_currency",
        prompt="The single-word name for the currency now used in France is the",
        expected_answers=("Euro",),
        target_concepts=("France",),
        literal_argument="France",
    ),
)


SWAP_CASES = (
    SwapCase("spider", " spider", " ant", ("6", "six")),
    SwapCase("france_capital", " France", " China", ("Beijing",)),
    SwapCase("france_language", " France", " China", ("Chinese",)),
    SwapCase("france_continent", " France", " China", ("Asia",)),
    SwapCase("france_currency", " France", " China", ("Yuan",)),
)


@dataclass(frozen=True, slots=True)
class TokenVariant:
    token_id: int
    surface: str


@dataclass(frozen=True, slots=True)
class ResolvedSwapCase:
    case: SwapCase
    read_case: ReadoutCase
    source: TokenVariant
    target: TokenVariant


def single_token_surface(tokenizer: Any, surface: str) -> TokenVariant:
    token_ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Configured swap surface {surface!r} must encode as exactly one token"
        )
    return TokenVariant(token_id=token_ids[0], surface=surface)


def resolve_swap_cases(
    cases: Sequence[ReadoutCase],
    swap_cases: Sequence[SwapCase],
    tokenizer: Any,
) -> tuple[ResolvedSwapCase, ...]:
    read_cases_by_key: dict[str, ReadoutCase] = {}
    for case in cases:
        if case.key in read_cases_by_key:
            raise ValueError(f"Duplicate readout case key: {case.key}")
        read_cases_by_key[case.key] = case

    seen_swap_keys: set[str] = set()
    resolved = []
    for case in swap_cases:
        if case.key in seen_swap_keys:
            raise ValueError(f"Duplicate swap case key: {case.key}")
        seen_swap_keys.add(case.key)
        if case.key not in read_cases_by_key:
            raise ValueError(f"Swap case has no matching readout case: {case.key}")
        resolved.append(
            ResolvedSwapCase(
                case=case,
                read_case=read_cases_by_key[case.key],
                source=single_token_surface(tokenizer, case.source_surface),
                target=single_token_surface(tokenizer, case.target_surface),
            )
        )

    missing_swap_keys = set(read_cases_by_key) - seen_swap_keys
    if missing_swap_keys:
        missing = ", ".join(sorted(missing_swap_keys))
        raise ValueError(f"Readout cases have no matching swap case: {missing}")
    return tuple(resolved)


def _concept_surfaces(concept: str) -> tuple[str, ...]:
    bases = (concept, concept.lower(), concept.capitalize(), concept.upper())
    ordered: list[str] = []
    for base in bases:
        for surface in (base, f" {base}"):
            if surface not in ordered:
                ordered.append(surface)
    return tuple(ordered)


def concept_token_variants(
    tokenizer: Any, concepts: Sequence[str]
) -> tuple[TokenVariant, ...]:
    variants: list[TokenVariant] = []
    seen_ids: set[int] = set()
    for concept in concepts:
        for surface in _concept_surfaces(concept):
            token_ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(token_ids) == 1 and token_ids[0] not in seen_ids:
                seen_ids.add(token_ids[0])
                variants.append(TokenVariant(token_id=token_ids[0], surface=surface))
    if not variants:
        raise ValueError(f"No single-token variants found for {tuple(concepts)!r}")
    return tuple(variants)
