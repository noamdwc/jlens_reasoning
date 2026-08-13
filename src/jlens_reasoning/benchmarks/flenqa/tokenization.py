"""Validate one full-prompt tokenization with character offsets."""

from __future__ import annotations

import hashlib
import operator
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenizedText:
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    signature: str
    special_token_ids: frozenset[int]


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    tolist = getattr(value, "tolist", None)
    value = tolist() if callable(tolist) else value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise ValueError(f"tokenizer {name} must be a sequence")


def _single_batch(value: Any, *, name: str) -> Sequence[Any]:
    values = _sequence(value, name=name)
    if values and isinstance(values[0], Sequence) and not isinstance(
        values[0], (str, bytes)
    ):
        if len(values) != 1:
            raise ValueError(f"tokenizer {name} must contain exactly one batch")
        return _sequence(values[0], name=f"{name} batch")
    return values


def _input_ids(value: Any) -> tuple[int, ...]:
    ids: list[int] = []
    for token_id in _single_batch(value, name="input_ids"):
        try:
            ids.append(operator.index(token_id))
        except TypeError as exc:
            raise ValueError("tokenizer input_ids must contain integers") from exc
    if not ids:
        raise ValueError("tokenizer input_ids must be nonempty")
    return tuple(ids)


def _offsets(value: Any) -> tuple[tuple[int, int], ...]:
    raw = _sequence(value, name="offset_mapping")
    is_offset = lambda item: (  # noqa: E731 - keeps the shape check local
        isinstance(item, Sequence)
        and not isinstance(item, (str, bytes))
        and len(item) == 2
        and not isinstance(item[0], Sequence)
    )
    if raw and not is_offset(raw[0]):
        if len(raw) != 1:
            raise ValueError(
                "tokenizer offset_mapping must contain exactly one batch"
            )
        raw = _sequence(raw[0], name="offset_mapping batch")

    offsets: list[tuple[int, int]] = []
    for item in raw:
        if not is_offset(item):
            raise ValueError("tokenizer offsets must be integer pairs")
        try:
            start = operator.index(item[0])
            end = operator.index(item[1])
        except TypeError as exc:
            raise ValueError("tokenizer offsets must be integer pairs") from exc
        if start < 0 or end < start:
            raise ValueError("tokenizer offsets must be non-negative and ordered")
        offsets.append((start, end))
    return tuple(offsets)


def _signature(input_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    try:
        for token_id in input_ids:
            digest.update(struct.pack(">q", token_id))
    except struct.error as exc:
        raise ValueError("tokenizer token ID is outside signed 64-bit range") from exc
    return digest.hexdigest()


def tokenize_with_offsets(text: str, tokenizer: Any) -> TokenizedText:
    """Tokenize once without truncation and validate the returned offset map."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        return_offsets_mapping=True,
    )
    if not isinstance(encoded, Mapping) and not hasattr(encoded, "__getitem__"):
        raise ValueError("tokenizer output must provide input_ids and offset_mapping")
    try:
        input_ids = _input_ids(encoded["input_ids"])
        offsets = _offsets(encoded["offset_mapping"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "tokenizer output must provide input_ids and offset_mapping"
        ) from exc

    if len(offsets) != len(input_ids):
        raise ValueError(
            "tokenizer input_ids and offset_mapping must have the same length"
        )

    previous = (0, 0)
    for start, end in offsets:
        if end > len(text):
            raise ValueError("tokenizer offset extends beyond the prompt text")
        if end > start:
            if start < previous[0] or end < previous[1]:
                raise ValueError("tokenizer nonzero offsets must be monotonic")
            previous = (start, end)

    raw_special_ids = getattr(tokenizer, "all_special_ids", ())
    special_ids = () if raw_special_ids is None else raw_special_ids
    return TokenizedText(
        input_ids=input_ids,
        offsets=offsets,
        signature=_signature(input_ids),
        special_token_ids=frozenset(int(token_id) for token_id in special_ids),
    )
