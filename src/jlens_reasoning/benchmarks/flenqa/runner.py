"""Stream a prepared FLenQA dataset through paired lens passes."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from tqdm.auto import tqdm

from jlens_reasoning.benchmarks.flenqa.dataset import FlenqaRow, deduplicate
from jlens_reasoning.benchmarks.flenqa.lens import LensRunners, run_prompt
from jlens_reasoning.benchmarks.flenqa.positions import PreparedPrompt, prepare_prompt
from jlens_reasoning.benchmarks.flenqa.storage import REQUIRED_TABLES, TABLE_SCHEMAS


@dataclass(frozen=True, slots=True)
class RunConfig:
    top_k: int = 25
    padding_sample_seed: int = 1729
    shard_size: int = 500
    max_seq_len: int = 4096
    logits_rtol: float = 1e-5
    logits_atol: float = 1e-6
    expected_source_rows: int = 12_000


@dataclass(frozen=True, slots=True)
class RunSummary:
    prompt_count: int
    returned_layers: tuple[int, ...]
    max_abs_logit_diff: float


def _validate_config(config: RunConfig) -> None:
    if type(config.top_k) is not int or config.top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if type(config.shard_size) is not int or config.shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    if type(config.max_seq_len) is not int or config.max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer")
    if type(config.expected_source_rows) is not int or config.expected_source_rows <= 0:
        raise ValueError("expected_source_rows must be a positive integer")
    if config.logits_rtol < 0 or config.logits_atol < 0:
        raise ValueError("logit tolerances must be non-negative")


def _contains_output(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and next(path.iterdir(), None) is not None)


def _prepare_output(root: Path) -> None:
    table_paths = [root / table for table in REQUIRED_TABLES]
    if any(path.exists() and _contains_output(path) for path in table_paths):
        raise FileExistsError(f"{root} already contains FLenQA output")
    root.mkdir(parents=True, exist_ok=True)
    for table in REQUIRED_TABLES:
        (root / table).mkdir(exist_ok=True)


def _chunks(
    prompts: Sequence[PreparedPrompt],
    size: int,
) -> Iterator[Sequence[PreparedPrompt]]:
    for start in range(0, len(prompts), size):
        yield prompts[start : start + size]


def _write_shard(
    prompts: Sequence[PreparedPrompt],
    *,
    shard_id: int,
    root: Path,
    runners: LensRunners,
    config: RunConfig,
    on_prompt_completed: Callable[[], None],
) -> tuple[tuple[int, ...], float]:
    stem = f"shard-{shard_id:05d}.parquet"
    writers = {
        table: pq.ParquetWriter(root / table / stem, TABLE_SCHEMAS[table])
        for table in REQUIRED_TABLES
    }
    returned_layers: tuple[int, ...] | None = None
    max_abs_logit_diff = 0.0
    try:
        for prepared in prompts:
            result = run_prompt(
                prepared,
                runners=runners,
                top_k=config.top_k,
                max_seq_len=config.max_seq_len,
                logits_rtol=config.logits_rtol,
                logits_atol=config.logits_atol,
            )
            if returned_layers is None:
                returned_layers = result.returned_layers
            elif returned_layers != result.returned_layers:
                raise RuntimeError("Returned layer keys differ between prompts")
            max_abs_logit_diff = max(
                max_abs_logit_diff,
                result.max_abs_logit_diff,
            )
            for table in REQUIRED_TABLES:
                writers[table].write_batch(result.batches[table])
            on_prompt_completed()
    finally:
        for writer in writers.values():
            writer.close()
    if returned_layers is None:
        raise RuntimeError("FLenQA shards must contain at least one prompt")
    return returned_layers, max_abs_logit_diff


def run_benchmark(
    rows: Sequence[FlenqaRow],
    *,
    output_dir: Path,
    tokenizer: object,
    runners: LensRunners,
    config: RunConfig,
    show_progress: bool = True,
) -> RunSummary:
    """Prepare and stream one non-resumable FLenQA benchmark run."""
    _validate_config(config)
    if len(rows) != config.expected_source_rows:
        raise ValueError(
            f"Expected {config.expected_source_rows} source rows; found {len(rows)}"
        )
    prompts = deduplicate(rows)
    if not prompts:
        raise ValueError("FLenQA benchmark requires at least one prompt")
    prepared = tuple(
        prepare_prompt(
            prompt,
            tokenizer,
            max_seq_len=config.max_seq_len,
            sample_seed=config.padding_sample_seed,
        )
        for prompt in prompts
    )
    root = Path(output_dir)
    _prepare_output(root)

    returned_layers: tuple[int, ...] | None = None
    max_abs_logit_diff = 0.0
    with tqdm(
        total=len(prepared),
        desc="FLenQA prompts",
        unit="prompt",
        disable=not show_progress,
    ) as progress:
        for shard_id, shard in enumerate(_chunks(prepared, config.shard_size)):
            shard_layers, shard_max_diff = _write_shard(
                shard,
                shard_id=shard_id,
                root=root,
                runners=runners,
                config=config,
                on_prompt_completed=progress.update,
            )
            if returned_layers is None:
                returned_layers = shard_layers
            elif returned_layers != shard_layers:
                raise RuntimeError("Returned layer keys differ between shards")
            max_abs_logit_diff = max(max_abs_logit_diff, shard_max_diff)

    if returned_layers is None:
        raise RuntimeError("FLenQA benchmark requires at least one prepared prompt")
    return RunSummary(
        prompt_count=len(prepared),
        returned_layers=returned_layers,
        max_abs_logit_diff=max_abs_logit_diff,
    )
