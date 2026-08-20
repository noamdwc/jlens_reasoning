"""Stream FLenQA prompts through paired lens passes into Parquet shards."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from tqdm.auto import tqdm

from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaPrompt,
    FlenqaRow,
    prepare_prompts,
)
from jlens_reasoning.benchmarks.flenqa.lens import (
    LensRunners,
    PromptResult,
    run_prompt,
)
from jlens_reasoning.benchmarks.flenqa.positions import prepare_prompt
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
    prompts: Sequence[FlenqaPrompt],
    size: int,
) -> Iterator[Sequence[FlenqaPrompt]]:
    for start in range(0, len(prompts), size):
        yield prompts[start : start + size]


class _ShardWriter:
    """Write prompt results directly to final Parquet shard files."""

    def __init__(self, *, root: Path, shard_id: int) -> None:
        stem = f"shard-{shard_id:05d}.parquet"
        self._paths = {table: root / table / stem for table in REQUIRED_TABLES}
        self._writers: dict[str, pq.ParquetWriter] = {}

    def __enter__(self) -> _ShardWriter:
        try:
            for table, path in self._paths.items():
                self._writers[table] = pq.ParquetWriter(path, TABLE_SCHEMAS[table])
        except BaseException:
            self._close()
            raise
        return self

    def write(self, result: PromptResult) -> None:
        for table in REQUIRED_TABLES:
            self._writers[table].write_batch(result.batches[table])

    def __exit__(self, *_: object) -> None:
        self._close()

    def _close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


def run_benchmark(
    rows: Sequence[FlenqaRow],
    *,
    output_dir: Path,
    tokenizer: object,
    runners: LensRunners,
    config: RunConfig,
    show_progress: bool = True,
) -> RunSummary:
    """Run one non-resumable FLenQA benchmark into empty table directories."""
    _validate_config(config)
    if len(rows) != config.expected_source_rows:
        raise ValueError(
            f"Expected {config.expected_source_rows} source rows; found {len(rows)}"
        )
    prompts = prepare_prompts(rows)
    if not prompts:
        raise ValueError("FLenQA benchmark requires at least one prompt")

    root = Path(output_dir)
    _prepare_output(root)

    returned_layers: tuple[int, ...] | None = None
    max_abs_logit_diff = 0.0
    with tqdm(
        total=len(prompts),
        desc="FLenQA prompts",
        unit="prompt",
        disable=not show_progress,
    ) as progress:
        for shard_id, shard in enumerate(_chunks(prompts, config.shard_size)):
            with _ShardWriter(root=root, shard_id=shard_id) as writer:
                for prompt in shard:
                    prepared_prompt = prepare_prompt(
                        prompt,
                        tokenizer,
                        max_seq_len=config.max_seq_len,
                        sample_seed=config.padding_sample_seed,
                    )
                    result = run_prompt(
                        prepared_prompt,
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
                    writer.write(result)
                    progress.update()

    if returned_layers is None:
        raise RuntimeError("FLenQA benchmark requires at least one prepared prompt")
    return RunSummary(
        prompt_count=len(prompts),
        returned_layers=returned_layers,
        max_abs_logit_diff=max_abs_logit_diff,
    )
