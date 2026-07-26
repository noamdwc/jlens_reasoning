import json
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

import torch

from jlens_reasoning.experiments_utils.artifacts import write_results


@dataclass(frozen=True)
class TypedArtifact:
    rank: torch.Tensor
    layers: tuple[int, ...]


@dataclass(frozen=True)
class DerivedStatusArtifact:
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def test_write_results_is_json_ready_and_byte_stable(tmp_path: Path) -> None:
    result = {
        "rank": torch.tensor(3),
        "layers": torch.tensor([7, 8]),
        "path": Path("runs/result.json"),
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_results(first, result)
    write_results(second, result)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == {
        "layers": [7, 8],
        "path": "runs/result.json",
        "rank": 3,
    }


def test_write_results_accepts_typed_experiment_results(tmp_path: Path) -> None:
    output = tmp_path / "typed.json"

    write_results(output, TypedArtifact(torch.tensor(3), (7, 8)))

    assert get_type_hints(write_results)["result"] is object
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "layers": [7, 8],
        "rank": 3,
    }


def test_write_results_includes_derived_top_level_passed(tmp_path: Path) -> None:
    output = tmp_path / "derived-status.json"

    write_results(
        output,
        DerivedStatusArtifact({"clean_baselines": True, "identity_control": False}),
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "checks": {
            "clean_baselines": True,
            "identity_control": False,
        },
        "passed": False,
    }
