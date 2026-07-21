import json
from pathlib import Path

import torch

from jlens_reasoning.experiments_utils.artifacts import write_results


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
