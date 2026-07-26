from pathlib import Path
from typing import Any

import nbformat

NOTEBOOK = Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb")


def _load_notebook_cases() -> tuple[Any, ...]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    case_cell = next(cell for cell in notebook.cells if cell.id == "define-cases")
    namespace: dict[str, Any] = {}
    exec(compile(case_cell.source, f"{NOTEBOOK}:define-cases", "exec"), namespace)
    return namespace["CASES"]


CASES = _load_notebook_cases()
