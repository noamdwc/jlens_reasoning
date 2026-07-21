import tomllib
from pathlib import Path

from setuptools.config.expand import find_packages

ROOT = Path(__file__).resolve().parents[1]


def test_setuptools_discovers_library_and_experiment_packages() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    find_config = config["tool"]["setuptools"]["packages"]["find"]

    assert find_config == {
        "where": ["src", "."],
        "include": ["jlens_reasoning*", "experiments*"],
        "namespaces": False,
    }
    discovered = set(
        find_packages(
            where=find_config["where"],
            include=find_config["include"],
            namespaces=find_config["namespaces"],
            root_dir=ROOT,
        )
    )
    assert "jlens_reasoning" in discovered
    assert "jlens_reasoning.experiments_utils" in discovered
    assert "experiments" in discovered
    assert "experiments.jlens_readout_sanity" in discovered
    assert not any(name.startswith("tests") for name in discovered)
