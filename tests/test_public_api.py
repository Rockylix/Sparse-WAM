"""Public names and exported configurations stay aligned with the CLI."""

import json
from pathlib import Path

import pytest

from sparse_wam import SparseWAMConfig
from sparse_wam.cli import main
from sparse_wam.runtime import SparseWAMRunner


@pytest.mark.parametrize("backbone", ["edge", "nano"])
def test_public_configuration_matches_export(backbone):
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "configs" / f"{backbone}.json").read_text())
    assert SparseWAMConfig(backbone).to_dict() == expected
    assert SparseWAMRunner.__name__ == "SparseWAMRunner"


@pytest.mark.parametrize("backbone", ["edge", "nano"])
def test_config_cli_uses_public_configuration(backbone, capsys):
    main(["config", "--backbone", backbone, "--execution", "compile-graph"])
    assert json.loads(capsys.readouterr().out) == SparseWAMConfig(backbone, "compile-graph").to_dict()
