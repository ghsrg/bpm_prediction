from __future__ import annotations

import ast
from pathlib import Path

import pytest


# MVP1 critical runtime files that must stay isolated from MVP2 components.
MVP1_RUNTIME_FILES = [
    Path("src/cli.py"),
    Path("src/application/use_cases/trainer.py"),
    Path("src/domain/services/baseline_graph_builder.py"),
]

FORBIDDEN_TOKENS = ("eopkg", "topologyextractor", "fusiongnn", "agentcritic", "reliability")


@pytest.mark.mvp1_regression
@pytest.mark.parametrize("file_path", MVP1_RUNTIME_FILES)
def test_mvp1_runtime_has_no_direct_mvp2_imports(file_path: Path):
    assert file_path.exists(), f"Missing runtime file: {file_path}"
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    lowered = " ".join(imported_modules).lower()
    for token in FORBIDDEN_TOKENS:
        assert token not in lowered, f"Forbidden MVP2 token '{token}' imported in {file_path}"

