#!/usr/bin/env python3
"""Architecture Guard for current Clean/Hexagonal layer boundaries."""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOMAIN_DIR = SRC / "domain"
APPLICATION_DIR = SRC / "application"
ADAPTERS_DIR = SRC / "adapters"


@dataclass
class Violation:
    file: Path
    line: int
    rule: str
    detail: str

    def render(self) -> str:
        rel = self.file.relative_to(ROOT)
        return f"[ARCH_GUARD] {rel}:{self.line} | {self.rule} | {self.detail}"


def collect_py_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def parse_file(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse {path}: {exc}") from exc


def import_targets(node: ast.AST) -> Iterable[tuple[int, str]]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield node.lineno, alias.name
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            yield node.lineno, node.module


def starts_with_any(value: str, prefixes: tuple[str, ...]) -> bool:
    return any(value == prefix or value.startswith(prefix + ".") for prefix in prefixes)


def check_domain_layer(files: list[Path]) -> list[Violation]:
    violations: list[Violation] = []
    forbidden_prefixes = ("src.application", "src.adapters", "src.infrastructure")

    for file in files:
        tree = parse_file(file)
        for node in ast.walk(tree):
            for line, module in import_targets(node):
                if starts_with_any(module, forbidden_prefixes):
                    violations.append(
                        Violation(
                            file=file,
                            line=line,
                            rule="DOMAIN_BOUNDARY",
                            detail=f"domain must not import '{module}'",
                        )
                    )
    return violations


def check_application_layer(files: list[Path]) -> list[Violation]:
    violations: list[Violation] = []
    forbidden_prefixes = ("src.adapters",)

    for file in files:
        tree = parse_file(file)
        for node in ast.walk(tree):
            for line, module in import_targets(node):
                if starts_with_any(module, forbidden_prefixes):
                    violations.append(
                        Violation(
                            file=file,
                            line=line,
                            rule="APPLICATION_BOUNDARY",
                            detail=f"application must not import '{module}'",
                        )
                    )
    return violations


def check_adapters_layer(files: list[Path]) -> list[Violation]:
    violations: list[Violation] = []

    for file in files:
        tree = parse_file(file)
        for node in ast.walk(tree):
            for line, module in import_targets(node):
                if starts_with_any(module, ("src.application.use_cases",)):
                    violations.append(
                        Violation(
                            file=file,
                            line=line,
                            rule="ADAPTERS_BOUNDARY",
                            detail=f"adapters must not depend on use-cases '{module}'",
                        )
                    )
    return violations


def main() -> int:
    if not SRC.exists():
        print("[ARCH_GUARD] src/ not found; no architectural checks executed.")
        return 0

    domain_files = collect_py_files(DOMAIN_DIR)
    application_files = collect_py_files(APPLICATION_DIR)
    adapters_files = collect_py_files(ADAPTERS_DIR)

    violations: list[Violation] = []
    violations.extend(check_domain_layer(domain_files))
    violations.extend(check_application_layer(application_files))
    violations.extend(check_adapters_layer(adapters_files))

    if violations:
        for violation in violations:
            print(violation.render())
        print(f"[ARCH_GUARD] FAILED with {len(violations)} violation(s).")
        return 1

    print("[ARCH_GUARD] OK: architecture dependency rules satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
