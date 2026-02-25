#!/usr/bin/env python3
"""Architecture Guard for Clean + Hexagonal dependency rules."""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

FORBIDDEN_CORE_IMPORT_PREFIXES = (
    "src.adapters",
    "src.pipeline",
    "mlflow",
    "neo4j",
    "sqlalchemy",
    "fastapi",
)


@dataclass
class Violation:
    file: Path
    line: int
    rule: str
    detail: str

    def render(self) -> str:
        rel = self.file.relative_to(ROOT)
        return f"[ARCH_GUARD] {rel}:{self.line} | {self.rule} | {self.detail}"


def is_py(path: Path) -> bool:
    return path.suffix == ".py" and path.is_file()


def collect_py_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def import_targets(node: ast.AST) -> Iterable[tuple[int, str, str]]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield node.lineno, alias.name, "import"
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        names = ",".join(a.name for a in node.names)
        yield node.lineno, module, f"from {module} import {names}"


def parse_file(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse {path}: {exc}") from exc


def starts_with_any(value: str, prefixes: tuple[str, ...]) -> bool:
    return any(value == p or value.startswith(p + ".") for p in prefixes)


def check_core_layer(core_files: list[Path]) -> list[Violation]:
    violations: list[Violation] = []
    for file in core_files:
        tree = parse_file(file)
        for node in ast.walk(tree):
            for line, module, _ in import_targets(node):
                if not module:
                    continue
                if starts_with_any(module, FORBIDDEN_CORE_IMPORT_PREFIXES):
                    violations.append(Violation(file, line, "CORE_BOUNDARY", f"illegal import '{module}'"))
                if module == "torch_geometric.data":
                    if isinstance(node, ast.ImportFrom):
                        if any(alias.name == "Data" for alias in node.names):
                            violations.append(
                                Violation(file, line, "CORE_BOUNDARY", "illegal import 'torch_geometric.data.Data'")
                            )
    return violations


def check_pipeline_layer(pipeline_files: list[Path]) -> list[Violation]:
    violations: list[Violation] = []
    for file in pipeline_files:
        tree = parse_file(file)
        for node in ast.walk(tree):
            for line, module, _ in import_targets(node):
                if not module:
                    continue
                if module.startswith("src.adapters"):
                    violations.append(Violation(file, line, "PIPELINE_DEP", f"pipeline must not import adapter '{module}'"))
                if module.startswith("src.core") and not module.startswith("src.core.interfaces"):
                    violations.append(
                        Violation(
                            file,
                            line,
                            "PIPELINE_PORTS_ONLY",
                            f"pipeline must depend only on ports, got '{module}'",
                        )
                    )
    return violations


def collect_interface_classes(interface_files: list[Path]) -> set[str]:
    names: set[str] = set()
    for file in interface_files:
        tree = parse_file(file)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                names.add(node.name)
    return names


def base_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def check_adapters_layer(adapter_files: list[Path], interface_classes: set[str]) -> list[Violation]:
    violations: list[Violation] = []
    for file in adapter_files:
        tree = parse_file(file)
        for node in ast.walk(tree):
            for line, module, _ in import_targets(node):
                if not module:
                    continue
                if module.startswith("src.pipeline"):
                    violations.append(Violation(file, line, "ADAPTER_DEP", f"adapter must not import pipeline '{module}'"))
                if module.startswith("src.core") and not module.startswith("src.core.interfaces"):
                    violations.append(
                        Violation(file, line, "ADAPTER_PORTS_ONLY", f"adapter may import only ports, got '{module}'")
                    )

        class_defs = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        if class_defs:
            has_port_impl = False
            for cls in class_defs:
                bases = {b for b in (base_name(expr) for expr in cls.bases) if b}
                if bases & interface_classes:
                    has_port_impl = True
                    break
            if not has_port_impl and interface_classes:
                violations.append(
                    Violation(
                        file,
                        class_defs[0].lineno,
                        "ADAPTER_INTERFACE_IMPL",
                        "adapter classes must inherit at least one interface from src/core/interfaces",
                    )
                )
    return violations


def check_circular(core_files: list[Path], adapter_files: list[Path], pipeline_files: list[Path]) -> list[Violation]:
    violations: list[Violation] = []

    def has_import(files: list[Path], prefix: str) -> bool:
        for f in files:
            tree = parse_file(f)
            for n in ast.walk(tree):
                for _, module, _ in import_targets(n):
                    if module.startswith(prefix):
                        return True
        return False

    core_to_adapters = has_import(core_files, "src.adapters")
    adapters_to_core = has_import(adapter_files, "src.core")
    pipeline_to_adapters = has_import(pipeline_files, "src.adapters")
    adapters_to_pipeline = has_import(adapter_files, "src.pipeline")

    if core_to_adapters and adapters_to_core:
        violations.append(Violation(ROOT / "src/core", 1, "CIRCULAR_DEP", "circular dependency core ↔ adapters"))
    if pipeline_to_adapters and adapters_to_pipeline:
        violations.append(Violation(ROOT / "src/pipeline", 1, "CIRCULAR_DEP", "circular dependency pipeline ↔ adapters"))

    return violations


def main() -> int:
    if not SRC.exists():
        print("[ARCH_GUARD] src/ not found; no architectural checks executed.")
        return 0

    core_files = collect_py_files(SRC / "core")
    pipeline_files = collect_py_files(SRC / "pipeline")
    adapter_files = collect_py_files(SRC / "adapters")
    interface_files = collect_py_files(SRC / "core" / "interfaces")
    interface_classes = collect_interface_classes(interface_files)

    violations: list[Violation] = []
    violations += check_core_layer(core_files)
    violations += check_pipeline_layer(pipeline_files)
    violations += check_adapters_layer(adapter_files, interface_classes)
    violations += check_circular(core_files, adapter_files, pipeline_files)

    if violations:
        for v in violations:
            print(v.render())
        print(f"[ARCH_GUARD] FAILED with {len(violations)} violation(s).")
        return 1

    print("[ARCH_GUARD] OK: architecture dependency rules satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
