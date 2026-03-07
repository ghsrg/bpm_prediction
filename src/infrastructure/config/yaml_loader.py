"""YAML config loader with recursive include support for modular playbooks."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 5 (план перед кодом, anti-blind coding) і розділ 6 (посилання на контракти)
# - ARCHITECTURE_RULES.md -> розділ 2-3 (інфраструктурні утиліти поза Domain)
# - DATA_MODEL_MVP1.MD -> розділ 4.0 (єдиний експеримент-конфіг для запуску)

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings where override wins and lists are replaced."""
    # Створюємо незалежну копію бази, щоб не мутувати вхідні структури і уникати побічних ефектів.
    merged: dict[str, Any] = deepcopy(base)

    # Ітеруємося по override: кожне поле або рекурсивно зливається, або повністю перезаписується.
    for key, value in override.items():
        current = merged.get(key)

        # Словники зливаються рекурсивно, щоб зберігати вкладені дефолти.
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
            continue

        # Списки (та будь-які інші типи) повністю замінюються значенням override.
        merged[key] = deepcopy(value)

    return merged


def _read_yaml_mapping(file_path: Path) -> dict[str, Any]:
    """Read YAML file and ensure top-level mapping contract."""
    with file_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping at top level: {file_path}")

    return loaded


def _normalize_include_list(raw_include: Any, file_path: Path) -> Iterable[str]:
    """Validate include node and normalize to iterable string paths."""
    if raw_include is None:
        return []

    if not isinstance(raw_include, list) or any(not isinstance(item, str) for item in raw_include):
        raise ValueError(f"'include' must be a list of string paths in file: {file_path}")

    return raw_include


def load_yaml_with_includes(file_path: str | Path, _visited: set[Path] | None = None) -> Dict[str, Any]:
    """Load YAML with recursive includes resolved relative to current file directory."""
    # Резолвимо шлях до абсолютного, щоб коректно працювати незалежно від cwd запуску.
    resolved_path = Path(file_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    # Ведемо множину відвіданих шляхів для захисту від циклічних include.
    visited = _visited if _visited is not None else set()
    if resolved_path in visited:
        chain = " -> ".join(str(path) for path in [*visited, resolved_path])
        raise ValueError(f"Cyclic include detected: {chain}")

    visited.add(resolved_path)
    raw_config = _read_yaml_mapping(resolved_path)

    # Витягуємо include на корені; всі інші ключі поточного файла вважаємо override-рівнем.
    include_paths = _normalize_include_list(raw_config.get("include"), resolved_path)

    merged_config: dict[str, Any] = {}

    # Послідовно зливаємо базові конфіги у порядку include, щоб пізніші могли перекривати попередні.
    for include_path in include_paths:
        nested_path = (resolved_path.parent / include_path).resolve()
        nested_config = load_yaml_with_includes(nested_path, visited)
        merged_config = deep_merge(merged_config, nested_config)

    # Видаляємо include з поточного файла і накладаємо його як верхній рівень override.
    current_override = deepcopy(raw_config)
    current_override.pop("include", None)

    result = deep_merge(merged_config, current_override)

    # Видаляємо файл із активної гілки обходу, щоб дозволити нециклічне повторне використання в інших гілках.
    visited.remove(resolved_path)
    return result
