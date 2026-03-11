"""YAML config loader with recursive include support for modular playbooks."""

# Р’С–РґРїРѕРІС–РґРЅРѕ РґРѕ:
# - AGENT_GUIDE.MD -> СЂРѕР·РґС–Р» 5 (РїР»Р°РЅ РїРµСЂРµРґ РєРѕРґРѕРј, anti-blind coding) С– СЂРѕР·РґС–Р» 6 (РїРѕСЃРёР»Р°РЅРЅСЏ РЅР° РєРѕРЅС‚СЂР°РєС‚Рё)
# - ARCHITECTURE_RULES.MD -> СЂРѕР·РґС–Р» 2-3 (С–РЅС„СЂР°СЃС‚СЂСѓРєС‚СѓСЂРЅС– СѓС‚РёР»С–С‚Рё РїРѕР·Р° Domain)
# - DATA_MODEL_MVP1.MD -> СЂРѕР·РґС–Р» 4.0 (С”РґРёРЅРёР№ РµРєСЃРїРµСЂРёРјРµРЅС‚-РєРѕРЅС„С–Рі РґР»СЏ Р·Р°РїСѓСЃРєСѓ)

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings where override wins and lists are replaced."""
    # РЎС‚РІРѕСЂСЋС”РјРѕ РЅРµР·Р°Р»РµР¶РЅСѓ РєРѕРїС–СЋ Р±Р°Р·Рё, С‰РѕР± РЅРµ РјСѓС‚СѓРІР°С‚Рё РІС…С–РґРЅС– СЃС‚СЂСѓРєС‚СѓСЂРё С– СѓРЅРёРєР°С‚Рё РїРѕР±С–С‡РЅРёС… РµС„РµРєС‚С–РІ.
    merged: dict[str, Any] = deepcopy(base)

    # Р†С‚РµСЂСѓС”РјРѕСЃСЏ РїРѕ override: РєРѕР¶РЅРµ РїРѕР»Рµ Р°Р±Рѕ СЂРµРєСѓСЂСЃРёРІРЅРѕ Р·Р»РёРІР°С”С‚СЊСЃСЏ, Р°Р±Рѕ РїРѕРІРЅС–СЃС‚СЋ РїРµСЂРµР·Р°РїРёСЃСѓС”С‚СЊСЃСЏ.
    for key, value in override.items():
        current = merged.get(key)

        # РЎР»РѕРІРЅРёРєРё Р·Р»РёРІР°СЋС‚СЊСЃСЏ СЂРµРєСѓСЂСЃРёРІРЅРѕ, С‰РѕР± Р·Р±РµСЂС–РіР°С‚Рё РІРєР»Р°РґРµРЅС– РґРµС„РѕР»С‚Рё.
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
            continue

        # РЎРїРёСЃРєРё (С‚Р° Р±СѓРґСЊ-СЏРєС– С–РЅС€С– С‚РёРїРё) РїРѕРІРЅС–СЃС‚СЋ Р·Р°РјС–РЅСЋСЋС‚СЊСЃСЏ Р·РЅР°С‡РµРЅРЅСЏРј override.
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
    # Р РµР·РѕР»РІРёРјРѕ С€Р»СЏС… РґРѕ Р°Р±СЃРѕР»СЋС‚РЅРѕРіРѕ, С‰РѕР± РєРѕСЂРµРєС‚РЅРѕ РїСЂР°С†СЋРІР°С‚Рё РЅРµР·Р°Р»РµР¶РЅРѕ РІС–Рґ cwd Р·Р°РїСѓСЃРєСѓ.
    resolved_path = Path(file_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    # Р’РµРґРµРјРѕ РјРЅРѕР¶РёРЅСѓ РІС–РґРІС–РґР°РЅРёС… С€Р»СЏС…С–РІ РґР»СЏ Р·Р°С…РёСЃС‚Сѓ РІС–Рґ С†РёРєР»С–С‡РЅРёС… include.
    visited = _visited if _visited is not None else set()
    if resolved_path in visited:
        chain = " -> ".join(str(path) for path in [*visited, resolved_path])
        raise ValueError(f"Cyclic include detected: {chain}")

    visited.add(resolved_path)
    raw_config = _read_yaml_mapping(resolved_path)

    # Р’РёС‚СЏРіСѓС”РјРѕ include РЅР° РєРѕСЂРµРЅС–; РІСЃС– С–РЅС€С– РєР»СЋС‡С– РїРѕС‚РѕС‡РЅРѕРіРѕ С„Р°Р№Р»Р° РІРІР°Р¶Р°С”РјРѕ override-СЂС–РІРЅРµРј.
    include_paths = _normalize_include_list(raw_config.get("include"), resolved_path)

    merged_config: dict[str, Any] = {}

    # РџРѕСЃР»С–РґРѕРІРЅРѕ Р·Р»РёРІР°С”РјРѕ Р±Р°Р·РѕРІС– РєРѕРЅС„С–РіРё Сѓ РїРѕСЂСЏРґРєСѓ include, С‰РѕР± РїС–Р·РЅС–С€С– РјРѕРіР»Рё РїРµСЂРµРєСЂРёРІР°С‚Рё РїРѕРїРµСЂРµРґРЅС–.
    for include_path in include_paths:
        nested_path = (resolved_path.parent / include_path).resolve()
        nested_config = load_yaml_with_includes(nested_path, visited)
        merged_config = deep_merge(merged_config, nested_config)

    # Р’РёРґР°Р»СЏС”РјРѕ include Р· РїРѕС‚РѕС‡РЅРѕРіРѕ С„Р°Р№Р»Р° С– РЅР°РєР»Р°РґР°С”РјРѕ Р№РѕРіРѕ СЏРє РІРµСЂС…РЅС–Р№ СЂС–РІРµРЅСЊ override.
    current_override = deepcopy(raw_config)
    current_override.pop("include", None)

    result = deep_merge(merged_config, current_override)

    # Р’РёРґР°Р»СЏС”РјРѕ С„Р°Р№Р» С–Р· Р°РєС‚РёРІРЅРѕС— РіС–Р»РєРё РѕР±С…РѕРґСѓ, С‰РѕР± РґРѕР·РІРѕР»РёС‚Рё РЅРµС†РёРєР»С–С‡РЅРµ РїРѕРІС‚РѕСЂРЅРµ РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РІ С–РЅС€РёС… РіС–Р»РєР°С….
    visited.remove(resolved_path)
    return result

