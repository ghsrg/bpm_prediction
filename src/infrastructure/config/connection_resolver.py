"""Centralized resolver for DB connection profiles and environment secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

from src.infrastructure.config.yaml_loader import load_yaml_with_includes


_ENV_FILES_LOADED = False


def load_local_env_files(
    *,
    env: MutableMapping[str, str] | None = None,
    candidates: tuple[str, ...] = (".env", ".env.local"),
) -> None:
    """Load optional local env files into process environment.

    Existing environment values are not overwritten.
    """
    global _ENV_FILES_LOADED
    if _ENV_FILES_LOADED:
        return

    target = env if env is not None else os.environ
    for candidate in candidates:
        path = Path(candidate)
        if not path.exists() or not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            env_key = key.strip()
            if not env_key:
                continue
            if env_key in target and str(target.get(env_key, "")).strip():
                continue
            target[env_key] = _strip_env_quotes(raw_value.strip())
    _ENV_FILES_LOADED = True


def resolve_mssql_connection_string(
    *,
    cfg: Mapping[str, Any],
    default_connections_file: str = "configs/connections/mssql_camunda.yaml",
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve MSSQL ODBC connection string from profile + env secrets."""
    env_map = env if env is not None else os.environ
    load_local_env_files()

    mssql_cfg = cfg.get("mssql", {})
    if not isinstance(mssql_cfg, dict):
        mssql_cfg = {}

    # 1) full connection string from env variable.
    conn_env_key = str(
        mssql_cfg.get("connection_string_env")
        or cfg.get("connection_string_env")
        or ""
    ).strip()
    if conn_env_key:
        conn_env_value = str(env_map.get(conn_env_key, "")).strip()
        if conn_env_value:
            return conn_env_value

    # 2) legacy direct connection string from config (backward compatibility).
    legacy_conn = str(mssql_cfg.get("connection_string", "")).strip()
    if legacy_conn:
        return legacy_conn

    # 3) profile-based composition.
    profile_name = str(
        mssql_cfg.get("connection_profile")
        or mssql_cfg.get("profile")
        or cfg.get("connection_profile")
        or cfg.get("profile")
        or "local"
    ).strip() or "local"

    connections_file = str(
        mssql_cfg.get("connections_file")
        or cfg.get("connections_file")
        or default_connections_file
    ).strip() or default_connections_file
    profile = _load_profile(
        connections_file=connections_file,
        section_key="mssql_camunda",
        profile_name=profile_name,
    )
    if not profile:
        return ""

    profile_conn_env = str(profile.get("connection_string_env", "")).strip()
    if profile_conn_env:
        conn_from_profile_env = str(env_map.get(profile_conn_env, "")).strip()
        if conn_from_profile_env:
            return conn_from_profile_env

    user = _resolve_secret_value(profile=profile, explicit_key="user", env_key="user_env", env=env_map)
    password = _resolve_secret_value(profile=profile, explicit_key="password", env_key="password_env", env=env_map)
    return _build_odbc_connection_string(profile=profile, user=user, password=password)


def resolve_neo4j_connection_settings(
    *,
    cfg: Mapping[str, Any],
    default_connections_file: str = "configs/connections/neo4j.yaml",
    env: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    """Resolve Neo4j connection settings from profile + env secrets."""
    env_map = env if env is not None else os.environ
    load_local_env_files()

    neo4j_cfg = cfg.get("neo4j", {})
    if not isinstance(neo4j_cfg, dict):
        neo4j_cfg = {}

    profile_name = str(
        neo4j_cfg.get("connection_profile")
        or neo4j_cfg.get("profile")
        or cfg.get("connection_profile")
        or cfg.get("profile")
        or "local"
    ).strip() or "local"
    connections_file = str(
        neo4j_cfg.get("connections_file")
        or cfg.get("connections_file")
        or default_connections_file
    ).strip() or default_connections_file

    profile = _load_profile(
        connections_file=connections_file,
        section_key="neo4j",
        profile_name=profile_name,
    )
    if not profile:
        return {}

    uri = str(profile.get("uri", "")).strip()
    database = str(profile.get("database", "neo4j")).strip() or "neo4j"
    user = _resolve_secret_value(profile=profile, explicit_key="user", env_key="user_env", env=env_map)
    password = _resolve_secret_value(profile=profile, explicit_key="password", env_key="password_env", env=env_map)
    if not uri:
        return {}
    return {
        "uri": uri,
        "database": database,
        "user": user,
        "password": password,
    }


def _load_profile(
    *,
    connections_file: str,
    section_key: str,
    profile_name: str,
) -> Dict[str, Any]:
    path = Path(connections_file)
    if not path.exists():
        return {}
    raw = load_yaml_with_includes(path)
    candidates: list[dict[str, Any]] = []
    if isinstance(raw.get("profiles"), dict):
        candidates.append(raw)
    section = raw.get(section_key)
    if isinstance(section, dict):
        candidates.append(section)
    for candidate in candidates:
        profiles = candidate.get("profiles", {})
        if not isinstance(profiles, dict):
            continue
        profile = profiles.get(profile_name)
        if isinstance(profile, dict):
            return dict(profile)
    if candidates:
        profiles = candidates[0].get("profiles", {})
        if isinstance(profiles, dict) and len(profiles) == 1:
            only_profile = next(iter(profiles.values()))
            if isinstance(only_profile, dict):
                return dict(only_profile)
    return {}


def _resolve_secret_value(
    *,
    profile: Mapping[str, Any],
    explicit_key: str,
    env_key: str,
    env: Mapping[str, str],
) -> str:
    direct = str(profile.get(explicit_key, "")).strip()
    if direct:
        return direct
    env_name = str(profile.get(env_key, "")).strip()
    if not env_name:
        return ""
    return str(env.get(env_name, "")).strip()


def _build_odbc_connection_string(
    *,
    profile: Mapping[str, Any],
    user: str,
    password: str,
) -> str:
    driver = str(profile.get("driver", "ODBC Driver 18 for SQL Server")).strip()
    host = str(profile.get("host", "localhost")).strip() or "localhost"
    port = str(profile.get("port", "")).strip()
    server = f"{host},{port}" if port else host
    database = str(profile.get("database", "")).strip()

    parts = [f"DRIVER={{{driver}}}", f"SERVER={server}"]
    if database:
        parts.append(f"DATABASE={database}")

    trusted_connection = bool(profile.get("trusted_connection", False))
    if user and password:
        parts.append(f"UID={user}")
        parts.append(f"PWD={password}")
    elif trusted_connection:
        parts.append("Trusted_Connection=yes")

    if "encrypt" in profile:
        parts.append(f"Encrypt={'yes' if bool(profile.get('encrypt')) else 'no'}")
    if "trust_server_certificate" in profile:
        parts.append(
            f"TrustServerCertificate={'yes' if bool(profile.get('trust_server_certificate')) else 'no'}"
        )

    extras = profile.get("extras", {})
    if isinstance(extras, dict):
        for key, value in extras.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            parts.append(f"{key_text}={value}")

    return ";".join(parts) + ";"


def _strip_env_quotes(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return raw[1:-1]
    return raw
