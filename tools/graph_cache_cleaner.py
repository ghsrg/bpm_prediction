"""Cleaner for disk-backed graph dataset cache produced by src.cli."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    dataset_name: str
    fingerprint: str
    last_access_utc: datetime
    created_utc: datetime
    size_bytes: int


def _parse_iso_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += int(item.stat().st_size)
            except OSError:
                continue
    return total


def _load_entry(entry_dir: Path) -> CacheEntry | None:
    meta_path = entry_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    last_access = _parse_iso_utc(payload.get("last_access_utc"))
    created = _parse_iso_utc(payload.get("created_utc"))
    if last_access is None and created is None:
        try:
            fallback = datetime.fromtimestamp(meta_path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            fallback = datetime.now(timezone.utc)
        last_access = fallback
        created = fallback
    elif last_access is None and created is not None:
        last_access = created
    elif created is None and last_access is not None:
        created = last_access
    size_bytes = _dir_size_bytes(entry_dir)
    return CacheEntry(
        path=entry_dir,
        dataset_name=str(payload.get("dataset_name", entry_dir.parent.name)),
        fingerprint=str(payload.get("fingerprint", entry_dir.name)),
        last_access_utc=last_access or datetime.now(timezone.utc),
        created_utc=created or datetime.now(timezone.utc),
        size_bytes=int(size_bytes),
    )


def _iter_entries(cache_dir: Path) -> List[CacheEntry]:
    entries: List[CacheEntry] = []
    if not cache_dir.exists():
        return entries
    for dataset_dir in sorted(cache_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for entry_dir in sorted(dataset_dir.iterdir()):
            if not entry_dir.is_dir():
                continue
            entry = _load_entry(entry_dir)
            if entry is not None:
                entries.append(entry)
    return entries


def _select_deletions(
    entries: Sequence[CacheEntry],
    *,
    keep_last: int,
    older_than_days: int | None,
    max_size_bytes: int | None,
) -> List[CacheEntry]:
    if not entries:
        return []
    sorted_entries = sorted(entries, key=lambda item: item.last_access_utc, reverse=True)
    keep_last = max(0, int(keep_last))
    protected = {item.path for item in sorted_entries[:keep_last]}

    deletions: Dict[Path, CacheEntry] = {}
    if older_than_days is not None and older_than_days >= 0:
        threshold = datetime.now(timezone.utc) - timedelta(days=int(older_than_days))
        for item in sorted_entries:
            if item.path in protected:
                continue
            if item.last_access_utc < threshold:
                deletions[item.path] = item

    if max_size_bytes is not None and max_size_bytes >= 0:
        total_size = sum(int(item.size_bytes) for item in sorted_entries)
        if total_size > max_size_bytes:
            # Remove oldest entries first (except protected) until cache fits target size.
            for item in sorted(sorted_entries, key=lambda row: row.last_access_utc):
                if total_size <= max_size_bytes:
                    break
                if item.path in protected:
                    continue
                if item.path not in deletions:
                    deletions[item.path] = item
                total_size -= int(item.size_bytes)

    return list(deletions.values())


def _format_bytes(num_bytes: int) -> str:
    value = float(max(0, num_bytes))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean graph dataset disk cache by age and/or size.")
    parser.add_argument("--cache-dir", default=".cache/graph_datasets", help="Graph dataset cache root directory.")
    parser.add_argument("--older-than-days", type=int, default=None, help="Delete entries older than N days.")
    parser.add_argument("--max-size-gb", type=float, default=None, help="Trim cache to this max size (GB).")
    parser.add_argument("--keep-last", type=int, default=0, help="Keep N most recently accessed entries.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned deletions without deleting files.")
    parser.add_argument("--summary-out", default="", help="Optional path for JSON summary.")
    args = parser.parse_args(argv)

    cache_dir = Path(str(args.cache_dir)).expanduser().resolve()
    entries = _iter_entries(cache_dir)
    total_before = sum(int(item.size_bytes) for item in entries)
    max_size_bytes = int(float(args.max_size_gb) * (1024 ** 3)) if args.max_size_gb is not None else None
    deletions = _select_deletions(
        entries,
        keep_last=int(args.keep_last),
        older_than_days=args.older_than_days,
        max_size_bytes=max_size_bytes,
    )
    deleted_size = 0
    deleted_items: List[Dict[str, Any]] = []
    for item in deletions:
        deleted_size += int(item.size_bytes)
        deleted_items.append(
            {
                "path": str(item.path),
                "dataset_name": item.dataset_name,
                "fingerprint": item.fingerprint,
                "last_access_utc": item.last_access_utc.isoformat(),
                "size_bytes": int(item.size_bytes),
            }
        )
        if not args.dry_run:
            try:
                shutil.rmtree(item.path, ignore_errors=False)
            except OSError:
                # Continue cleaning even if one entry fails.
                continue

    total_after = max(0, total_before - deleted_size)
    summary = {
        "status": "ok",
        "mode": "graph-cache-clean",
        "cache_dir": str(cache_dir),
        "entries_total": int(len(entries)),
        "entries_deleted": int(len(deletions)),
        "size_before_bytes": int(total_before),
        "size_deleted_bytes": int(deleted_size),
        "size_after_bytes": int(total_after),
        "size_before_human": _format_bytes(total_before),
        "size_deleted_human": _format_bytes(deleted_size),
        "size_after_human": _format_bytes(total_after),
        "dry_run": bool(args.dry_run),
        "criteria": {
            "older_than_days": args.older_than_days,
            "max_size_gb": args.max_size_gb,
            "keep_last": int(args.keep_last),
        },
        "deleted": deleted_items,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    summary_out = str(args.summary_out).strip()
    if summary_out:
        target = Path(summary_out).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
