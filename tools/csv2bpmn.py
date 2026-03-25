"""Extract BPMN XML files from CSV payload column (base64/hex/raw).

python tools/csv2bpmn.py  --csv data/camunda_exports/bpmn_bytes_base64.csv --out data/camunda_exports/bpmn_xml
  --id-column deployment_id \
  --payload-column bpmn_xml_content \
  --format base64 \
  --validate-xml \
  --on-duplicate suffix

"""



from __future__ import annotations

import argparse
import base64
import csv
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET


ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_path(raw_path: str, *, base_dir: Path) -> Path:
    path = Path(str(raw_path).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _safe_file_stem(raw_value: str, *, fallback: str) -> str:
    text = str(raw_value).strip()
    if not text:
        text = fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return safe.strip("._") or fallback


def _normalize_base64_payload(raw: str) -> str:
    text = str(raw).strip()
    if text.lower().startswith("data:") and ";base64," in text.lower():
        text = text.split(",", 1)[1]
    text = re.sub(r"\s+", "", text)
    pad_len = len(text) % 4
    if pad_len:
        text += "=" * (4 - pad_len)
    return text


def _decode_payload(raw_payload: str, payload_format: str) -> bytes:
    fmt = str(payload_format).strip().lower()
    if fmt == "base64":
        normalized = _normalize_base64_payload(raw_payload)
        return base64.b64decode(normalized, validate=False)
    if fmt == "hex":
        text = str(raw_payload).strip().lower()
        if text.startswith("0x"):
            text = text[2:]
        text = re.sub(r"\s+", "", text)
        return bytes.fromhex(text)
    if fmt == "raw":
        return str(raw_payload).encode("utf-8")
    raise ValueError(f"Unsupported payload format: {payload_format}")


def _is_valid_xml(payload: bytes) -> bool:
    try:
        ET.fromstring(payload)
        return True
    except ET.ParseError:
        return False


def _next_unique_path(path: Path) -> Path:
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    candidate = path
    while candidate.exists():
        candidate = parent / f"{stem}_{idx}{suffix}"
        idx += 1
    return candidate


def _resolve_write_path(path: Path, on_duplicate: str) -> Path | None:
    mode = str(on_duplicate).strip().lower()
    if mode == "overwrite":
        return path
    if mode == "skip":
        return None if path.exists() else path
    if mode == "suffix":
        return _next_unique_path(path)
    raise ValueError(f"Unsupported duplicate mode: {on_duplicate}")


def _build_csv_reader(handle, *, delimiter: str) -> csv.DictReader:
    if delimiter == "auto":
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return csv.DictReader(handle, dialect=dialect)
    return csv.DictReader(handle, delimiter=delimiter)


def _iter_rows(reader: csv.DictReader, max_rows: int | None) -> Iterable[tuple[int, dict[str, str]]]:
    for idx, row in enumerate(reader, start=1):
        if max_rows is not None and idx > max_rows:
            break
        normalized = {
            str(key).lstrip("\ufeff"): ("" if value is None else str(value))
            for key, value in row.items()
        }
        yield idx, normalized


def _set_csv_field_size_limit(limit: int | None) -> int:
    target = int(limit) if limit is not None else int(sys.maxsize)
    if target <= 0:
        raise ValueError("--field-size-limit must be > 0.")
    # Windows builds can throw OverflowError for very large values; degrade gracefully.
    while target > 0:
        try:
            csv.field_size_limit(target)
            return int(target)
        except OverflowError:
            target //= 10
    raise ValueError("Failed to set csv field size limit to a positive value.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decode BPMN payload from CSV and write .bpmn files.")
    parser.add_argument("--csv", default="bpmn_bytes_base64.csv", help="Input CSV path.")
    parser.add_argument("--out", default="data/camunda_exports/bpmn_xml", help="Output directory for .bpmn files.")
    parser.add_argument("--id-column", default="deployment_id", help="CSV column used as file identifier.")
    parser.add_argument("--payload-column", default="bpmn_xml_content", help="CSV column with payload.")
    parser.add_argument("--format", default="base64", choices=["base64", "hex", "raw"], help="Payload encoding format.")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter or 'auto'.")
    parser.add_argument("--encoding", default="utf-8", help="CSV file encoding.")
    parser.add_argument("--on-duplicate", default="suffix", choices=["suffix", "overwrite", "skip"], help="Duplicate output file handling.")
    parser.add_argument("--validate-xml", action="store_true", help="Validate decoded payload as XML before writing.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any row fails.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional hard limit for processed CSV rows.")
    parser.add_argument(
        "--field-size-limit",
        type=int,
        default=None,
        help="Maximum CSV field size in bytes. Default: auto (largest supported).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only validate/plan writes without writing files.")
    args = parser.parse_args(argv)

    effective_field_limit = _set_csv_field_size_limit(args.field_size_limit)

    csv_path = _resolve_path(args.csv, base_dir=ROOT_DIR)
    out_dir = _resolve_path(args.out, base_dir=ROOT_DIR)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    failed = 0

    with csv_path.open("r", encoding=str(args.encoding), newline="") as handle:
        reader = _build_csv_reader(handle, delimiter=str(args.delimiter))
        raw_field_names = [str(name) for name in (reader.fieldnames or [])]
        field_names = [name.lstrip("\ufeff") for name in raw_field_names]
        if reader.fieldnames:
            reader.fieldnames = field_names
        if args.id_column not in field_names:
            raise ValueError(f"Missing id column '{args.id_column}'. Available: {field_names}")
        if args.payload_column not in field_names:
            raise ValueError(f"Missing payload column '{args.payload_column}'. Available: {field_names}")

        for row_idx, row in _iter_rows(reader, max_rows=args.max_rows):
            raw_id = row.get(args.id_column, "")
            raw_payload = row.get(args.payload_column, "")
            file_stem = _safe_file_stem(raw_id, fallback=f"row_{row_idx:06d}")
            target = out_dir / f"{file_stem}.bpmn"
            target_path = _resolve_write_path(target, args.on_duplicate)
            if target_path is None:
                skipped += 1
                print(f"[skip] row={row_idx} reason=duplicate policy=skip file={target.name}")
                continue
            try:
                binary_payload = _decode_payload(raw_payload, args.format)
                if len(binary_payload) == 0:
                    raise ValueError("decoded payload is empty")
                if args.validate_xml and not _is_valid_xml(binary_payload):
                    raise ValueError("decoded payload is not valid XML")
                if not args.dry_run:
                    target_path.write_bytes(binary_payload)
                written += 1
                print(f"[ok] row={row_idx} file={target_path}")
            except Exception as exc:
                failed += 1
                print(f"[error] row={row_idx} id={raw_id!r} reason={exc}")
                if args.strict:
                    break

    print("==== summary ====")
    print(f"csv={csv_path}")
    print(f"out={out_dir}")
    print(f"field_size_limit={effective_field_limit}")
    print(f"written={written} skipped={skipped} failed={failed}")
    if failed > 0 and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
