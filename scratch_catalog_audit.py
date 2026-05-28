import csv
import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CATALOG_PATH = ROOT_DIR / "configs" / "ui" / "config_catalog.yaml"
MATRIX_PATH = ROOT_DIR / "outputs" / "ui" / "desktop_ui_field_dependency_matrix.csv"

def audit():
    print(f"Loading catalog: {CATALOG_PATH}")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    
    catalog_fields = catalog.get("fields", {})
    print(f"Catalog contains {len(catalog_fields)} fields.")

    print(f"Loading matrix: {MATRIX_PATH}")
    matrix_fields = {}
    if MATRIX_PATH.exists():
        with open(MATRIX_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                matrix_fields[row["path"]] = row
        print(f"Matrix contains {len(matrix_fields)} fields.")
    else:
        print("Matrix does not exist.")

    # 1. Check mismatch
    catalog_keys = set(catalog_fields.keys())
    matrix_keys = set(matrix_fields.keys())

    missing_in_matrix = catalog_keys - matrix_keys
    missing_in_catalog = matrix_keys - catalog_keys

    print(f"\nFields in Catalog but missing in Matrix ({len(missing_in_matrix)}):")
    for k in sorted(missing_in_matrix):
        print(f"  - {k}")

    print(f"\nFields in Matrix but missing in Catalog ({len(missing_in_catalog)}):")
    for k in sorted(missing_in_catalog):
        print(f"  - {k}")

    # 2. Check empty or default descriptions in catalog
    empty_desc = []
    default_desc = []
    for k, v in catalog_fields.items():
        desc = v.get("description", "").strip()
        if not desc:
            empty_desc.append(k)
        elif "Службовий параметр" in desc or "Model architecture" in desc or "data/structure mapping" in desc:
            default_desc.append((k, desc))

    print(f"\nFields with empty descriptions in Catalog ({len(empty_desc)}):")
    for k in sorted(empty_desc):
        print(f"  - {k}")

    print(f"\nFields with generic/default descriptions in Catalog ({len(default_desc)}):")
    for k, d in sorted(default_desc)[:15]:
        print(f"  - {k}: {d}")
    if len(default_desc) > 15:
        print(f"  ... and {len(default_desc) - 15} more.")

if __name__ == "__main__":
    audit()
