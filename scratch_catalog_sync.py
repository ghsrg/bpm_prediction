import csv
import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CATALOG_PATH = ROOT_DIR / "configs" / "ui" / "config_catalog.yaml"
MATRIX_PATH = ROOT_DIR / "outputs" / "ui" / "desktop_ui_field_dependency_matrix.csv"

def sync_and_audit():
    print("Loading catalog...")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    
    fields = catalog.get("fields", {})

    # 1. Update stats_time_policy metadata in Catalog
    if "experiment.stats_time_policy" in fields:
        fields["experiment.stats_time_policy"].update({
            "label": "Політика вибору статистики зі снепшотів",
            "description": "Політика вибору статистики зі снепшотів: latest або strict_asof.",
            "affects": "Визначає, які дані потрапляють в обробку: поточний час або останній доступний. Для швидкого/початкового експерименту підходить latest, для дисертаційного наукового аналізу — лише strict_asof.",
            "required_when": {"experiment.statistic_enabled": "true"},
            "active_when": {"experiment.statistic_enabled": "true"}
        })
        print("Updated experiment.stats_time_policy in catalog.")

    # 2. Update load_checkpoint metadata in Catalog
    if "experiment.load_checkpoint" in fields:
        fields["experiment.load_checkpoint"].update({
            "label": "Завантаження checkpoint",
            "description": "Шлях до checkpoint (.pth), який використовується для оцінки (eval_*) або дрейфу (eval_drift).",
            "affects": "Забезпечує тестування правильних навчених ваг. UI інтегрує розумний пошук: фільтрує за поточним experiment.name та сортує за датою оновлення (newest first).",
            "required_when": {"experiment.mode": ["eval_drift", "eval_cross_dataset"]},
            "active_when": {"experiment.mode": ["eval_drift", "eval_cross_dataset", "train"]}
        })
        print("Updated experiment.load_checkpoint in catalog.")

    # Write back the updated catalog
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(catalog, f, default_flow_style=False, allow_unicode=True, sort_keys=True)
    print("Saved catalog.")

    # 3. Audit and update CSV matrix
    matrix_rows = {}
    if MATRIX_PATH.exists():
        with open(MATRIX_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                matrix_rows[row["path"]] = row
        print(f"Loaded {len(matrix_rows)} rows from matrix.")

    # Update CSV fields or add missing ones
    for path, meta in fields.items():
        # Determine tab and group fallbacks
        proposed_level = meta.get("ui_level", "advanced")
        ui = meta.get("ui") or {}
        proposed_tab = meta.get("ui_tab", ui.get("tab", "Advanced"))
        proposed_group = meta.get("ui_group", ui.get("group", "General"))
        current_ui_order = int(ui.get("order", 1000))

        # Check required_in_modes
        required_in_modes_str = "|".join(meta.get("required_in_modes", []))
        required_when_str = "; ".join(f"{k}={v}" for k, v in meta.get("required_when", {}).items())
        active_when_str = "; ".join(f"{k}={v}" for k, v in meta.get("active_when", {}).items())
        enum_str = "|".join(str(x) for x in meta.get("enum", []) if x is not None)
        
        row_data = {
            "path": path,
            "section": meta.get("section", path.split(".", 1)[0]),
            "current_ui_tab": proposed_tab.lower(),
            "current_ui_group": proposed_group.lower(),
            "current_ui_order": str(current_ui_order),
            "proposed_level": proposed_level,
            "proposed_tab": proposed_tab,
            "proposed_group": proposed_group,
            "active_when": active_when_str or "always",
            "required_in_modes": required_in_modes_str,
            "required_when": required_when_str,
            "enum": enum_str,
            "default": str(meta.get("default", "")),
            "runtime_consumers": "; ".join(meta.get("runtime_consumers", [])),
            "description": meta.get("description", "")
        }

        # Keep or override existing rows
        if path in matrix_rows:
            # Let's override description, affects, active_when, required_when, required_in_modes
            matrix_rows[path].update({
                "description": meta.get("description", matrix_rows[path].get("description", "")),
                "active_when": row_data["active_when"],
                "required_in_modes": row_data["required_in_modes"],
                "required_when": row_data["required_when"],
                "enum": row_data["enum"],
                "default": row_data["default"],
            })
            # Specially for the two updated fields
            if path in ["experiment.stats_time_policy", "experiment.load_checkpoint"]:
                matrix_rows[path].update({
                    "proposed_level": row_data["proposed_level"],
                    "proposed_tab": row_data["proposed_tab"],
                    "proposed_group": row_data["proposed_group"],
                })
        else:
            matrix_rows[path] = row_data

    # Write updated matrix CSV
    fieldnames = [
        "path", "section", "current_ui_tab", "current_ui_group", "current_ui_order",
        "proposed_level", "proposed_tab", "proposed_group", "active_when",
        "required_in_modes", "required_when", "enum", "default",
        "runtime_consumers", "description"
    ]
    with open(MATRIX_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Sort matrix rows by path
        for path in sorted(matrix_rows.keys()):
            # Filter row keys to match fieldnames
            row = {k: matrix_rows[path].get(k, "") for k in fieldnames}
            writer.writerow(row)
            
    print(f"Saved updated matrix CSV with {len(matrix_rows)} entries.")

if __name__ == "__main__":
    sync_and_audit()
