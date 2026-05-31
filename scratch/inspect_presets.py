import json

with open("outputs/ui/experiment_ui_presets.json", "r", encoding="utf-8") as f:
    data = json.load(f)

preset = data.get("_EOPKGTC-UN")
payload = preset['payload']
for key in ["general_tracking_form"]:
    if key in payload:
        print(f"=== {key} ===")
        print(json.dumps(payload[key], indent=2))
