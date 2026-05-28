import json

presets_file = "outputs/ui/experiment_ui_presets.json"
preset_name = "_EOPKGTC-UN"

with open(presets_file, "r") as f:
    presets = json.load(f)

if preset_name in presets:
    print(f"Preset '{preset_name}' found!")
    payload = presets[preset_name].get("payload", {})
    
    groups_to_print = ["vars", "general_experiment_form", "general_training_form", "input_data_form"]
    for group_name in groups_to_print:
        variables = payload.get(group_name, {})
        print(f"\n[{group_name}]")
        for k, v in sorted(variables.items()):
            print(f"  {k}: {v}")
else:
    print(f"Preset '{preset_name}' NOT found in {presets_file}!")
    # Print available presets starting with _EOPKG
    matching = [k for k in presets.keys() if k.startswith("_EOPKG")]
    print(f"Available _EOPKG* presets: {matching}")
