from pathlib import Path

def get_metrics_and_tags(run_id):
    for exp_dir in Path("mlruns").iterdir():
        if not exp_dir.is_dir():
            continue
        rd = exp_dir / run_id
        if rd.is_dir():
            metrics = {}
            for m in (rd / "metrics").glob("*"):
                lines = m.read_text(encoding="utf-8").strip().splitlines()
                if lines:
                    metrics[m.name] = (len(lines), lines[0].split()[1], lines[-1].split()[1])
            params = {}
            for p in (rd / "params").glob("*"):
                params[p.name] = p.read_text(encoding="utf-8").strip()
            return metrics, params
    return None, None

m_old, p_old = get_metrics_and_tags("302fb003f78d468b867a9477e40e9d50")
m_new, p_new = get_metrics_and_tags("f71502fd16624ce2b838f50f91dd160a")

print("=== OLD RUN ===")
for k in ["data_num_traces", "data_num_events", "drift_window_macro_f1", "drift_window_target_in_mask_rate"]:
    print(k, m_old.get(k))

print("\n=== NEW RUN ===")
for k in ["data_num_traces", "data_num_events", "drift_window_macro_f1", "drift_window_target_in_mask_rate"]:
    print(k, m_new.get(k))
