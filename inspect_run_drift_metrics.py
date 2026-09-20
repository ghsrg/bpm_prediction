from pathlib import Path

rd = Path("mlruns/854778689611649472/f71502fd16624ce2b838f50f91dd160a/metrics")
for m in sorted(rd.glob("drift_*")):
    lines = m.read_text(encoding="utf-8").strip().splitlines()
    print(f"{m.name}: step0={lines[0].split()[1] if lines else 'empty'} count={len(lines)}")

# Also check if any audit metrics were logged!
audit_metrics = list(rd.glob("drift_admissible_*")) + list(rd.glob("drift_strict_correct_*")) + list(rd.glob("drift_out_of_mask_*"))
print("Audit metrics found:", [a.name for a in audit_metrics])
