from pathlib import Path
import xml.etree.ElementTree as ET

# Count traces in XES
xes_path = Path("outputs/simulation/loan_v1_v5_complex_simulated.xes")
print("XES exists:", xes_path.exists(), "size:", xes_path.stat().st_size)

# Count <trace> tags
count = 0
with open(xes_path, "r", encoding="utf-8") as f:
    for line in f:
        if "<trace>" in line or "<trace " in line:
            count += 1
print("Total traces in XES:", count)
