import os

paths = [
    "mlruns/0/2c866c7723594324ae59d970b3288774/params/training.candidate_batch_topology_policy",
    "mlruns/0/2c866c7723594324ae59d970b3288774/params/training.candidate_contract_mode"
]

for p in paths:
    # search other experiment dirs if needed
    if not os.path.exists(p):
        for exp in os.listdir("mlruns"):
            test_path = os.path.join("mlruns", exp, os.path.relpath(p, "mlruns/0"))
            if os.path.exists(test_path):
                p = test_path
                break
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            print(f"{p}: '{f.read().strip()}'")
    else:
        print(f"File not found: {p}")
