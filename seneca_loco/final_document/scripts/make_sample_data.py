"""Create example CSVs so `generate_all.py` runs out of the box."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import DATA_DIR

DATA_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# 1. PPO training log: 5 seeds, 2 methods, reward saturating
rows = []
for method in ("baseline", "imitation"):
    asymptote = 800 if method == "imitation" else 550
    for seed in range(5):
        for step in range(0, 200_001, 2000):
            base = asymptote * (1 - np.exp(-step / 70_000))
            noise = rng.normal(0, 35)
            rows.append((method, seed, step, base + noise))
pd.DataFrame(rows, columns=["method", "seed", "step", "reward"]).to_csv(
    DATA_DIR / "ppo_training_log.csv", index=False)

# 2. Mocap reference + sim rollout (3 joints, 2 s @ 200 Hz)
t = np.linspace(0, 2.0, 400)
ref = pd.DataFrame({
    "t":         t,
    "hip_flex":  0.5 * np.sin(2 * np.pi * 1.2 * t),
    "knee_flex": 0.7 * np.sin(2 * np.pi * 1.2 * t + 0.6),
    "ankle_flex": 0.3 * np.sin(2 * np.pi * 1.2 * t + 1.2),
})
sim = ref.copy()
for col in ("hip_flex", "knee_flex", "ankle_flex"):
    sim[col] = sim[col] * 0.92 + rng.normal(0, 0.03, len(t))
ref.to_csv(DATA_DIR / "mocap_reference.csv", index=False)
sim.to_csv(DATA_DIR / "sim_rollout.csv", index=False)

# 3. Tracking error
err = pd.DataFrame({
    "t":              t,
    "hip_flex_err":   np.abs(ref["hip_flex"]  - sim["hip_flex"]),
    "knee_flex_err":  np.abs(ref["knee_flex"] - sim["knee_flex"]),
    "ankle_flex_err": np.abs(ref["ankle_flex"]- sim["ankle_flex"]),
})
err.to_csv(DATA_DIR / "tracking_error.csv", index=False)

print(f"Wrote sample CSVs to {DATA_DIR}")
