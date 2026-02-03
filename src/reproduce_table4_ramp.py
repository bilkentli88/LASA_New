"""
Reproduce_Table4_Ramp.py

Description:
    This script reproduces the "Early Detection of Incipient Drifts" experiment (Table 4).
    It evaluates the system's ability to detect "Ramp Attacks"—attacks that start
    with zero intensity and gradually increase over time (linear drift).

    This tests the "Latency-Robustness Trade-off" in dynamic environments.
    It compares:
    1. Random Noise Injection (Stochastic Baseline)
    2. Full Generative Repair (LASA - Gradient Guided)

    Hypothesis: In high-dimensional spaces (Dim=100), random noise fails to find
    the specific drift direction, while LASA's gradient guidance detects it much earlier.

Usage:
    python reproduce_table4_ramp.py

Output:
    - results/table4_ramp.json
    - results/figures/ramp_detection_gap.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================
# Configuration
# ============================================================
SEEDS = [
    88, 109, 253, 371, 458,
    555, 666, 793, 907, 1009,
    1103, 1201, 1301, 1409, 1511,
    1601, 1789, 1877, 1971, 2025
]

OUTPUT_DIR = "results"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# PART 1: Core Classes (Standalone)
# ============================================================

class OnlineLogReg:
    def __init__(self, n_features, lr=0.01, l2=1e-4):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2

    def predict_proba(self, x):
        z = np.dot(self.w, x) + self.b
        return 1.0 / (1.0 + np.exp(-z)) if z >= 0 else np.exp(z) / (1.0 + np.exp(z))

    def predict(self, x, thr=0.5):
        return 1 if self.predict_proba(x) >= thr else 0

    def update(self, x, y):
        p = self.predict_proba(x)
        grad = p - y
        self.w = (1 - self.lr * self.l2) * self.w - self.lr * grad * x
        self.b -= self.lr * grad


@dataclass
class OnlineMetrics:
    tp: int = 0;
    tn: int = 0;
    fp: int = 0;
    fn: int = 0;
    flips: int = 0
    last_pred: int = None;
    in_event: bool = False;
    event_start: int = -1;
    detected_in_event: bool = False
    delays: list = None

    def __post_init__(self):
        self.delays = []

    def step(self, t, y_true, y_pred):
        if self.last_pred is not None and y_pred != self.last_pred: self.flips += 1
        self.last_pred = y_pred

        if y_true == 1 and not self.in_event: self.in_event, self.event_start, self.detected_in_event = True, t, False
        if y_true == 1 and y_pred == 1 and not self.detected_in_event: self.detected_in_event, _ = True, self.delays.append(
            t - self.event_start)
        if y_true == 0: self.in_event = False

    def summary(self):
        return {"mean_detection_delay_steps": np.mean(self.delays) if self.delays else None}


# ============================================================
# PART 2: Ramp Attack Generator
# ============================================================
def make_ramp_stream(n_steps, n_features, seed):
    rng = np.random.default_rng(seed)

    # 1. Background Noise
    X = rng.normal(0, 1.0, size=(n_steps, n_features))
    y = np.zeros(n_steps, dtype=np.int64)

    # 2. Define a "Drift Direction" (The secret attack vector)
    drift_vector = rng.normal(0, 1, size=n_features)
    drift_vector /= np.linalg.norm(drift_vector)  # Normalize

    # 3. Inject Drifting Attacks (Ramps)
    n_attacks = 10
    attack_len = 100

    for _ in range(n_attacks):
        start = rng.integers(100, n_steps - attack_len - 100)
        end = start + attack_len
        y[start:end] = 1

        # Ramp: 0.0 -> 1.5 sigma (Linear Growth)
        # This forces a delay; the question is "how long?"
        slope = np.linspace(0, 1.5, attack_len)

        for i in range(attack_len):
            X[start + i] += slope[i] * drift_vector

    return X, y


# ============================================================
# PART 3: Experiment Logic
# ============================================================
def run_stream_local(X, y, gen_strategy, use_fn_bias):
    model = OnlineLogReg(n_features=X.shape[1], lr=0.01, l2=1e-4)
    metrics = OnlineMetrics()
    rng = np.random.default_rng(42)

    for t in range(len(X)):
        x = X[t]
        yp = model.predict(x, thr=0.5)
        metrics.step(t, y[t], yp)
        model.update(x, y[t])

        # Trigger on Missed Attack (False Negative)
        if y[t] == 1 and (not metrics.detected_in_event):

            # 1. Random Noise Strategy
            if gen_strategy == "random":
                x_hard = x + rng.normal(0, 0.5, size=x.shape)
                model.update(x_hard, 1)

            # 2. Optimized (LASA) Strategy
            else:
                n_cands = max(64, int(4 * len(x)))  # Scale search with dim
                best, best_score = x, -1e18
                p0 = model.predict_proba(x)

                for _ in range(n_cands):
                    delta = rng.normal(0, 0.5, size=x.shape)
                    x2 = x + delta
                    p = model.predict_proba(x2)

                    unc = 1.0 - abs(p - 0.5) * 2.0
                    dis = abs(p - p0)
                    fn_bias = (0.5 - p) if use_fn_bias else 0.0  # Latency bias

                    score = unc + dis + (2.0 * fn_bias)
                    if score > best_score: best_score, best = score, x2

                model.update(best, 1)

    return metrics.summary()


def mean_std(data):
    clean = [d for d in data if d is not None]
    if not clean: return 0.0, 0.0
    arr = np.array(clean)
    return arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0


# ============================================================
# PART 4: Execution & Visualization
# ============================================================
def run_ramp_experiment():
    dimensions = [20, 100]
    configs = [
        ("Random Noise", dict(gen_strategy="random", use_fn_bias=False)),
        ("LASA (Ours)", dict(gen_strategy="optimized", use_fn_bias=True)),
    ]

    print(f"\nRunning RAMP ATTACK Experiment (Drift Detection)...")
    print("-" * 65)
    print(f"{'Dim':<5} | {'Method':<16} | {'Delay (Steps)':<22} | {'Gap'}")
    print("-" * 65)

    final_results = {}

    for n_feat in dimensions:
        final_results[str(n_feat)] = {}

        # Store results for plotting
        res_map = {}

        for name, cfg in configs:
            delays = []
            for seed in SEEDS:
                X, y = make_ramp_stream(n_steps=20000, n_features=n_feat, seed=seed)
                out = run_stream_local(X, y, **cfg)

                val = out["mean_detection_delay_steps"]
                if val is None: val = 50.0  # Penalty for complete miss
                delays.append(val)

            res_map[name] = delays
            final_results[str(n_feat)][name] = delays

        # Calc Stats
        rand_m, rand_s = mean_std(res_map["Random Noise"])
        lasa_m, lasa_s = mean_std(res_map["LASA (Ours)"])
        gap = rand_m - lasa_m

        print(f"{n_feat:<5} | Random Noise     | {rand_m:.2f} +/- {rand_s:.2f}")
        print(f"{'':<5} | LASA (Ours)      | {lasa_m:.2f} +/- {lasa_s:.2f}")
        print(f"{'':<5} | >> GAP           | +{gap:.2f} steps (Saved)")
        print("-" * 65)

    # Save JSON
    with open(os.path.join(OUTPUT_DIR, "table4_ramp.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    return final_results


def plot_ramp_results(results):
    dims = ["20", "100"]
    rand_means = [mean_std(results[d]["Random Noise"])[0] for d in dims]
    lasa_means = [mean_std(results[d]["LASA (Ours)"])[0] for d in dims]

    x = np.arange(len(dims))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, rand_means, width, label='Random Noise', color='#d62728', alpha=0.8)
    plt.bar(x + width / 2, lasa_means, width, label='LASA (Ours)', color='#1f77b4', alpha=0.8)

    plt.ylabel('Mean Detection Delay (Steps)')
    plt.xlabel('Feature Dimensionality')
    plt.title('Early Detection of Incipient Drifts (Ramp Attack)')
    plt.xticks(x, [f"Dim={d}" for d in dims])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # Add gap annotation
    gap_100 = rand_means[1] - lasa_means[1]
    plt.text(1, rand_means[1] + 2, f"Gap: +{gap_100:.1f}", ha='center', fontweight='bold')

    path = os.path.join(FIGURE_DIR, "ramp_detection_gap.png")
    plt.savefig(path, dpi=300)
    print(f"\n[Figure Saved] {path}")


if __name__ == "__main__":
    res = run_ramp_experiment()
    plot_ramp_results(res)