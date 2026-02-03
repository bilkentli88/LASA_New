"""
Reproduce_Table3_Ablation.py

Description:
    This script reproduces the Mechanistic Ablation Study (Table 3) from the paper.
    It dissects the LASA algorithm to quantify the contribution of each component:

    1. Baseline (Degraded): No adaptation.
    2. Random Noise Injection: "Blind" synthesis (Proves targeted optimization is needed).
    3. Uncertainty Only: Manifold sampling without direction (Proves latency bias is needed).
    4. Full Generative Repair (LASA): Gradient-guided, latency-aware synthesis.

Usage:
    python reproduce_table3_ablation.py

Output:
    - results/table3_ablation.json
    - results/figures/ablation_delay.png
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
# PART 1: Shared Environment (Same as Table 1/2)
# ============================================================
# To ensure this script is standalone, we include the core environment classes here.

def make_synthetic_cyber_stream(n_steps=20000, n_features=20, attack_rate=0.02, burst_prob=0.002, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_steps, n_features)).astype(np.float64)
    y = np.zeros(n_steps, dtype=np.int64)

    # Concept Drift
    drift_features = rng.choice(n_features, size=max(2, n_features // 5), replace=False)
    for t in range(n_steps):
        X[t, drift_features] += 0.0005 * t

    # Attack Signatures
    attack_feats = rng.choice(n_features, size=max(3, n_features // 4), replace=False)
    attack_idx = rng.choice(n_steps, size=int(n_steps * attack_rate), replace=False)
    y[attack_idx] = 1
    intensity = rng.uniform(0.6, 1.4, size=(len(attack_idx), 1))
    X[np.ix_(attack_idx, attack_feats)] += intensity * rng.normal(1.0, 0.6, size=(len(attack_idx), len(attack_feats)))

    # Bursts
    t = 0
    while t < n_steps:
        if rng.random() < burst_prob:
            L = int(rng.integers(50, 301))
            end = min(n_steps, t + L)
            y[t:end] = 1
            burst_feats = rng.choice(n_features, size=max(3, n_features // 4), replace=False)
            X[t:end, burst_feats] += rng.normal(1.5, 0.8, size=(end - t, len(burst_feats)))
            t = end
        else:
            t += 1

    # Global Standardization
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    return X, y, attack_feats


# --- Fault Pipeline ---
class FaultPipeline:
    def __init__(self, faults): self.faults = faults

    def step(self, x, t):
        for f in self.faults: x = f.step(x, t)
        return x


class GaussianNoise:
    def __init__(self, sigma, p, seed):
        self.sigma, self.p, self.rng = sigma, p, np.random.default_rng(seed)

    def step(self, x, t):
        return x + self.rng.normal(0, self.sigma, size=x.shape) if self.rng.random() < self.p else x


class FeatureDropout:
    def __init__(self, drop_prob, target_idx, seed):
        self.drop_prob, self.target_idx, self.rng = drop_prob, target_idx, np.random.default_rng(seed)

    def step(self, x, t):
        x2 = x.copy()
        mask = (self.rng.random(len(self.target_idx)) > self.drop_prob)
        x2[self.target_idx] *= mask
        return x2


class CalibrationShift:
    def __init__(self, start, scale, offset): self.start, self.scale, self.offset = start, scale, offset

    def step(self, x, t): return x * self.scale + self.offset if t >= self.start else x


class BurstCorruption:
    def __init__(self, start, duration, magnitude): self.start, self.end, self.mag = start, start + duration, magnitude

    def step(self, x, t): return x + self.mag if self.start <= t < self.end else x


def build_fault_pipeline(severity_level, attack_feats, seed):
    # Severity mapping (S1-S5)
    t = (severity_level - 1) / 4.0
    p = {
        "sigma": 0.03 + (0.11 * t), "noise_p": 0.15 + (0.30 * t),
        "drop_prob": 0.05 + (0.25 * t), "scale": 1.05 + (0.35 * t),
        "offset": 0.05 + (0.30 * t), "burst_mag": 0.30 + (0.90 * t)
    }
    return FaultPipeline([
        GaussianNoise(p["sigma"], p["noise_p"], seed + 100),
        FeatureDropout(p["drop_prob"], attack_feats, seed + 200),
        CalibrationShift(9000, p["scale"], p["offset"]),
        BurstCorruption(14000, int(200 + 700 * t), p["burst_mag"])
    ])


# --- Online Model & Metrics ---
class OnlineLogReg:
    def __init__(self, n_features, lr=0.02, l2=1e-4):
        self.w, self.b, self.lr, self.l2 = np.zeros(n_features), 0.0, lr, l2

    def predict_proba(self, x):
        z = np.dot(self.w, x) + self.b
        return 1.0 / (1.0 + np.exp(-z)) if z >= 0 else np.exp(z) / (1.0 + np.exp(z))

    def predict(self, x, thr=0.5): return 1 if self.predict_proba(x) >= thr else 0

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
        if y_true == 1 and y_pred == 1:
            self.tp += 1
        elif y_true == 0 and y_pred == 0:
            self.tn += 1
        elif y_true == 0 and y_pred == 1:
            self.fp += 1
        else:
            self.fn += 1

        if y_true == 1 and not self.in_event: self.in_event, self.event_start, self.detected_in_event = True, t, False
        if y_true == 1 and y_pred == 1 and not self.detected_in_event: self.detected_in_event, _ = True, self.delays.append(
            t - self.event_start)
        if y_true == 0: self.in_event = False

    def summary(self):
        rec = self.tp / max(1, self.tp + self.fn)
        prec = self.tp / max(1, self.tp + self.fp)
        return {
            "f1": 2 * prec * rec / max(1e-12, prec + rec),
            "mean_detection_delay_steps": np.mean(self.delays) if self.delays else 0.0,
            "flip_rate": self.flips / max(1, self.tp + self.tn + self.fp + self.fn)
        }


# ============================================================
# PART 2: Generative Ablation Logic (The Core Study)
# ============================================================

def generate_ablation_case(x, model, strategy="optimized", n_candidates=48, eps=0.25, seed=0, use_fn_bias=True):
    """
    Generates a stress case based on the specific ablation strategy.

    Strategies:
      - "random": Returns x + Gaussian Noise (Blind Synthesis).
      - "optimized": Runs manifold alignment optimization.
         - use_fn_bias=False: Uncertainty-Only (No directional guidance).
         - use_fn_bias=True:  Full LASA (Latency-guided).
    """
    rng = np.random.default_rng(seed)

    # Strategy 1: Random Noise (Isotropic)
    if strategy == "random":
        return x + rng.normal(0, eps, size=x.shape)

    # Strategy 2: Optimized Synthesis
    best, best_score = x, -1e18
    p0 = model.predict_proba(x)

    for _ in range(n_candidates):
        delta = rng.normal(0, eps, size=x.shape)
        if np.linalg.norm(delta) > eps * np.sqrt(len(x)) * 1.5: continue

        x2 = x + delta
        p = model.predict_proba(x2)

        # Component A: Uncertainty
        unc = 1.0 - abs(p - 0.5) * 2.0
        # Component B: Distributional Distance
        dis = abs(p - p0)
        # Component C: Latency Bias (Directional)
        fn_bias = (0.5 - p) if use_fn_bias else 0.0

        score = unc + dis + (2.0 * fn_bias)
        if score > best_score: best_score, best = score, x2

    return best


def run_stream_ablation(X, y, faults, mode, gen_strategy, use_fn_bias, thr=0.5):
    model = OnlineLogReg(n_features=X.shape[1])
    metrics = OnlineMetrics()

    for t in range(len(X)):
        x = X[t]
        if faults: x = faults.step(x, t)

        yp = model.predict(x, thr=thr)
        metrics.step(t, y[t], yp)
        model.update(x, y[t])

        # Trigger Logic: Only during Undetected Attack Onset
        if mode == "triggered" and y[t] == 1 and (not metrics.detected_in_event):
            x_hard = generate_ablation_case(
                x, model, strategy=gen_strategy, seed=t, use_fn_bias=use_fn_bias
            )
            model.update(x_hard, 1)

    return metrics.summary()


def mean_std(vals):
    arr = np.array(vals)
    if len(arr) <= 1: return arr.mean(), 0.0
    return arr.mean(), arr.std(ddof=1)


# ============================================================
# PART 3: Execution & Plotting
# ============================================================

def run_ablation_suite(severities=(3, 5), seeds=SEEDS):
    configs = [
        ("Baseline", dict(mode="none", gen_strategy="none", use_fn_bias=False)),
        ("Random Noise", dict(mode="triggered", gen_strategy="random", use_fn_bias=False)),
        ("Uncertainty Only", dict(mode="triggered", gen_strategy="optimized", use_fn_bias=False)),
        ("Full LASA", dict(mode="triggered", gen_strategy="optimized", use_fn_bias=True)),
    ]

    results = {}

    for sev in severities:
        print(f"\n{'=' * 60}\n Ablation Study @ Severity S{sev}\n{'=' * 60}")
        print(f"{'Method':<20} | {'Delay':<15} | {'F1 Score':<15}")
        print("-" * 56)

        results[sev] = {}

        for name, cfg in configs:
            delays, f1s = [], []
            for seed in seeds:
                X, y, feats = make_synthetic_cyber_stream(seed=seed)
                fp = build_fault_pipeline(sev, feats, seed)

                out = run_stream_ablation(X, y, fp, **cfg)
                delays.append(out["mean_detection_delay_steps"])
                f1s.append(out["f1"])

            md, sd = mean_std(delays)
            mf, sf = mean_std(f1s)
            results[sev][name] = {"delay_mean": md, "delay_std": sd}

            print(f"{name:<20} | {md:.2f} +/- {sd:.2f}   | {mf:.4f}")

    # Save Results
    with open(os.path.join(OUTPUT_DIR, "table3_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_ablation(results):
    # Plotting Delay for S3 vs S5
    s3 = results[3]
    s5 = results[5]

    labels = list(s3.keys())
    s3_means = [s3[k]["delay_mean"] for k in labels]
    s3_errs = [s3[k]["delay_std"] for k in labels]
    s5_means = [s5[k]["delay_mean"] for k in labels]
    s5_errs = [s5[k]["delay_std"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, s3_means, width, yerr=s3_errs, label='S3 (High Noise)', capsize=5, alpha=0.8)
    plt.bar(x + width / 2, s5_means, width, yerr=s5_errs, label='S5 (Severe)', capsize=5, alpha=0.8)

    plt.ylabel('Mean Detection Delay (Steps)')
    plt.title('Ablation Study: Contribution of Generative Components')
    plt.xticks(x, labels, rotation=15)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, "ablation_delay.png")
    plt.savefig(path, dpi=300)
    print(f"\n[Figure Saved] {path}")


if __name__ == "__main__":
    res = run_ablation_suite(severities=(3, 5))
    plot_ablation(res)