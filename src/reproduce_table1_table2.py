"""
Reproduce_Table1_Table2.py

Description:
    This script reproduces the experimental results for Table 1 (Multi-Seed Validation)
    and Table 2 (Severity Analysis) from the paper.

    It simulates a synthetic cyber stream with:
    1. Concept drift
    2. Sparse attack signatures
    3. Telemetry degradation (Noise, Dropout, Calibration Shift, Bursts)

    It compares a standard Online Logistic Regression model against a
    Delay-Aware Hardened version (LASA/Hardening).

Usage:
    python reproduce_table1_table2.py

Output:
    - results/table1_multiseed.json
    - results/table2_severity_sweep.json
    - results/figures/*.png
"""

import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================
# Configuration & Seeds
# ============================================================
SEEDS = [
    88, 109, 253, 371, 458,
    555, 666, 793, 907, 1009,
    1103, 1201, 1301, 1409, 1511,
    1601, 1789, 1877, 1971, 2025
]

OUTPUT_DIR = "results"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# Ensure output directories exist
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# 1) Synthetic "Cyber Stream" Generator
# ============================================================
def make_synthetic_cyber_stream(
        n_steps=20000,
        n_features=20,
        attack_rate=0.02,
        burst_prob=0.002,
        burst_len_range=(50, 300),
        seed=42,
):
    """
    Generates (X_t, y_t) for a streaming IDS-like setting.
    Includes slow concept drift, rare attacks, and attack bursts.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_steps, n_features)).astype(np.float64)
    y = np.zeros(n_steps, dtype=np.int64)

    # Slow drift in a subset of features
    drift_features = rng.choice(n_features, size=max(2, n_features // 5), replace=False)
    for t in range(n_steps):
        X[t, drift_features] += 0.0005 * t

    # Attack signature subset (discriminative features)
    attack_feats = rng.choice(n_features, size=max(3, n_features // 4), replace=False)

    # Rare single-step attacks: sparse, variable intensity
    attack_idx = rng.choice(n_steps, size=int(n_steps * attack_rate), replace=False)
    y[attack_idx] = 1
    intensity = rng.uniform(0.6, 1.4, size=(len(attack_idx), 1))
    X[np.ix_(attack_idx, attack_feats)] += intensity * rng.normal(
        1.0, 0.6, size=(len(attack_idx), len(attack_feats))
    )

    # Bursty attacks
    t = 0
    while t < n_steps:
        if rng.random() < burst_prob:
            L = int(rng.integers(burst_len_range[0], burst_len_range[1] + 1))
            end = min(n_steps, t + L)
            y[t:end] = 1

            burst_feats = rng.choice(n_features, size=max(3, n_features // 4), replace=False)
            X[t:end, burst_feats] += rng.normal(1.5, 0.8, size=(end - t, len(burst_feats)))
            t = end
        else:
            t += 1

    # Standardize globally
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    return X, y, attack_feats


# ============================================================
# 2) Telemetry Degradation (Faults)
# ============================================================
class Fault:
    def reset(self):
        pass

    def step(self, x, t):
        return x


class GaussianNoise(Fault):
    def __init__(self, sigma=0.10, p=0.35, seed=0):
        self.sigma, self.p = sigma, p
        self.rng = np.random.default_rng(seed)

    def step(self, x, t):
        if self.rng.random() < self.p:
            return x + self.rng.normal(0, self.sigma, size=x.shape)
        return x


class FeatureDropout(Fault):
    def __init__(self, drop_prob=0.15, target_idx=None, seed=1):
        self.drop_prob = drop_prob
        self.target_idx = target_idx
        self.rng = np.random.default_rng(seed)

    def step(self, x, t):
        x2 = x.copy()
        idx = self.target_idx if self.target_idx is not None else np.arange(len(x2))
        mask = (self.rng.random(len(idx)) > self.drop_prob).astype(x2.dtype)
        x2[idx] *= mask
        return x2


class CalibrationShift(Fault):
    def __init__(self, start=9000, scale=1.25, offset=0.20):
        self.start, self.scale, self.offset = start, scale, offset

    def step(self, x, t):
        if t >= self.start:
            return x * self.scale + self.offset
        return x


class BurstCorruption(Fault):
    def __init__(self, start=14000, duration=600, magnitude=0.9):
        self.start, self.duration, self.magnitude = start, duration, magnitude

    def step(self, x, t):
        if self.start <= t < self.start + self.duration:
            return x + self.magnitude
        return x


class FaultPipeline:
    def __init__(self, faults):
        self.faults = faults

    def reset(self):
        for f in self.faults:
            f.reset()

    def step(self, x, t):
        x2 = x
        for f in self.faults:
            x2 = f.step(x2, t)
        return x2


# ============================================================
# 3) Online Model & Hardening Logic
# ============================================================
class OnlineLogReg:
    def __init__(self, n_features, lr=0.02, l2=1e-4):
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2

    def predict_proba(self, x):
        z = float(np.dot(self.w, x) + self.b)
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        ez = np.exp(z)
        return ez / (1.0 + ez)

    def predict(self, x, thr=0.5):
        return 1 if self.predict_proba(x) >= thr else 0

    def update(self, x, y):
        p = self.predict_proba(x)
        grad = (p - y)
        self.w = (1.0 - self.lr * self.l2) * self.w - self.lr * grad * x
        self.b = self.b - self.lr * grad


def generate_hard_case(x, model, n_candidates=48, eps=0.25, keep_l2=True, seed=0):
    """
    Generates a synthetic 'stress case' to harden the model against false negatives.
    """
    rng = np.random.default_rng(seed)
    best = x
    best_score = -1e18
    p0 = model.predict_proba(x)

    for _ in range(n_candidates):
        delta = rng.normal(0, eps, size=x.shape)
        if keep_l2 and np.linalg.norm(delta) > eps * np.sqrt(len(x)) * 1.5:
            continue
        x2 = x + delta
        p = model.predict_proba(x2)

        unc = 1.0 - abs(p - 0.5) * 2.0
        dis = abs(p - p0)
        fn_bias = (0.5 - p)  # bias towards cases the model underestimates

        score = unc + dis + 2.0 * fn_bias

        if score > best_score:
            best_score = score
            best = x2

    return best


# ============================================================
# 4) Metrics
# ============================================================
@dataclass
class OnlineMetrics:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    flips: int = 0
    last_pred: int = None
    in_event: bool = False
    event_start: int = -1
    detected_in_event: bool = False
    delays: list = None

    def __post_init__(self):
        if self.delays is None:
            self.delays = []

    def step(self, t, y_true, y_pred):
        if self.last_pred is not None and y_pred != self.last_pred:
            self.flips += 1
        self.last_pred = y_pred

        if y_true == 1 and y_pred == 1:
            self.tp += 1
        elif y_true == 0 and y_pred == 0:
            self.tn += 1
        elif y_true == 0 and y_pred == 1:
            self.fp += 1
        else:
            self.fn += 1

        # Detection delay logic
        if y_true == 1 and not self.in_event:
            self.in_event = True
            self.event_start = t
            self.detected_in_event = False

        if y_true == 1 and y_pred == 1 and not self.detected_in_event:
            self.detected_in_event = True
            self.delays.append(t - self.event_start)

        if y_true == 0 and self.in_event:
            self.in_event = False
            self.event_start = -1
            self.detected_in_event = False

    def summary(self):
        precision = self.tp / max(1, (self.tp + self.fp))
        recall = self.tp / max(1, (self.tp + self.fn))
        f1 = (2 * precision * recall) / max(1e-12, (precision + recall))
        flip_rate = self.flips / max(1, (self.tp + self.tn + self.fp + self.fn))
        mean_delay = float(np.mean(self.delays)) if len(self.delays) else None
        return {
            "precision": precision, "recall": recall, "f1": f1,
            "flip_rate": flip_rate, "mean_detection_delay_steps": mean_delay,
            "tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn
        }


def _mean_std(values):
    arr = np.array(values, dtype=np.float64)
    if len(arr) <= 1: return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


# ============================================================
# 5) Experiment Runners
# ============================================================
def run_stream_experiment(X, y, faults=None, harden=False, thr=0.5):
    model = OnlineLogReg(n_features=X.shape[1], lr=0.02, l2=1e-4)
    metrics = OnlineMetrics()

    for t in range(len(X)):
        x = X[t]
        if faults is not None:
            x = faults.step(x, t)

        yp = model.predict(x, thr=thr)
        metrics.step(t, y[t], yp)
        model.update(x, y[t])

        if harden and y[t] == 1 and (not metrics.detected_in_event):
            x_hard = generate_hard_case(x, model, n_candidates=48, eps=0.25, seed=t)
            model.update(x_hard, 1)

    return metrics.summary()


def run_one_seed(seed: int):
    X, y, attack_feats = make_synthetic_cyber_stream(seed=seed)

    # Default severe pipeline for Table 1
    fault_pipe = FaultPipeline([
        GaussianNoise(sigma=0.10, p=0.35, seed=seed + 100),
        FeatureDropout(drop_prob=0.15, target_idx=attack_feats, seed=seed + 200),
        CalibrationShift(start=9000, scale=1.25, offset=0.20),
        BurstCorruption(start=14000, duration=600, magnitude=0.9),
    ])

    clean = run_stream_experiment(X, y, faults=None, harden=False)
    faulty = run_stream_experiment(X, y, faults=fault_pipe, harden=False)
    hardened = run_stream_experiment(X, y, faults=fault_pipe, harden=True)
    return clean, faulty, hardened


# --- Table 1 Logic ---
def run_table1_experiment(seeds, save_json=True):
    print("\n" + "=" * 50)
    print(" Reproducing Table 1: Multi-Seed Validation")
    print("=" * 50)
    clean_list, faulty_list, hard_list = [], [], []

    for seed in seeds:
        clean, faulty, hardened = run_one_seed(seed)
        clean_list.append(clean)
        faulty_list.append(faulty)
        hard_list.append(hardened)

    def summarize(name, results):
        keys = ["f1", "mean_detection_delay_steps", "flip_rate"]
        print(f"\n--- {name} ---")
        for key in keys:
            vals = [r[key] for r in results if r[key] is not None]
            m, s = _mean_std(vals)
            print(f"{key:28s} {m:.6f} ± {s:.6f}")

    summarize("CLEAN", clean_list)
    summarize("FAULT-INJECTED", faulty_list)
    summarize("FAULT + HARDENING (LASA)", hard_list)

    if save_json:
        out_path = os.path.join(OUTPUT_DIR, "table1_multiseed.json")
        out = {"seeds": list(seeds), "clean": clean_list, "fault": faulty_list, "hardened": hard_list}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n[Saved] {out_path}")


# --- Table 2 Logic ---
def severity_to_fault_params(level: int):
    def lerp(a, b, t): return a + (b - a) * t

    t = (level - 1) / 4.0
    return {
        "sigma": lerp(0.03, 0.14, t),
        "noise_p": lerp(0.15, 0.45, t),
        "drop_prob": lerp(0.05, 0.30, t),
        "scale": lerp(1.05, 1.40, t),
        "offset": lerp(0.05, 0.35, t),
        "calib_start": 9000,
        "burst_start": 14000,
        "burst_duration": int(round(lerp(200, 900, t))),
        "burst_magnitude": lerp(0.30, 1.20, t),
    }


def build_fault_pipeline(severity_level, attack_feats, seed):
    p = severity_to_fault_params(severity_level)
    return FaultPipeline([
        GaussianNoise(sigma=p["sigma"], p=p["noise_p"], seed=seed + 100),
        FeatureDropout(drop_prob=p["drop_prob"], target_idx=attack_feats, seed=seed + 200),
        CalibrationShift(start=p["calib_start"], scale=p["scale"], offset=p["offset"]),
        BurstCorruption(start=p["burst_start"], duration=p["burst_duration"], magnitude=p["burst_magnitude"]),
    ])


def run_table2_experiment(seeds, severity_levels=(1, 2, 3, 4, 5), save_json=True):
    print("\n" + "=" * 50)
    print(" Reproducing Table 2: Severity Analysis")
    print("=" * 50)

    results = {
        "seeds": list(seeds),
        "severity_levels": list(severity_levels),
        "summary": {}
    }

    for sev in severity_levels:
        fault_runs, hard_runs = [], []
        for seed in seeds:
            X, y, attack_feats = make_synthetic_cyber_stream(seed=seed)

            # Run Faulty
            fp1 = build_fault_pipeline(sev, attack_feats, seed)
            fault_runs.append(run_stream_experiment(X, y, faults=fp1, harden=False))

            # Run Hardened
            fp2 = build_fault_pipeline(sev, attack_feats, seed)
            hard_runs.append(run_stream_experiment(X, y, faults=fp2, harden=True))

        # Aggregation
        summary = {}
        for k in ["f1", "mean_detection_delay_steps", "flip_rate"]:
            f_vals = [r[k] for r in fault_runs if r[k] is not None]
            h_vals = [r[k] for r in hard_runs if r[k] is not None]
            mf, sf = _mean_std(f_vals)
            mh, sh = _mean_std(h_vals)
            summary[k] = {"fault_mean": mf, "fault_std": sf, "hard_mean": mh, "hard_std": sh}

        results["summary"][str(sev)] = summary

        s = summary
        print(f"\n[Severity S{sev}]")
        print(
            f"Delay:     Fault {s['mean_detection_delay_steps']['fault_mean']:.2f} -> Hard {s['mean_detection_delay_steps']['hard_mean']:.2f}")

    if save_json:
        out_path = os.path.join(OUTPUT_DIR, "table2_severity_sweep.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] {out_path}")

    return results


def plot_results(results):
    sev_levels = [int(s) for s in results["severity_levels"]]
    metrics = [
        ("f1", "F1 Score", "f1_vs_severity.png"),
        ("mean_detection_delay_steps", "Detection Delay", "delay_vs_severity.png")
    ]

    for key, label, fname in metrics:
        y_f = [results["summary"][str(s)][key]["fault_mean"] for s in sev_levels]
        y_h = [results["summary"][str(s)][key]["hard_mean"] for s in sev_levels]

        plt.figure(figsize=(6, 4))
        plt.plot(sev_levels, y_f, 'o--', label="Standard (Fault)")
        plt.plot(sev_levels, y_h, 's-', label="LASA (Hardened)")
        plt.title(f"{label} across Severity Levels")
        plt.xlabel("Severity (1=Low, 5=High)")
        plt.ylabel(label)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, fname), dpi=300)
        plt.close()
    print(f"\n[Figures Saved] to {FIGURE_DIR}/")


# ============================================================
# Main Execution
# ============================================================
def main():
    # 1. Reproduce Table 1 (Multi-seed validation)
    run_table1_experiment(SEEDS)

    # 2. Reproduce Table 2 (Severity sweep)
    res_table2 = run_table2_experiment(SEEDS, severity_levels=(1, 2, 3, 4, 5))

    # 3. Generate plots
    plot_results(res_table2)


if __name__ == "__main__":
    main()