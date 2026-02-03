"""
Reproduce_Table5_RealWorld.py

Description:
    This is the MAIN reproduction script for the paper.
    It evaluates LASA on the real-world CIC-IDS2017 dataset (Friday DDoS capture).

    It reproduces Table 5 by comparing 5 methods under 3 noise regimes:
    1. Standard Online Learning (Baseline)
    2. Adversarial Training (PGD)
    3. Dynamic Threshold Regulation (DTR)
    4. Random Stress Augmentation (RSA)
    5. LASA (Ours)

    Metrics Reported:
    - Detection Delay (Mean & Max)
    - F1-Score
    - Prediction Stability (Flip Rate)

Usage:
    python reproduce_table5_realworld.py

Requirements:
    - The file 'CICIDS2017_day.csv' must be in the same directory.

Output:
    - results/table5_realworld.json
    - Console summary table
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from joblib import Parallel, delayed

# =========================
# CONFIGURATION
# =========================
DATA_FILENAME = "CICIDS2017_day.csv"
OUTPUT_DIR = "results"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "table5_realworld.json")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 20 Fixed Seeds for Reproducibility
SEEDS = [
    88, 109, 253, 371, 458, 555, 666, 793, 907, 1009,
    1103, 1201, 1301, 1409, 1511, 1601, 1789, 1877, 1971, 2025
]

# Hyperparameters
TOPK_FEATURES = 20
HARDENING_STEPS = 5
PROBA_TRIGGER_LOW = 0.3
PROBA_TRIGGER_HIGH = 0.7
UNCERTAINTY_THRESHOLD = 0.60
FN_BIAS_WEIGHT = 5.0
PGD_EPSILON = 0.5
CLIP_VALUE = 1e12

# Operational Settings
WARMUP_STEPS = 2000
PREDICT_START_INDEX = 2000
N_JOBS = -1  # Use all cores


# =========================
# FAULT INJECTION MODEL
# =========================
@dataclass
class FaultParams:
    noise_std: float;
    dropout_p: float;
    drift_scale: float
    burst_p: float;
    burst_len: int;
    burst_noise_mult: float


# Noise Regimes (S1=Low, S2=Medium, S3=High/Severe)
SEVERITIES = {
    "S1": FaultParams(0.01, 0.02, 0.002, 0.002, 10, 2.0),
    "S2": FaultParams(0.05, 0.10, 0.015, 0.01, 20, 3.0),
    "S3": FaultParams(0.15, 0.30, 0.05, 0.02, 30, 6.0),
}


class TelemetryDegradation:
    def __init__(self, params, rng):
        self.p = params
        self.rng = rng
        self.t = 0
        self.in_burst_until = -1

    def apply(self, x):
        x_f = x.copy()

        # 1. Calibration Drift
        drift = 1.0 + (self.p.drift_scale * self.t)
        mask = self.rng.random(x_f.shape[0]) < 0.2
        x_f[mask] *= drift

        # 2. Burst Noise
        if self.t > self.in_burst_until and (self.rng.random() < self.p.burst_p):
            self.in_burst_until = self.t + self.p.burst_len
        in_burst = self.t <= self.in_burst_until

        # 3. Additive Gaussian Noise
        std = self.p.noise_std * (self.p.burst_noise_mult if in_burst else 1.0)
        if std > 0: x_f += self.rng.normal(0.0, std, size=x_f.shape[0])

        # 4. Feature Dropout
        drop_p = min(0.99, self.p.dropout_p * (2.0 if in_burst else 1.0))
        if drop_p > 0: x_f[self.rng.random(x_f.shape[0]) < drop_p] = 0.0

        self.t += 1
        return x_f


# =========================
# DYNAMIC NEURAL NETWORK
# =========================
class DynamicMLP(nn.Module):
    def __init__(self, input_dim):
        super(DynamicMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x): return self.net(x)

    def predict_proba(self, x):
        with torch.no_grad(): return torch.sigmoid(self.forward(x)).item()

    def get_input_gradient(self, x, target):
        x_grad = x.clone().detach().requires_grad_(True)
        loss = self.loss_fn(self.forward(x_grad), target).mean()
        loss.backward()
        return x_grad.grad.detach().numpy()

    def online_update(self, x, y, weight=1.0):
        self.optimizer.zero_grad()
        loss = (self.loss_fn(self.forward(x), y) * weight).mean()
        loss.backward()
        self.optimizer.step()


def compute_detection_delay(y_true, y_pred, onset_idx):
    for t in range(onset_idx, len(y_true)):
        if y_true[t] == 1 and y_pred[t] == 1: return t - onset_idx
    return None


def get_topk_indices(grad, k): return np.argsort(np.abs(grad.ravel()))[-k:]


# =========================
# EXPERIMENT LOOP
# =========================
def run_single_seed(X, y, onset_idx, seed, fault_params):
    device = torch.device("cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    injector = TelemetryDegradation(fault_params, rng)
    scaler = StandardScaler()

    methods = ["Standard", "PGD", "DTR", "RSA", "LASA"]
    results = {}

    for mode in methods:
        # Reset Randomness & Model for each method to ensure fair comparison
        rng = np.random.default_rng(seed)
        injector = TelemetryDegradation(fault_params, rng)
        model = DynamicMLP(X.shape[1]).to(device)

        # Warmup
        warm_end = min(WARMUP_STEPS, len(X))
        WX = [injector.apply(X[i]) for i in range(warm_end)]
        scaler.fit(WX)
        wx_t = torch.tensor(scaler.transform(WX), dtype=torch.float32).to(device)
        wy_t = torch.tensor(y[:warm_end].reshape(-1, 1), dtype=torch.float32).to(device)
        for _ in range(5): model.online_update(wx_t, wy_t)

        # Streaming
        y_pred = np.zeros_like(y)
        for t in range(warm_end, len(X)):
            x_raw = injector.apply(X[t])
            x_scaled = scaler.transform(x_raw.reshape(1, -1))
            xt = torch.tensor(x_scaled, dtype=torch.float32).to(device)
            yt = torch.tensor([[y[t]]], dtype=torch.float32).to(device)

            proba = model.predict_proba(xt)

            # Dynamic Thresholding
            thresh = 0.5
            if mode in ["LASA", "RSA", "DTR"]:
                if PROBA_TRIGGER_LOW < proba < PROBA_TRIGGER_HIGH: thresh = 0.35
            y_pred[t] = 1 if proba >= thresh else 0

            model.online_update(xt, yt)

            # Repair Logic
            if (y[t] == 1) and (proba < UNCERTAINTY_THRESHOLD):
                target_one = torch.tensor([[1.0]], dtype=torch.float32).to(device)
                dynamic_step = max(0.5, fault_params.noise_std * 5.0)

                if mode == "PGD":
                    grad = model.get_input_gradient(xt, yt)
                    p_xt = torch.tensor(x_scaled + np.sign(grad) * PGD_EPSILON, dtype=torch.float32).to(device)
                    model.online_update(p_xt, yt)

                elif mode == "LASA":
                    grad = model.get_input_gradient(xt, target_one)
                    feats = get_topk_indices(grad, TOPK_FEATURES)
                    for _ in range(HARDENING_STEPS):
                        s = x_raw.copy()
                        s[feats] += rng.normal(0, fault_params.noise_std * 2.0, size=len(feats))
                        s_s = scaler.transform(s.reshape(1, -1))
                        s_s[0, feats] += (np.sign(grad[0, feats]) * dynamic_step)
                        model.online_update(torch.tensor(s_s).float().to(device), target_one, weight=FN_BIAS_WEIGHT)

                elif mode == "RSA":
                    for _ in range(HARDENING_STEPS):
                        s = x_raw.copy()
                        idx = rng.choice(X.shape[1], TOPK_FEATURES, replace=False)
                        s[idx] += rng.normal(0, dynamic_step, size=len(idx))
                        s_s = scaler.transform(s.reshape(1, -1))
                        model.online_update(torch.tensor(s_s).float().to(device), target_one, weight=FN_BIAS_WEIGHT)

        delay = compute_detection_delay(y, y_pred, onset_idx)
        eval_p = y_pred[PREDICT_START_INDEX:]
        eval_y = y[PREDICT_START_INDEX:]
        flips = np.mean(np.abs(eval_p[1:] - eval_p[:-1])) if len(eval_p) > 1 else 0

        results[mode] = {"delay": delay, "f1": f1_score(eval_y, eval_p, zero_division=0), "flip": flips}

    return seed, results


# =========================
# MAIN
# =========================
def main():
    print(f"\n{'=' * 60}\n LASA REPRODUCTION: Table 5 (Real World)\n{'=' * 60}")

    if not os.path.exists(DATA_FILENAME):
        print(f"[Error] '{DATA_FILENAME}' not found. Please download CIC-IDS2017 dataset.")
        sys.exit(1)

    print("Loading Data...")
    df = pd.read_csv(DATA_FILENAME, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    label_col = [c for c in ["Label", "label", "Attack"] if c in df.columns][0]
    y_raw = df[label_col].astype(str).values
    y = np.array([0 if v in ["BENIGN", "Benign", "0"] else 1 for v in y_raw], dtype=int)
    X = df.select_dtypes(include=[np.number]).fillna(0).clip(-CLIP_VALUE, CLIP_VALUE).values

    # Find Onset
    onset_idx = 0
    in_seg, start = False, 0
    segments = []
    for i, v in enumerate(y):
        if v == 1 and not in_seg:
            in_seg, start = True, i
        elif v == 0 and in_seg:
            segments.append((start, i - 1)); in_seg = False
    if in_seg: segments.append((start, len(y) - 1))

    cands = [s for s in segments if s[0] > WARMUP_STEPS]
    if cands: onset_idx = max(cands, key=lambda x: x[1] - x[0])[0]  # Max duration attack

    end_idx = min(len(y), onset_idx + 2500)
    X, y = X[:end_idx], y[:end_idx]
    print(f"Data Loaded. Length: {len(y)}, Attack Onset: {onset_idx}")

    final_results = {}

    for sev_name, sev_params in SEVERITIES.items():
        print(f"\nRunning Regime: {sev_name}...")
        results_list = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_seed)(X, y, onset_idx, seed, sev_params) for seed in SEEDS
        )

        sev_data = {str(seed): res for seed, res in results_list}

        # Summarize
        print(f"{'-' * 60}")
        print(f"{'Method':<10} | {'Delay (Mean / Max)':<20} | {'F1 Score':<10} | {'Flip Rate':<12}")
        print(f"{'-' * 60}")

        summary = {}
        for m in ["Standard", "PGD", "DTR", "RSA", "LASA"]:
            delays = [sev_data[str(s)][m]["delay"] for s in SEEDS if sev_data[str(s)][m]["delay"] is not None]
            d_mean = np.mean(delays) if delays else -1
            d_max = np.max(delays) if delays else -1
            f1 = np.mean([sev_data[str(s)][m]["f1"] for s in SEEDS])
            flip = np.mean([sev_data[str(s)][m]["flip"] for s in SEEDS])

            print(f"{m:<10} | {d_mean:.2f} / {d_max:<4}     | {f1:.3f}      | {flip:.4f}")
            summary[m] = {"delay_mean": float(d_mean), "delay_max": float(d_max), "f1": float(f1), "flip": float(flip)}

        final_results[sev_name] = {"summary": summary, "raw": sev_data}

    with open(OUTPUT_FILENAME, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[Saved] {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()