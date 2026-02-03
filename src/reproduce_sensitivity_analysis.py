"""
Reproduce_Sensitivity_Analysis.py

Description:
    This script reproduces the "Sensitivity to Trigger Thresholds" experiment (Section 6.1.4).
    It verifies that LASA's performance is robust to changes in its hyperparameter
    settings (specifically the uncertainty triggers: tau_lower, tau_upper).

    Hypothesis:
    The method should yield consistent detection delays (approx 4.0 steps) even
    when the thresholds vary significantly, proving it does not require precise tuning.

Usage:
    python reproduce_sensitivity_analysis.py

Output:
    - results/sensitivity_analysis.json
    - Console Grid Table
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# =========================
# CONFIGURATION
# =========================
DATA_FILENAME = "CICIDS2017_day.csv"
OUTPUT_DIR = "results"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "sensitivity_analysis.json")
N_JOBS = -1

# We use a subset of seeds for this specific ablation to save time,
# as established in the paper (Section 6.1.4).
SEEDS = [88, 109, 253, 371, 458]

# Fixed LASA Parameters
TOPK_FEATURES = 20
HARDENING_STEPS = 5
UNCERTAINTY_THRESHOLD = 0.60
FN_BIAS_WEIGHT = 5.0
CLIP_VALUE = 1e12
WARMUP_STEPS = 2000

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# CORE CLASSES
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

    def get_input_gradient(self, x, y):
        x_grad = x.clone().detach().requires_grad_(True)
        loss = self.loss_fn(self.forward(x_grad), y).mean()
        loss.backward()
        return x_grad.grad.detach().numpy()

    def online_update(self, x, y, weight=1.0):
        self.optimizer.zero_grad()
        loss = (self.loss_fn(self.forward(x), y) * weight).mean()
        loss.backward()
        self.optimizer.step()


class TelemetryDegradation:
    def __init__(self, rng):
        # HARDCODED S3 (SEVERE) PARAMETERS for stress testing
        self.noise_std = 0.15
        self.dropout_p = 0.30
        self.drift_scale = 0.05
        self.burst_p = 0.02
        self.burst_len = 30
        self.burst_mult = 6.0
        self.rng = rng
        self.t = 0
        self.in_burst_until = -1

    def apply(self, x):
        x_f = x.copy()
        drift = 1.0 + (self.drift_scale * self.t)
        mask = self.rng.random(x_f.shape[0]) < 0.2
        x_f[mask] *= drift

        if self.t > self.in_burst_until and (self.rng.random() < self.burst_p):
            self.in_burst_until = self.t + self.burst_len
        in_burst = self.t <= self.in_burst_until

        std = self.noise_std * (self.burst_mult if in_burst else 1.0)
        x_f += self.rng.normal(0.0, std, size=x_f.shape[0])

        drop = min(0.99, self.dropout_p * (2.0 if in_burst else 1.0))
        x_f[self.rng.random(x_f.shape[0]) < drop] = 0.0
        self.t += 1
        return x_f


def compute_delay(y_true, y_pred, onset):
    for t in range(onset, len(y_true)):
        if y_true[t] == 1 and y_pred[t] == 1: return t - onset
    return None


def get_topk(grad, k): return np.argsort(np.abs(grad.ravel()))[-k:]


# =========================
# EXPERIMENT RUNNER
# =========================
def run_sensitivity_seed(X, y, onset_idx, seed, tau_low, tau_high):
    # Re-import for parallel workers
    import torch
    import numpy as np

    device = torch.device("cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    injector = TelemetryDegradation(rng)
    scaler = StandardScaler()
    model = DynamicMLP(X.shape[1]).to(device)

    # Warmup
    warm_slice = min(WARMUP_STEPS, len(X))
    WX = [injector.apply(X[i]) for i in range(warm_slice)]
    scaler.fit(WX)

    # Pre-transform warmup data
    wx_np = scaler.transform(WX)
    wy_np = y[:warm_slice].reshape(-1, 1)
    wx_t = torch.tensor(wx_np, dtype=torch.float32).to(device)
    wy_t = torch.tensor(wy_np, dtype=torch.float32).to(device)

    for _ in range(5): model.online_update(wx_t, wy_t)

    y_pred = np.zeros_like(y)

    # Streaming Loop
    for t in range(warm_slice, len(X)):
        x_raw = injector.apply(X[t])
        x_scaled = scaler.transform(x_raw.reshape(1, -1))
        xt = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        yt = torch.tensor([[y[t]]], dtype=torch.float32).to(device)

        proba = model.predict_proba(xt)

        # --- SENSITIVITY TEST LOGIC ---
        # Variable thresholds based on function args
        threshold = 0.5
        if tau_low < proba < tau_high:
            threshold = 0.35

        y_pred[t] = 1 if proba >= threshold else 0
        model.online_update(xt, yt)

        if (y[t] == 1) and (proba < UNCERTAINTY_THRESHOLD):
            target_one = torch.tensor([[1.0]], dtype=torch.float32).to(device)
            grad = model.get_input_gradient(xt, target_one)
            feats = get_topk(grad, TOPK_FEATURES)

            for _ in range(HARDENING_STEPS):
                s = x_raw.copy()
                s[feats] += rng.normal(0, 0.30, size=len(feats))
                s_s = scaler.transform(s.reshape(1, -1))
                step_size = max(0.5, 0.75)
                s_s[0, feats] += (np.sign(grad[0, feats]) * step_size)
                model.online_update(torch.tensor(s_s).float().to(device), target_one, weight=FN_BIAS_WEIGHT)

    return compute_delay(y, y_pred, onset_idx)


# =========================
# MAIN EXECUTION
# =========================
def main():
    print(f"\n{'=' * 60}\n LASA SENSITIVITY ANALYSIS (Section 6.1.4)\n{'=' * 60}")

    if not os.path.exists(DATA_FILENAME):
        print(f"[Error] '{DATA_FILENAME}' not found.")
        sys.exit(1)

    print("Loading Data...")
    df = pd.read_csv(DATA_FILENAME, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    label_col = [c for c in ["Label", "label", "Attack"] if c in df.columns][0]
    y_raw = df[label_col].astype(str).values
    y = np.array([0 if v in ["BENIGN", "Benign", "0"] else 1 for v in y_raw], dtype=int)
    X = df.select_dtypes(include=[np.number]).fillna(0).clip(-CLIP_VALUE, CLIP_VALUE).values

    # Find Main Attack Onset
    segments = []
    in_seg, start = False, 0
    for i, v in enumerate(y):
        if v == 1 and not in_seg:
            in_seg, start = True, i
        elif v == 0 and in_seg:
            segments.append((start, i - 1)); in_seg = False
    if in_seg: segments.append((start, len(y) - 1))

    cands = [s for s in segments if s[0] > WARMUP_STEPS]
    onset_idx = max(cands, key=lambda x: x[1] - x[0])[0] if cands else 0

    end_idx = min(len(y), onset_idx + 2500)
    X, y = X[:end_idx], y[:end_idx]
    print(f"Data Loaded. Length: {len(y)}, Onset: {onset_idx}")

    # Define Sensitivity Grid
    test_grid = [
        (0.2, 0.6), (0.2, 0.7), (0.2, 0.8),
        (0.3, 0.6), (0.3, 0.7), (0.3, 0.8),  # Default (0.3, 0.7)
        (0.4, 0.6), (0.4, 0.7), (0.4, 0.8)
    ]

    print(f"\nRunning Analysis on Regime S3 (Severe)...")
    print("-" * 50)
    print(f"{'Low':<8} | {'High':<8} | {'Mean Delay (Steps)':<20}")
    print("-" * 50)

    final_results = []

    for (low, high) in test_grid:
        delays = Parallel(n_jobs=N_JOBS)(
            delayed(run_sensitivity_seed)(X, y, onset_idx, seed, low, high)
            for seed in SEEDS
        )

        valid = [d for d in delays if d is not None]
        mean_delay = np.mean(valid) if valid else -1

        print(f"{low:<8} | {high:<8} | {mean_delay:.4f}")

        final_results.append({
            "tau_low": low,
            "tau_high": high,
            "mean_delay": float(mean_delay),
            "n_seeds": len(SEEDS)
        })

    print("-" * 50)

    with open(OUTPUT_FILENAME, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[Saved] {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()