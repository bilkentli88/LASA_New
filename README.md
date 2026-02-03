# Generative Repair: Decoupling Robustness from Latency in Intrusion Detection Systems

![Status](https://img.shields.io/badge/Status-Under_Review-yellow) ![License](https://img.shields.io/badge/License-MIT-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)

This repository contains the official PyTorch implementation of **Latency-Aware Stress Adaptation (LASA)**, a framework designed to resolve the trade-off between robustness and responsiveness in streaming intrusion detection systems.

**Paper:** *Generative Repair: Decoupling Robustness from Latency in Intrusion Detection Systems*
**Status:** Submitted to *Expert Systems with Applications* (2026)

## Overview

Real-time intrusion detection systems operating under degraded telemetry (noise, packet loss, drift) often face a critical failure mode we term **Sparsity-Induced Latency**. Passive robustness mechanisms, such as adversarial training (PGD), stabilize predictions by enforcing conservative behavior near the decision boundary, but this comes at the cost of delayed detection during the critical onset of an attack.

**LASA** shifts the paradigm from passive smoothing to **active repair**. It uses a lightweight, gradient-guided stress synthesis mechanism that activates only during latency-critical windows to strengthen local decision support.

## Key Features

* **Drastically Reduces Latency:** Cuts detection delay by >80% (from ~21 steps to 3.7 steps) under severe noise compared to passive baselines.
* **Gradient-Guided Repair:** Uses model gradients to synthesize targeted "stress cases" online, fixing false negatives without expensive offline retraining.
* **Low Operational Overhead:** Adds negligible computational cost (milliseconds) compared to the operational risk of delayed detection.
* **Proven Robustness:** Validated on the **CIC-IDS2017** benchmark against standard, adversarial (PGD), and stochastic (RSA) baselines.

## Installation

The implementation is self-contained and relies on standard deep learning and data science libraries.

    # Clone the repository
    git clone https://github.com/your-username/LASA.git
    cd LASA

    # Install dependencies
    pip install numpy pandas torch scikit-learn joblib matplotlib

**Requirements:**
* Python 3.8+
* PyTorch >= 1.9
* NumPy, Pandas, Scikit-learn, Joblib, Matplotlib

## Dataset Preparation

This framework is evaluated on the **CIC-IDS2017** dataset (Friday Afternoon DDoS capture).

1.  Download the dataset from the [Canadian Institute for Cybersecurity (CIC)](https://www.unb.ca/cic/datasets/ids-2017.html).
2.  Extract the CSV file corresponding to `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`.
3.  Rename the file to `CICIDS2017_day.csv`.
4.  **Move the CSV file into the `src/` folder.** (The scripts expect the data to be in the same directory).

## Reproduction Instructions

All reproduction scripts are located in the `src/` folder. Each script corresponds to a specific Table or analysis in the paper.

**Note:** Please navigate to the source directory before running the scripts:

    cd src

### 1. Synthetic Benchmarks (Tables 1 & 2)
Evaluates LASA against baselines on a controlled synthetic stream with variable noise severity.

    python reproduce_table1_table2.py

* **Outputs:** `results/table1_multiseed.json`, `results/table2_severity_sweep.json`
* **Plots:** Generates `delay_vs_severity.png` and `f1_vs_severity.png`.

### 2. Mechanistic Ablation (Table 3)
Isolates the contribution of each component (Random Noise vs. Uncertainty vs. Gradient Guidance).

    python reproduce_table3_ablation.py

* **Outputs:** `results/table3_ablation.json`
* **Plots:** Generates `ablation_delay.png` comparing component efficacy.

### 3. Drift Detection / Ramp Attack (Table 4)
Tests the ability to detect incipient "Ramp Attacks" in high-dimensional feature spaces.

    python reproduce_table4_ramp.py

* **Outputs:** `results/table4_ramp.json`
* **Key Insight:** Demonstrates the "Detection Gap" between stochastic noise and gradient-guided repair.

### 4. Real-World Validation (Table 5 - Main Result)
The primary evaluation on the **CIC-IDS2017** dataset, comparing all 5 methods (Standard, PGD, DTR, RSA, LASA) under three noise regimes.

    python reproduce_table5_realworld.py

* **Outputs:** `results/table5_realworld.json`
* **Console:** Prints the full summary table (Delay, F1, Flip Rate) used in the manuscript.

### 5. Sensitivity Analysis (Section 6.1.4)
Verifies that performance is robust to changes in the uncertainty trigger thresholds ($\tau_{lower}, \tau_{upper}$).

    python reproduce_sensitivity_analysis.py

* **Outputs:** `results/sensitivity_analysis.json`
* **Goal:** Confirms that detection delay remains stable (~4.0 steps) across a wide hyperparameter grid.

## Results Summary

**Detection Latency under Severe Degradation (Regime S3 - Real Data):**
*Lower is better.*

| Method | Mean Delay (Steps) | Flip Rate (Stability) | F1-Score |
| :--- | :--- | :--- | :--- |
| **Standard MLP** | 20.90 ± 7.59 | 0.0013 | 0.987 |
| **Adversarial (PGD)** | 8.95 ± 1.77 | 0.0006 | 0.994 |
| **Random Stress (RSA)** | 4.35 ± 1.56 | 0.0050 | 0.985 |
| **LASA (Ours)** | **3.70 ± 1.14** | 0.0052 | 0.985 |

> **Insight:** While PGD offers high stability, it suffers from significant delay (8.95 steps). LASA achieves the fastest response (3.70 steps) while maintaining competitive stability.

## Citation

If you use this code or concepts in your research, please cite our paper:

    @article{lasa2026,
        title={Generative Repair: Decoupling Robustness from Latency in Intrusion Detection Systems},
        author={Your Name and Co-Authors},
        journal={Expert Systems with Applications},
        year={2026},
        note={Under Review}
    }

## License

This project is licensed under the MIT License - see the LICENSE file for details.# LASA_New
Official implementation of Latency-Aware Stress Adaptation (LASA) for robust and responsive intrusion detection.
