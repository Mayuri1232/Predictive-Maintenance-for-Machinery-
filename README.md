# Predictive Maintenance: Machine Failure Prediction

A two-stage machine learning system that predicts (1) whether an industrial machine will fail, and (2) if so, which type of failure is most likely — built on the AI4I 2020 Predictive Maintenance dataset, with a live Streamlit demo for testing predictions on custom sensor readings.

## Overview

Unplanned machine failure is expensive. This project asks two questions in sequence:

1. **Will this machine fail**, based on its current sensor readings?
2. **If it fails, what type of failure is it** — tool wear, heat dissipation, power failure, overstrain, or random?

Rather than treating this as one big multi-class problem, the two questions are answered by two separate, purpose-built models — mirroring how a real maintenance system would actually be used: you don't ask "what kind of failure?" until you already suspect one is coming.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — 10,000 machines, 3.39% failure rate (339 failures), with sensor readings (air/process temperature, rotational speed, torque, tool wear) and a product quality tier (L/M/H).

## Methodology

### Phase 1 — Binary Classification ("Will it fail?")

| Model | Accuracy | Precision (failure) | Recall (failure) | F1 (failure) |
|---|---|---|---|---|
| Logistic Regression (balanced) | 82.5% | 0.14 | 0.82 | 0.24 |
| Random Forest (weighted) | 98.1% | 0.92 | 0.49 | 0.63 |
| **XGBoost (threshold 0.7)** | **99%** | **0.80** | **0.76** | **0.78** |

XGBoost with `scale_pos_weight` (to counter the 96.6%/3.4% class imbalance) and a manually tuned decision threshold of 0.7 was selected as the final model — it catches 76% of real failures while raising only 13 false alarms out of 2,000 test machines, the strongest precision/recall balance of everything tested.

### Phase 2 — Multi-Class Classification ("Which failure type?")

Using the same cleaned features, a second XGBoost model predicts the specific failure mechanism. Rows with overlapping failure flags (24 out of 10,000, where more than one failure type fired simultaneously) were excluded as genuinely ambiguous ground truth.

- Reliably distinguishes **HDF, OSF, and PWF**
- Cannot detect **TWF** or **RNF** — RNF by design (random failures have no learnable pattern); TWF due to too few real examples (42 total) for any tested approach to learn a distinct signature
- **SMOTE oversampling was tested and rejected**: it improved TWF recall from 0.00 to 0.12, but introduced 30+ false alarms on the much larger "No Failure" class — a poor trade-off for a real maintenance system

### Feature Importance

Rotational speed, Torque, and Tool wear account for ~80% of the final model's decision-making. Rotational speed ranks highest despite showing only a small difference in simple group averages during EDA — it appears to matter primarily through *interaction* with other readings, a pattern only detectable by a model that splits on feature combinations, not simple averages.

## Live Demo (Streamlit)

The `app/` folder contains a Streamlit interface where you can input sensor readings and get a live two-stage prediction:

```bash
cd app
streamlit run app.py
```

The app runs the binary model first; the multi-class model only activates if a failure is predicted, and explicitly excludes "No Failure" from its own prediction — since at that point, we already know from Stage 1 that it's failing.

## Repository Structure

```
predictive-maintenance/
├── README.md
├── notebook/
│   └── predictive_maintenance.ipynb    (full analysis: EDA, both models, feature importance)
├── app/
│   ├── app.py                          (Streamlit prediction interface)
│   ├── binary_failure_model.json
│   ├── failure_type_model.json
│   └── label_encoder.pkl
├── data/
│   └── maintainance.csv
```

## Limitations

- Trained on a single, moderately small dataset (10,000 rows, 339 failures) — real-world deployment would need continuous retraining as more failure data becomes available
- RNF and TWF detection should not be relied upon given current data constraints
- The two models are trained independently; the app's logic (excluding "No Failure" from Stage 2) is a deliberate design choice to prevent contradictory outputs, not a guarantee the two models fully agree in all cases

## Tools Used

Python, pandas, scikit-learn, XGBoost, Streamlit

---

*Built as a self-directed learning project in classification, class imbalance handling, and model deployment.*
