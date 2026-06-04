# ⚡ Electricity Peak Demand Prediction System

A dual-granularity machine learning framework that predicts electricity peak demand at both **daily** and **hourly** levels, deployed as an interactive Streamlit dashboard. Built on the AEP transmission zone load series merged with multi-city meteorological data (Oct 2012 – Nov 2017).

> **Published Research** · *Peak Electricity Demand Prediction Using Dual-Granularity Random Forest Classification: A Case Study on the AEP Transmission Zone* · Mayank, Aishwarya Shelke, Dr. Seema Shukla · Sharda University

---

## Features

- **Daily Peak-Day Prediction** — calibrated probability score for each calendar day (ROC-AUC: 0.907, F1: 0.457)
- **Hourly Peak-Hour Localization** — identifies which specific hours within a day are most likely to spike (ROC-AUC: 0.910)
- **Demand-Based Cost Risk Index** — per-hour economic risk score derived from load ratio
- **Interactive Streamlit Dashboard** — date selector, load curve, peak-hour alerts, feature importance chart
- **Grid Stress Indicators** — color-coded High / Moderate / Normal risk classification

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML Models | `scikit-learn` RandomForestClassifier (n=200, balanced class weights) |
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `altair` |
| Dashboard | `streamlit 1.54.0` |
| Model Serialization | `joblib` |

---

## Project Structure

```
electricity-peak-demand/
│
├── app.py                          # Streamlit dashboard
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── AEP_hourly.csv          # AEP transmission zone load series
│   │   ├── temperature.csv         # Multi-city hourly temperature (36 cities)
│   │   └── humidity.csv            # Multi-city hourly humidity (36 cities)
│   │
│   └── processed/
│       ├── daily_data.csv          # 1,887 daily observations with features + labels
│       ├── hourly_data.csv         # 45,248 hourly observations
│       ├── daily_predictions.csv   # Daily data + Peak_Day_Prob scores
│       └── top_peak_hours.csv      # Per-day peak hour candidates with probabilities
│
├── models/
│   ├── peak_day_model.pkl          # Trained daily RF classifier
│   └── peak_hour_model.pkl         # Trained hourly RF classifier
│
└── notebooks/
    └── model_analysis.ipynb        # EDA, training, evaluation
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/xianceor/electricity-peak-demand.git
cd electricity-peak-demand
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Use the sidebar date picker to explore any date in the dataset range (Oct 2012 – Nov 2017).

---

## Dataset

| Source | Description | Records |
|---|---|---|
| AEP Hourly Load Series | PJM Interconnection — AEP zone net load (MW) | 121,273 hourly obs. |
| Temperature | Multi-city hourly temperature (K), 36 locations | 45,253 records/city |
| Humidity | Multi-city relative humidity (%), 36 locations | 45,253 records/city |

**Processed subset used:** Oct 2012 – Nov 2017 · 1,887 daily obs. · 45,248 hourly obs.

> AEP hourly data is publicly available via the [PJM Data Miner portal](https://www.pjm.com/markets-and-operations/metered-data).

---

## Methodology

### Peak Labeling

- **Peak Day** — day whose `Daily_Max_Load` ≥ 95th percentile of the full distribution → 62 positive cases (3.29%), class ratio 30.4:1
- **Peak Hour** — the hour with maximum load on each calendar day → ~6.9% positive rate

### Models

**Random Forest (Primary)** — 200 trees, `class_weight='balanced'`, decision threshold optimized at 0.15 via F1 maximization on the precision-recall curve.

**Baselines** — Logistic Regression (ℓ2, balanced) and Decision Tree (CART, balanced).

### Features

| Feature | Level | Rationale |
|---|---|---|
| Daily_Max_Load | Daily | Direct demand ceiling |
| Max_Temperature | Daily | AC load spikes above ~295 K |
| Avg_Humidity | Daily | Heat-index amplification |
| Month | Both | Seasonal demand pattern |
| Year | Daily | Load growth trend (2012–2017) |
| IsWeekend | Both | Commercial load drops 15–25% on weekends |
| Hour | Hourly | Dominant intra-day feature (importance: 0.399) |
| Load (hourly) | Hourly | Strongest hourly discriminator (importance: 0.240) |

---

## Results

### Daily Peak-Day Model (Test set: n=378, 12 peak-day instances)

| Model | Accuracy | Precision† | Recall† | F1† | ROC-AUC | Threshold |
|---|---|---|---|---|---|---|
| **Random Forest (balanced)** | **0.950** | **0.348** | **0.667** | **0.457** | **0.907** | **0.15** |
| Gradient Boosting | 0.958 | 0.300 | 0.250 | 0.273 | 0.903 | 0.15 |
| Logistic Regression (balanced) | 0.770 | 0.113 | 0.917 | 0.202 | 0.915 | 0.50 |
| Decision Tree (balanced) | 0.952 | 0.286 | 0.333 | 0.308 | 0.653 | 0.50 |
| Majority Baseline | 0.968 | 0.000 | 0.000 | 0.000 | — | — |

† Metrics for the positive (peak day) class.

### Hourly Peak-Hour Model (Test set: n=9,050)

| Metric | Value |
|---|---|
| ROC-AUC | 0.9104 |
| Accuracy | 0.9581 |
| Mean Peak_Hour_Prob (top hours) | 0.6924 |
| Max Peak_Hour_Prob | 0.9624 |

### Confusion Matrix (Daily Model @ threshold=0.15)

```
                Predicted Non-Peak   Predicted Peak
Actual Non-Peak      351                  15
Actual Peak            4                   8
```

15 false positives = low-cost precautionary reserve activations.  
4 false negatives = missed peaks requiring costly emergency response.

---

## Dashboard Walkthrough

1. **Date Selector** — sidebar date input constrained to dataset range
2. **Peak Day Risk Panel** — `Peak_Day_Prob` with High (>0.60) / Moderate (>0.30) / Normal classification
3. **Grid Stress Indicator** — mirrors risk classification for operator readability
4. **Load Curve** — hourly load chart for the selected date
5. **Demand-Based Cost Risk Index** — `C_h = 1 + 0.6 × (L_h / L̄ − 1)` per hour
6. **Top Peak Hours** — top-2 predicted peak hours with probabilities
7. **Full-Day Peak Hour Probability Chart** — live inference via `peak_hour_model.pkl`; critical alerts at P > 0.70
8. **Feature Importance Chart** — MDI scores from the hourly Random Forest

---

## Limitations

- Models are trained on a single transmission zone (AEP); retraining required for other regions
- Test set contains only 12 peak-day instances — F1 estimates are sensitive to individual errors
- Random Forest does not model temporal autocorrelation; multi-day extreme events are represented only by current-day features
- Cost Risk Index coefficient (0.6) is a simplification of real demand tariff structures

---

## Future Work

- Add lagged load features (prior-day max, 7-day rolling mean) to improve recall on multi-day heat waves
- Apply SMOTE oversampling before training (expected +8–14% F1 improvement per literature)
- Explore Temporal Fusion Transformer for explicit sequential conditioning
- Connect to live grid telemetry API for real-time demand-response alerts

---

## Citation

If you use this work, please cite:

```
Mayank, A. Shelke, and S. Shukla, "Peak Electricity Demand Prediction Using
Dual-Granularity Random Forest Classification: A Case Study on the AEP
Transmission Zone," Sharda University, Greater Noida, India.
```

---

## License

MIT License · Feel free to use, modify, and distribute with attribution.
