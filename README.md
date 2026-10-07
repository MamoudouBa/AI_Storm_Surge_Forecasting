# Deep Learning Coastal Storm Surge & Total Water Level Forecasting Suite

A deep learning framework designed to produce high-precision, multi-cycle operational forecasts for coastal storm surge and Total Water Level (TWL). 

The system leverages a hybrid approach blending **Global Forecast System (GFS)** meteorological guidance and **Extra Tropical Storm Surge (ETSS)** physics-based baseline predictions with gated recurrent deep architectures (**GRU / CNN / U-Net CHRONO models**).

---

## 🌟 Key Features

* **Hybrid Surrogate Architecture**: Blends physics-based numerical hydrodynamic guidance (ETSS + Astronomical Tide) with real-time station observations and GFS numerical weather forecasts.
* **CHRONO Pivot Rollout Engine**: Uses autoregressive sequential step rollouts (12h lookahead horizon) anchored to dynamic real-time observed water level pivots.
* **Dynamic Inverse-Variance Ensemble Blending**: Automatically blends multi-member lookback horizons (24h and 48h) using rolling variance weighting to maximize stability across storm surge peaks.
* **Custom Aggressive Loss Function**: Combines cubic peak weighting (|y_true|^3 + 1.0) with rate-of-change derivative loss to eliminate peak clipping and phase lag during rapid storm surges.
* **Station Registry Execution**: Scalable pipeline capable of running single-target or multi-station coastal evaluations via registry configurations (`stations_config.csv`).

---

## 📊 Feature Vector Schema

The standard model input schema uses a **6-feature meteorological/guidance set** appended with **1 historical target feedback track** (total vector dimension = 7 columns):

| Feature Name | Description | Source Stream |
| :--- | :--- | :--- |
| `air_pressure` / `air_pressure_gui` | Atmospheric Surface Pressure (mb) | Station Obs / GFS Guidance |
| `air_temp` / `air_temp_gui` | Surface Air Temperature (°C) | Station Obs / GFS Guidance |
| `wind_speed` / `wind_speed_gui` | Surface Wind Speed (m/s or kts) | Station Obs / GFS Guidance |
| `wind_direction` / `wind_direction_gui` | Surface Wind Direction (degrees) | Station Obs / GFS Guidance |
| `wind_gust` / `wind_gust_gui` | Peak Surface Wind Gust (m/s or kts) | Station Obs / GFS Guidance |
| `etss_surge_guidance` | Physics-based Surge Baseline (ft) | ETSS Forecast Stream |
| `surge_residual` *(Target Track)* | Filtered Surge Elevation (ft) | Observed Water Level - Tide Baseline |

---

## 🧠 Model Architectures

1. **CHRONO GRU Models (Primary Operational Engine)**:
   * **Input Shape**: (B, N_past + N_future, 7)
   * **Layers**: Stacked 128-unit and 64-unit Gated Recurrent Units (GRU) with Dropout (0.2) and dense output projections (N_future = 12h).
   * **Scaling**: Robust Scaling fit independently per station on historical storm blocks.

2. **Spatial Verification Baselines (Diurnal Verification)**:
   * **CNN 20-Var / CNN 21-Var / U-Net 20-Var**: Spatial grids used for evaluating spatial relaxation scales (11km, 22km, 33km) across operational lead times (f01 through f08).

---

## 📂 Repository Structure

├── storm_surge/                           # Saved asset & dataset directory
│   ├── stations_config.csv               # Target station configuration matrix
│   ├── obs_8726520_2020_2026.csv          # Station observation datasets
│   ├── test_gfs_etss_data_8726520_2026.csv # Operational GFS/ETSS guidance streams
│   └── *.h5 / *.json / *.joblib          # Saved model weights, structures & scalers
│
├── storm_surge_scripts/                  # Primary execution scripts
│   ├── gru_obs_gfs_etss_storms_training.py  # Model training engine
│   ├── plot_gru_pivot_gfs_etss_stats.py  # Monthly rollout & evaluation plotter
│   └── plot_2021_2022_diurnal_stats.py   # Diurnal spatial skill verification plotter
│
├── validation_plots/                     # Exported monthly rollout figures & cards
└── README.md                             # Project documentation

---

## 🚀 Environment & Setup

### HPC Cluster / Conda Setup
Activate your environment (e.g., `mlaws2t` with Python 3.11, TensorFlow 2.x, NumPy, pandas, SciPy, Matplotlib, and Scikit-learn):

```bash
conda activate mlaws2
