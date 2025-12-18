#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
# This script was built, optimized, and tuned with help of Gemini
#
# --- Imports ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K

# --- Configuration (Must Match Training) ---
BASE_FEATURES = ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust']
CYCLES = ['00z', '06z', '12z', '18z']
EXPANDED_COLS = [f'{col}_{cycle}' for cycle in CYCLES for col in BASE_FEATURES]
TARGET_COLUMN = 'water_level'
PAST_FEATURE_COLS = EXPANDED_COLS + [TARGET_COLUMN]

# 🛑 DATUM ADJUSTMENT
# Set this to the difference between your GFS/ETSS zero and local MLLW (e.g., 0.5 or -1.2)
DATUM_OFFSET = 0.0

# Force CPU
tf.config.set_visible_devices([], 'GPU')

def create_dilate_loss(alpha=0.3):
    def dilate_loss(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred) + 1e-6, axis=-1)
        y_true_shifted = K.concatenate([K.expand_dims(y_true[:, 0], -1), y_true[:, :-1]], axis=-1)
        y_pred_shifted = K.concatenate([K.expand_dims(y_pred[:, 0], -1), y_pred[:, :-1]], axis=-1)
        shape_loss = K.mean(K.square((y_true - y_true_shifted) - (y_pred - y_pred_shifted)) + 1e-6, axis=-1)
        return alpha * mse + (1.0 - alpha) * shape_loss
    dilate_loss.__name__ = 'dilate_loss'
    return dilate_loss

def load_and_pivot_data(gfs_path, obs_path):
    date_fmt = "%Y-%m-%d %H %M %S"

    # 1. Process GFS (Features in Feet)
    gfs_data = pd.read_csv(gfs_path)
    gfs_data['timestamp'] = pd.to_datetime(gfs_data['timestamp'].str.strip(), format=date_fmt, errors='coerce')
    gfs_df = gfs_data.dropna(subset=['timestamp'])
    gfs_df['timestamp'] = gfs_df['timestamp'].dt.tz_localize(None).dt.round('h')

    # Save raw GFS water level for plot
    gfs_wl_raw = gfs_df[['timestamp', TARGET_COLUMN]].copy()
    gfs_wl_raw[TARGET_COLUMN] = (gfs_wl_raw[TARGET_COLUMN] * -1) + DATUM_OFFSET
    gfs_wl_raw = gfs_wl_raw.drop_duplicates(subset=['timestamp']).set_index('timestamp')

    # Prep Features
    gfs_df[BASE_FEATURES] = gfs_df[BASE_FEATURES] * -1
    gfs_df['cycle_hour'] = gfs_df['timestamp'].dt.hour % 6 * 6
    gfs_df['cycle_prefix'] = gfs_df['cycle_hour'].astype(str).str.zfill(2) + 'z'

    gfs_pivot = gfs_df.pivot_table(index='timestamp', columns='cycle_prefix', values=BASE_FEATURES, aggfunc='first')
    gfs_pivot.columns = [f'{col[0]}_{col[1]}' for col in gfs_pivot.columns]

    full_range = pd.date_range(start=gfs_pivot.index.min(), end=gfs_pivot.index.max(), freq='h')
    gfs_pivot = gfs_pivot.reindex(full_range).ffill(limit=18)
    gfs_wl_series = gfs_wl_raw.reindex(full_range).ffill(limit=18)[TARGET_COLUMN]

    # 2. Process Obs (Meters to Feet)
    obs_data = pd.read_csv(obs_path)
    obs_data['timestamp'] = pd.to_datetime(obs_data['timestamp'].str.strip(), format=date_fmt, errors='coerce')
    obs_df = obs_data.dropna(subset=['timestamp'])
    obs_df['timestamp'] = obs_df['timestamp'].dt.tz_localize(None).dt.round('h')
    obs_df = obs_df[['timestamp', TARGET_COLUMN]].copy()

    # 🛑 CORRECTED: Conversion from Meters to Feet + Offset
    obs_df[TARGET_COLUMN] = (obs_df[TARGET_COLUMN] * 3.28) + DATUM_OFFSET

    obs_df = obs_df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    obs_df = obs_df.reindex(full_range).ffill(limit=12)

    df = gfs_pivot.join(obs_df, how='inner')
    return df, gfs_wl_series

if __name__ == '__main__':
    N_PAST_HOURS = int(input("Enter N_past (24): "))
    N_FUTURE_HOURS = int(input("Enter N_future (12): "))

    GFS_FILE = '../storm_surge/sandy_hook_test_data_gfs_etss_2025.csv'
    OBS_FILE = '../storm_surge/observations_sandy_hook_nj_2022_2025.csv'
    MODEL_NAME = f'sandy_hook_nj_2022_2024_multi_cycle_gru'

    df, gfs_wl_series = load_and_pivot_data(GFS_FILE, OBS_FILE)

    scaler_features = MinMaxScaler().fit(df[PAST_FEATURE_COLS])
    scaler_target = MinMaxScaler().fit(df[[TARGET_COLUMN]])

    last_idx = len(df) - N_FUTURE_HOURS
    group_df = df.iloc[last_idx - N_PAST_HOURS : last_idx + N_FUTURE_HOURS]
    scaled_f = scaler_features.transform(group_df[PAST_FEATURE_COLS])

    past_input = scaled_f[:N_PAST_HOURS, :]
    future_input = np.zeros((N_FUTURE_HOURS, len(PAST_FEATURE_COLS)), dtype=np.float32)
    future_guidance = scaled_f[N_PAST_HOURS:, :]
    guidance_idxs = [PAST_FEATURE_COLS.index(c) for c in EXPANDED_COLS]
    for idx in guidance_idxs:
        future_input[:, idx] = future_guidance[:, idx]

    X_input = np.concatenate([past_input, future_input], axis=0).reshape(1, N_PAST_HOURS + N_FUTURE_HOURS, -1)
    X_input = np.nan_to_num(X_input)

    dilate_loss_fn = create_dilate_loss(alpha=0.3)
    with open(f'../storm_surge/{MODEL_NAME}.json', 'r') as f:
        model = model_from_json(f.read(), custom_objects={'dilate_loss': dilate_loss_fn})
    model.load_weights(f'../storm_surge/{MODEL_NAME}.h5')
    preds = scaler_target.inverse_transform(model.predict(X_input, verbose=0)).flatten()

    # --- Plotting ---
    history_times = df.index[last_idx - N_PAST_HOURS : last_idx]
    forecast_times = df.index[last_idx : last_idx + N_FUTURE_HOURS]

    h1, h2 = history_times[:12], history_times[12:]

    plt.figure(figsize=(15, 8))
    plt.plot(h1, df[TARGET_COLUMN].loc[h1], color='blue', label='Obs History (H-24 to H-12)')
    plt.plot(h2, df[TARGET_COLUMN].loc[h2], color='cyan', label='Obs History (H-12 to H-0)')
    plt.plot(h1, gfs_wl_series.loc[h1], color='orange', linestyle=':', label='ETSS History (H-24 to H-12)')
    plt.plot(h2, gfs_wl_series.loc[h2], color='gold', linestyle=':', label='ETSS History (H-12 to H-0)')
    plt.plot(forecast_times, df[TARGET_COLUMN].loc[forecast_times], color='black', linestyle='--', label='Actual (Obs Future)')
    plt.plot(forecast_times, gfs_wl_series.loc[forecast_times], color='dodgerblue', linestyle='--', label='ETSS Future Guidance')
    plt.plot(forecast_times, preds, color='red', marker='d', markersize=8, label=f'GRU Prediction (Offset: {DATUM_OFFSET}ft)')

    plt.axvline(forecast_times[0], color='green', linewidth=2, label='Forecast Start')
    plt.title(f"Segmented Storm Surge Forecast (MLLW Corrected)\nModel: {MODEL_NAME}")
    plt.ylabel("Water Level (feet)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(f"../storm_surge/offset_prediction_{time.strftime('%Y%m%d_%H%M%S')}.png")
    print("✅ Prediction complete. The offset has been applied to both observations and predictions.")
