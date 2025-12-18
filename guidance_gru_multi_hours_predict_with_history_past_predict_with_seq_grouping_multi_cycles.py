#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
#
# This script was built, optimized, and tuned with help of Gemini
# FINAL ROBUST VERSION: Fixes ValueError: The feature names should match those that were passed during fit.
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppresses all but fatal errors
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

# --- Configuration ---
BASE_FEATURES = ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust']
CYCLES = ['00z', '06z', '12z', '18z']
EXPANDED_COLS = [f'{col}_{cycle}' for cycle in CYCLES for col in BASE_FEATURES]
TARGET_COLUMN = 'water_level'
PAST_FEATURE_COLS = EXPANDED_COLS + [TARGET_COLUMN]
DATUM_OFFSET = 0.0

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
    gfs_data = pd.read_csv(gfs_path)
    gfs_data['timestamp'] = pd.to_datetime(gfs_data['timestamp'].str.strip(), format=date_fmt, errors='coerce')
    gfs_df = gfs_data.dropna(subset=['timestamp'])
    gfs_df['timestamp'] = gfs_df['timestamp'].dt.tz_localize(None).dt.round('h')

    # 🛑 FIX: Ensure ETSS guidance is extracted and aligned before pivot
    gfs_wl_raw = gfs_df[['timestamp', TARGET_COLUMN]].copy()
    gfs_wl_raw[TARGET_COLUMN] = (gfs_wl_raw[TARGET_COLUMN] * -1) + DATUM_OFFSET
    gfs_wl_raw = gfs_wl_raw.drop_duplicates(subset=['timestamp']).set_index('timestamp')

    gfs_df[BASE_FEATURES] = gfs_df[BASE_FEATURES] * -1
    gfs_df['cycle_hour'] = gfs_df['timestamp'].dt.hour % 6 * 6
    gfs_df['cycle_prefix'] = gfs_df['cycle_hour'].astype(str).str.zfill(2) + 'z'

    gfs_pivot = gfs_df.pivot_table(index='timestamp', columns='cycle_prefix', values=BASE_FEATURES, aggfunc='first')
    gfs_pivot.columns = [f'{col[0]}_{col[1]}' for col in gfs_pivot.columns]

    full_range = pd.date_range(start=gfs_pivot.index.min(), end=gfs_pivot.index.max(), freq='h')
    gfs_pivot = gfs_pivot.reindex(full_range).ffill(limit=18)

    # Reindex guidance to full hourly range for plotting
    gfs_wl_series = gfs_wl_raw.reindex(full_range).ffill(limit=18)[TARGET_COLUMN]

    obs_data = pd.read_csv(obs_path)
    obs_data['timestamp'] = pd.to_datetime(obs_data['timestamp'].str.strip(), format=date_fmt, errors='coerce')
    obs_df = obs_data.dropna(subset=['timestamp'])
    obs_df['timestamp'] = obs_df['timestamp'].dt.tz_localize(None).dt.round('h')
    obs_df = obs_df[['timestamp', TARGET_COLUMN]].copy()
    obs_df[TARGET_COLUMN] = (obs_df[TARGET_COLUMN] * 3.28) + DATUM_OFFSET
    obs_df = obs_df.drop_duplicates(subset=['timestamp']).set_index('timestamp').reindex(full_range).ffill(limit=12)

    df = gfs_pivot.join(obs_df, how='inner')
    return df, gfs_wl_series

if __name__ == '__main__':
    N_PAST_HOURS, N_FUTURE_HOURS = 24, 12
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

    # Load Model & Predict
    dilate_loss_fn = create_dilate_loss(alpha=0.3)
    with open(f'../storm_surge/{MODEL_NAME}.json', 'r') as f:
        model = model_from_json(f.read(), custom_objects={'dilate_loss': dilate_loss_fn})
    model.load_weights(f'../storm_surge/{MODEL_NAME}.h5')
    preds = scaler_target.inverse_transform(model.predict(X_input, verbose=0)).flatten()

    # --- METRICS ---
    actuals = df[TARGET_COLUMN].iloc[last_idx : last_idx + N_FUTURE_HOURS].values
    rmse = np.sqrt(np.mean((preds - actuals)**2))
    peak_err = np.max(preds) - np.max(actuals)
    forecast_times = df.index[last_idx : last_idx + N_FUTURE_HOURS]
    t_err = (forecast_times[np.argmax(preds)] - forecast_times[np.argmax(actuals)]).total_seconds() / 3600

    # --- PLOTTING ---
    history_times = df.index[last_idx - N_PAST_HOURS : last_idx]
    h1_times, h2_times = history_times[:12], history_times[12:]

    plt.figure(figsize=(15, 8))

    # 1. Segmented Obs History
    plt.plot(h1_times, df[TARGET_COLUMN].loc[h1_times], color='blue', label='Obs History (H-24 to H-12)')
    plt.plot(h2_times, df[TARGET_COLUMN].loc[h2_times], color='cyan', label='Obs History (H-12 to H-0)')

    # 2. Segmented ETSS History (NOW EXPLICITLY PLOTTED)
    plt.plot(h1_times, gfs_wl_series.loc[h1_times], color='orange', linestyle=':', label='ETSS History (H-24 to H-12)')
    plt.plot(h2_times, gfs_wl_series.loc[h2_times], color='gold', linestyle=':', label='ETSS History (H-12 to H-0)')

    # 3. Future Comparisons
    plt.plot(forecast_times, actuals, color='black', linestyle='--', label='Actual (Future)')
    plt.plot(forecast_times, gfs_wl_series.loc[forecast_times], color='dodgerblue', linestyle='--', label='ETSS Future Guidance')
    plt.plot(forecast_times, preds, color='red', marker='d', label='GRU Prediction')

    # Stats Textbox
    stats = f"RMSE: {rmse:.3f} ft\nPeak Err: {peak_err:.3f} ft\nTime Err: {t_err:.1f} hr"
    plt.gca().text(0.02, 0.05, stats, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.axvline(forecast_times[0], color='green', label='Forecast Start')
    plt.title(f"Direct Multi-Cycle Forecast: Segmented Tracking\nModel: {MODEL_NAME}")
    plt.ylabel("Water Level (feet)")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = f"../storm_surge/final_segmented_prediction_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out_file)
    print(f"✅ Success! ETSS History plotted. Plot saved: {out_file}")
