#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
########################################################################
#Script developed with the assistance of the Gemini AI code assistant. #
########################################################################
import os, sys, numpy as np, pandas as pd, tensorflow as tf, joblib
# 🌟 FORCE KERAS 2 DESERIALIZER (Leaves NumPy 1.26.4 untouched)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision

# --- 1. Environment & Policy ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Dynamic GPU Memory Growth Engaged.")
    except RuntimeError as e:
        print(f"⚠️ GPU Growth config failed: {e}")

CONFIG_FILE = '../storm_surge/stations_config_gulf_atlantic.csv'
if not os.path.exists(CONFIG_FILE):
    print(f"❌ Master configuration file missing at: {CONFIG_FILE}")
    sys.exit(1)

df_config = pd.read_csv(CONFIG_FILE, dtype={'station_id': str})

# --- Constants ---
MODELS_PATH = '../storm_surge/'
ALPHA = 0.50
N_FUTURE = 12       # 🌟 SYNCED: Set to 24-hour block window
GAP_THRESHOLD = 24.0

# --- Custom Loss Function ---
def create_aggressive_weighted_loss(alpha):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        weights = tf.pow(tf.abs(y_true), 3) + 1.0
        mse = K.mean(weights * K.square(y_true - y_pred), axis=-1)
        y_true_diff = y_true[:, 1:] - y_true[:, :-1]
        y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        shape_loss = K.mean(K.square(y_true_diff - y_pred_diff), axis=-1)
        return alpha * mse + (1.0 - alpha) * shape_loss
    return loss

# --- 3. Operational Ingestion Engine (5-Meteo Direct Mode) ---
def create_operational_hybrid_sequences(group_df, n_past, n_future, target_col, scaler_p, obs_met_cols, gfs_met_cols):
    X, y = [], []
    for i in range(len(group_df) - (n_past + n_future) + 1):
        window = group_df.iloc[i : i + n_past + n_future]

        # 🌪️ WEATHER TRACK
        past_met = window.iloc[:n_past][obs_met_cols].values
        future_met = window.iloc[n_past:][gfs_met_cols].values
        hybrid_met = np.vstack([past_met, future_met])

        # 🌊 WATER TRACK (Direct mode: Zero-padded future space)
        past_surge = window.iloc[:n_past][target_col].values.reshape(-1, 1)
        future_surge_zero = np.zeros((n_future, 1))
        hybrid_surge = np.vstack([past_surge, future_surge_zero])

        combined = np.hstack([hybrid_met, hybrid_surge])
        X.append(scaler_p.transform(combined))

        # Absolute target tracking
        actual_future_surge = window.iloc[n_past:][target_col].values
        y.append(actual_future_surge)

    return np.array(X), np.array(y)

# --- 4. Master Station Execution Loop ---
for idx, row in df_config.iterrows():
    STATION_ID = str(row['station_id']).strip()
    STATION_NAME = str(row['station_name']).strip().replace(", ", "_").replace(" ", "_").replace(".", "")

    OBS_DATA_FILE = os.path.join(MODELS_PATH, f'obs_storms_{STATION_ID}_2021_2026.csv')
    GFS_DATA_FILE = os.path.join(MODELS_PATH, f'training_data_{STATION_ID}_2021_2025.csv')

    if not (os.path.exists(OBS_DATA_FILE) and os.path.exists(GFS_DATA_FILE)):
        continue

    print("\n" + "="*80)
    print(f"🏭 OPERATIONAL 5-METEO DIRECT TRAINING FOR: {STATION_NAME} ({STATION_ID})")
    print("="*80)

    try:
        df_obs = pd.read_csv(OBS_DATA_FILE)
        df_obs['dt'] = pd.to_datetime(df_obs['timestamp'], errors='coerce', format='mixed').dt.round('h')
        df_obs = df_obs.dropna(subset=['dt']).set_index('dt').sort_index()

        df_gfs = pd.read_csv(GFS_DATA_FILE)
        df_gfs['dt'] = pd.to_datetime(df_gfs['timestamp_gui'], errors='coerce', format='mixed').dt.round('h')
        df_gfs = df_gfs.dropna(subset=['dt']).set_index('dt').sort_index()

        data = df_gfs.join(df_obs, how='inner', rsuffix='_obs').sort_index()

        raw_numeric_cols = ['water_level', 'etss_tide', 'cycle',
                            'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust',
                            'air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_direction_gui', 'wind_gust_gui']
        for col in raw_numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        obs_weather = ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust']
        gfs_guidance = ['air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_direction_gui', 'wind_gust_gui']

        if 'surge_residual' not in data.columns:
            raw_surge = data['water_level'] - data['etss_tide']
            data['surge_residual'] = pd.Series(raw_surge.values, index=raw_surge.index).rolling(window=12, center=True, min_periods=1).mean()

        # Chronological Operational Stitching Filter
        active_cycle_mask = data['cycle'] == (data.index.hour // 6) * 6
        data = data[active_cycle_mask].copy()

        data = data.ffill().bfill()
        TARGET = 'surge_residual'

    except Exception as e:
        print(f"❌ Setup failure for station {STATION_ID}: {e}")
        continue

    # --- 5. Training Horizon Loops ---
    for n_past in [24, 48]:

        # 🌟 SYNCED: Strict DIRECT naming pattern definitions
        MODEL_NAME = f"{STATION_NAME}_{N_FUTURE}_OPERATIONAL_GFS_DIRECT_BASELINE_ROBUST_{n_past}past_CHRONO"
        print(f"\n🚀 Training Model Configuration Block: {MODEL_NAME}")

        data['storm_block_id'] = (data.index.to_series().diff() > pd.Timedelta(hours=GAP_THRESHOLD)).cumsum()

        scaler_p = RobustScaler().fit(data[obs_weather + [TARGET]].values)
        scaler_t = RobustScaler().fit(data[[TARGET]].values)

        all_X, all_y = [], []
        storm_groups = data.groupby('storm_block_id')

        for name, grp in storm_groups:
            if len(grp) >= (n_past + N_FUTURE):
                X_g, y_g = create_operational_hybrid_sequences(grp, n_past, N_FUTURE, TARGET, scaler_p, obs_weather, gfs_guidance)
                if X_g.size > 0:
                    all_X.append(X_g)
                    all_y.append(y_g)

        if not all_X:
            continue

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        # 🌟 ACTIVE: Robust target tensor scaling conversion
        shape_y = y.shape
        y = scaler_t.transform(y.reshape(-1, 1)).reshape(shape_y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

        model = Sequential([
            GRU(128, activation='tanh', input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
            Dropout(0.2),
            GRU(64, activation='tanh', return_sequences=False),
            Dropout(0.2),
            Dense(N_FUTURE)
        ])

        baseline_loss_fn = create_aggressive_weighted_loss(alpha=ALPHA)
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        model.compile(optimizer=opt, loss=baseline_loss_fn)

        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=350, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop])

        model.save_weights(os.path.join(MODELS_PATH, f'{MODEL_NAME}.h5'))

        # 🌟 SYNCED: Scaler filenames updated with explicit direct tags
        joblib.dump(scaler_p, os.path.join(MODELS_PATH, f'{STATION_NAME}_{N_FUTURE}_scaler_gfs_only_direct_multi_robust_{n_past}h_CHRONO.joblib'))
        joblib.dump(scaler_t, os.path.join(MODELS_PATH, f'{STATION_NAME}_{N_FUTURE}_scaler_gfs_only_direct_target_multi_robust_{n_past}h_CHRONO.joblib'))

        with open(os.path.join(MODELS_PATH, f'{MODEL_NAME}.json'), "w") as f:
            f.write(model.to_json())

print("\n🏁 5-Meteo Chronological Direct Training Completed Cleanly.")
