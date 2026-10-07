#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
########################################################################
# Script developed with the assistance of the Gemini AI code assistant. #
########################################################################

import os, sys

# 🌟 MUST BE BEFORE IMPORT TENSORFLOW:
# Prioritize system CUDA/cuDNN drivers over broken Conda env binaries
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np, pandas as pd, tensorflow as tf, joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Prevent TF from hogging all GPU VRAM instantly
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Dynamic GPU Memory Growth Engaged.")
    except RuntimeError as e:
        print(f"⚠️ GPU Growth config failed: {e}")

# Load configuration matrix dynamically from filesystem
CONFIG_FILE = '../storm_surge/stations_config.csv'
if not os.path.exists(CONFIG_FILE):
    print(f"❌ Master configuration file missing at: {CONFIG_FILE}")
    sys.exit(1)

df_config = pd.read_csv(CONFIG_FILE, dtype={'station_id': str})

# --- Constants ---
MODELS_PATH = '../storm_surge/'
ALPHA = 0.50
N_FUTURE = 12     # 12-hour chunk operational layout
GAP_THRESHOLD = 24.0
EPOCHS = 50       # Global training epoch budget

# =====================================================================
# 🌟 CUSTOM AGGRESSIVE WEIGHTED LOSS FUNCTION
# =====================================================================
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

# --- 3. Operational Cycle Ingestion Engine ---
def create_operational_hybrid_sequences(group_df, n_past, n_future, target_col, scaler_p, obs_met_cols, gfs_met_cols):
    X, y = [], []
    for i in range(len(group_df) - (n_past + n_future) + 1):
        window = group_df.iloc[i : i + n_past + n_future]
        pivot_value = window.iloc[n_past - 1][target_col]

        # 🌪️ WEATHER TRACK: History uses Obs features; Horizon uses GFS guidance features
        past_met = window.iloc[:n_past][obs_met_cols].values
        future_met = window.iloc[n_past:][gfs_met_cols].values
        hybrid_met = np.vstack([past_met, future_met])

        # 🌊 WATER TRACK: Observed surge for history; pivot value for unobserved horizon
        past_surge = window.iloc[:n_past][target_col].values.reshape(-1, 1)
        future_surge_lkv = np.full((n_future, 1), pivot_value)
        hybrid_surge = np.vstack([past_surge, future_surge_lkv])

        # Form combined feature array (6 met/guidance features + 1 water track = 7 total features)
        combined = np.hstack([hybrid_met, hybrid_surge])
        X.append(scaler_p.transform(combined))

        actual_future_surge = window.iloc[n_past:][target_col].values
        y.append(actual_future_surge - pivot_value)

    return np.array(X), np.array(y)

# --- 4. Master Station Execution Loop ---
for idx, row in df_config.iterrows():
    STATION_ID = str(row['station_id']).strip()
    if STATION_ID == "8570283": continue
    STATION_NAME = str(row['station_name']).strip().replace(", ", "_").replace(" ", "_").replace(".", "")

    OBS_DATA_FILE = os.path.join(MODELS_PATH, f'obs_{STATION_ID}_2021_2026.csv')
    GFS_DATA_FILE = os.path.join(MODELS_PATH, f'training_data_{STATION_ID}_2021_2025.csv')

    if not os.path.exists(GFS_DATA_FILE) or not os.path.exists(OBS_DATA_FILE):
        print(f"⚠️ [Skipping] Necessary file pairs absent for station: {STATION_ID}")
        continue

    print("\n" + "="*80)
    print(f"🏭 OPERATIONAL PIVOT GENERATOR ENGINE FOR: {STATION_NAME} ({STATION_ID})")
    print("="*80)

    try:
        print(f"🔄 Syncing Observation History and GFS Guidance Matrices...")
        df_obs = pd.read_csv(OBS_DATA_FILE, low_memory=False)
        t_obs_col = 'timestamp' if 'timestamp' in df_obs.columns else 'timestamp_gui'
        df_obs['dt'] = pd.to_datetime(df_obs[t_obs_col], errors='coerce', format='mixed').dt.round('h')
        df_obs = df_obs.dropna(subset=['dt']).set_index('dt').sort_index()

        df_gfs = pd.read_csv(GFS_DATA_FILE, low_memory=False)
        t_gfs_col = 'timestamp_gui' if 'timestamp_gui' in df_gfs.columns else 'timestamp'
        df_gfs['dt'] = pd.to_datetime(df_gfs[t_gfs_col], errors='coerce', format='mixed').dt.round('h')
        df_gfs = df_gfs.dropna(subset=['dt']).set_index('dt').sort_index()

        raw_numeric_cols = [
            'water_level', 'etss_tide', 'water_level_etss', 'etss_water_level', 'cycle',
            'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust',
            'air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_direction_gui', 'wind_gust_gui'
        ]
        for col in raw_numeric_cols:
            if col in df_obs.columns:
                df_obs[col] = pd.to_numeric(df_obs[col], errors='coerce')
            if col in df_gfs.columns:
                df_gfs[col] = pd.to_numeric(df_gfs[col], errors='coerce')

        # Join datasets on inner index alignment
        data = df_gfs.join(df_obs, how='inner', rsuffix='_obs').sort_index()

        if data.empty:
            print(f"⚠️ [Skipping] {STATION_NAME} ({STATION_ID}) has no overlapping timestamps.")
            continue

        # Fallback for GFS gust if absent
        if 'wind_gust_gui' not in data.columns and 'wind_speed_gui' in data.columns:
            data['wind_gust_gui'] = data['wind_speed_gui'] * 1.2
        if 'wind_gust' not in data.columns and 'wind_speed' in data.columns:
            data['wind_gust'] = data['wind_speed'] * 1.2

        # --- Compute ETSS Guidance Fields ---
        if 'etss_water_level' in data.columns:
            data['etss_wl_base'] = data['etss_water_level']
        elif 'water_level_etss' in data.columns:
            data['etss_wl_base'] = data['water_level_etss']
        else:
            data['etss_wl_base'] = data['etss_tide']

        raw_etss_surge = data['etss_wl_base'] - data['etss_tide']
        data['etss_surge_guidance'] = pd.Series(raw_etss_surge.values, index=raw_etss_surge.index)\
                                        .rolling(window=12, center=True, min_periods=1).mean()

        if 'surge_residual' not in data.columns:
            data['surge_residual'] = data['water_level'] - data['etss_tide']

        # Clean NaNs across numerical fields
        data = data.ffill().bfill()

        # Operational cycle filtering (if present)
        if 'cycle' in data.columns and not data['cycle'].isna().all():
            active_cycle_mask = (data['cycle'] == (data.index.hour // 6) * 6)
            if active_cycle_mask.sum() > 0:
                data = data[active_cycle_mask].copy()

        obs_weather = ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust', 'etss_surge_guidance']
        gfs_guidance = ['air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_direction_gui', 'wind_gust_gui', 'etss_surge_guidance']
        TARGET = 'surge_residual'

        print(f"📊 Aligned Data Matrix: {data.shape[0]} active rows for {STATION_NAME}")

    except Exception as e:
        print(f"❌ Synchronization setup failure for station {STATION_ID}: {e}")
        continue

    # --- 5. Training Horizon Loops ---
    for n_past in [24, 48]:
        MODEL_NAME = f"{STATION_NAME}_{N_FUTURE}_GRU_GFS_ETSS_PIVOT_SAMPLE_WEIGHTING_ROBUST_{n_past}past_CHRONO"
        print(f"\n🚀 Training Model Configuration Block: {MODEL_NAME}")

        data['storm_block_id'] = (data.index.to_series().diff() > pd.Timedelta(hours=GAP_THRESHOLD)).cumsum()

        # Fit Scalers using 7 total features (6 met/guidance features + 1 water target track)
        scaler_p = RobustScaler().fit(data[obs_weather + [TARGET]].values)
        scaler_t = RobustScaler().fit(data[[TARGET]].values)

        all_X, all_y = [], []
        storm_groups = data.groupby('storm_block_id')

        for name, grp in storm_groups:
            if len(grp) >= (n_past + N_FUTURE):
                X_g, y_g = create_operational_hybrid_sequences(
                    grp, n_past, N_FUTURE, TARGET, scaler_p, obs_weather, gfs_guidance
                )
                if X_g.size > 0:
                    all_X.append(X_g)
                    all_y.append(y_g)

        if not all_X:
            print(f"  ⚠️ Insufficient sequence continuity for lookback {n_past}h. Skipping loop.")
            continue

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        print(f"  📈 Built Sequence Tensor: X={X.shape}, y={y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

        model = Sequential([
            GRU(128, activation='tanh', recurrent_activation='sigmoid', reset_after=False, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
            Dropout(0.2),
            GRU(64, activation='tanh', recurrent_activation='sigmoid', reset_after=False, return_sequences=False),
            Dropout(0.2),
            Dense(N_FUTURE)
        ])

        baseline_loss_fn = create_aggressive_weighted_loss(alpha=ALPHA)
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        model.compile(optimizer=opt, loss=baseline_loss_fn)

        model_save_path = os.path.join(MODELS_PATH, f'{MODEL_NAME}_best.h5')

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        model.save_weights(os.path.join(MODELS_PATH, f'{MODEL_NAME}.h5'))

        s_f_name = f"{STATION_NAME}_gru_scaler_gfs_etss_pivot_sample_weighting_robust_{n_past}h_CHRONO.joblib"
        s_t_name = f"{STATION_NAME}_gru_scaler_gfs_etss_pivot_target_sample_weighting_robust_{n_past}h_CHRONO.joblib"

        joblib.dump(scaler_p, os.path.join(MODELS_PATH, s_f_name))
        joblib.dump(scaler_t, os.path.join(MODELS_PATH, s_t_name))

        with open(os.path.join(MODELS_PATH, f'{MODEL_NAME}.json'), "w") as f:
            f.write(model.to_json())

        print(f"✅ Exported CHRONO model and scalers for {STATION_NAME} ({n_past}h lookback)")

print("\n🏁 Master Chronological Stitched Training Protocol Completed Cleanly.")
