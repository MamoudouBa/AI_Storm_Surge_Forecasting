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
CONFIG_FILE = '../storm_surge/stations_config_gulf_atlantic.csv'
if not os.path.exists(CONFIG_FILE):
    print(f"❌ Master configuration file missing at: {CONFIG_FILE}")
    sys.exit(1)

df_config = pd.read_csv(CONFIG_FILE, dtype={'station_id': str})

# --- Constants ---
MODELS_PATH = '../storm_surge/'
ALPHA = 0.50
N_FUTURE =  102     # Horizon window target set to 12 hours operational layout
GAP_THRESHOLD = 24.0

# =====================================================================
# 🌟 CUSTOM AGGRESSIVE WEIGHTED LOSS FUNCTION (BASELINE)
# =====================================================================
def create_aggressive_weighted_loss(alpha):
    """
    Your original baseline loss function combining cubic true-value weighting
    with a first-derivative temporal slope penalty.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Cubic weight emphasis on large surge peaks
        weights = tf.pow(tf.abs(y_true), 3) + 1.0
        mse = K.mean(weights * K.square(y_true - y_pred), axis=-1)

        # First-derivative slope matching penalty
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

        # 🌪️ WEATHER TRACK: History leverages Obs features; Horizon leverages GFS forecasts features
        past_met = window.iloc[:n_past][obs_met_cols].values
        future_met = window.iloc[n_past:][gfs_met_cols].values
        hybrid_met = np.vstack([past_met, future_met])

        # 🌊 WATER TRACK: Known observations for history; locked pivot values for unobserved forecast space
        past_surge = window.iloc[:n_past][target_col].values.reshape(-1, 1)
        future_surge_lkv = np.full((n_future, 1), pivot_value)
        hybrid_surge = np.vstack([past_surge, future_surge_lkv])

        # Form feature arrays of width 5 (4 weather features + 1 water track feature)
        combined = np.hstack([hybrid_met, hybrid_surge])
        X.append(scaler_p.transform(combined))

        actual_future_surge = window.iloc[n_past:][target_col].values
        y.append(actual_future_surge - pivot_value)

    return np.array(X), np.array(y)

# --- 4. Master Station Execution Loop ---
for idx, row in df_config.iterrows():
    STATION_ID = str(row['station_id']).strip()
    if STATION_ID == "8766072": continue
    STATION_NAME = str(row['station_name']).strip().replace(", ", "_").replace(" ", "_").replace(".", "")

    OBS_DATA_FILE = os.path.join(MODELS_PATH, f'obs_{STATION_ID}_2020_2026.csv')
    GFS_DATA_FILE = os.path.join(MODELS_PATH, f'training_data_{STATION_ID}_2021_2025.csv')

# 🌟 FIXED: Use 'not' instead of '!' and combine for efficiency
    if not os.path.exists(GFS_DATA_FILE) or not os.path.exists(OBS_DATA_FILE):
        print(f"⚠️ [Skipping] Necessary file pairs absent for station: {STATION_ID}")
        continue

    print("\n" + "="*80)
    print(f"🏭 OPERATIONAL PIVOT GENERATOR ENGINE FOR: {STATION_NAME} ({STATION_ID})")
    print("="*80)

    try:
        print(f"🔄 Syncing Observation History and GFS Guidance Matrices...")
        df_obs = pd.read_csv(OBS_DATA_FILE)
        df_obs['dt'] = pd.to_datetime(df_obs['timestamp'], errors='coerce', format='mixed').dt.round('h')
        df_obs = df_obs.dropna(subset=['dt']).set_index('dt').sort_index()

        df_gfs = pd.read_csv(GFS_DATA_FILE)
        df_gfs['dt'] = pd.to_datetime(df_gfs['timestamp_gui'], errors='coerce', format='mixed').dt.round('h')
        df_gfs = df_gfs.dropna(subset=['dt']).set_index('dt').sort_index()

# 1. Compile the inputs via index inner alignment match
        data = df_gfs.join(df_obs, how='inner', rsuffix='_obs').sort_index()

        # 2. 🌟 THE EXIT DOOR: Check if the join resulted in an empty dataframe
        if data.empty:
            print(f"⚠️ [Skipping] {STATION_NAME} ({STATION_ID}) has no overlapping data rows. Moving to next station...")
            continue  # 🚀 This instantly jumps to the next station in the outer loop!

        # 3. Clean numerical targets (Only runs if data is NOT empty)
        raw_numeric_cols = ['water_level', 'etss_tide', 'cycle',
                            'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust',
                            'air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_direction_gui', 'wind_gust_gui']
        for col in raw_numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Symmetric wind vector decomposition (Obs AND GFS Guidance)
        rad_obs = np.radians(data['wind_direction'].values)
        data['obs_wind_u'] = data['wind_speed'].values * np.cos(rad_obs)
        data['obs_wind_v'] = data['wind_speed'].values * np.sin(rad_obs)

        rad_gui = np.radians(data['wind_direction_gui'].values)
        data['wind_u'] = data['wind_speed_gui'].values * np.cos(rad_gui)
        data['wind_v'] = data['wind_speed_gui'].values * np.sin(rad_gui)

        # Map perfectly symmetric feature vectors (Both are exactly 4 elements long)
        obs_weather = ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust']
        gfs_guidance = ['air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_direction_gui', 'wind_gust_gui']
        
        if 'surge_residual' not in data.columns:
            data['surge_residual'] = data['water_level'] - data['etss_tide']

        # =====================================================================
        # 🌟 CRITICAL REFACTOR: CHRONOLOGICAL OPERATIONAL STITCHING FILTER
        # =====================================================================
        # Selects the active forecast cycle covering each valid hour block sequentially:
        # Hour 00-05 -> Cycle 00, Hour 06-11 -> Cycle 06, Hour 12-17 -> Cycle 12, Hour 18-23 -> Cycle 18
        active_cycle_mask = data['cycle'] == (data.index.hour // 6) * 6
        data = data[active_cycle_mask].copy()

        data = data.ffill().bfill()
        TARGET = 'surge_residual'

    except Exception as e:
        print(f"❌ Synchronization setup failure for station {STATION_ID}: {e}")
        continue

    # --- 5. Training Horizon Loops (Cycle Loop Removed) ---
    for n_past in [24, 48]:
        
        # 🌟 UPDATED: Suffix adjusted to denote unified chronological stitching identity
        MODEL_NAME = f'{STATION_NAME}_{N_FUTURE}_OPERATIONAL_GFS_PIVOT_BASELINE_ROBUST_{n_past}past_CHRONO'
        print(f"\n🚀 Training Model Configuration Block: {MODEL_NAME}")

        data['storm_block_id'] = (data.index.to_series().diff() > pd.Timedelta(hours=GAP_THRESHOLD)).cumsum()

        # Fit Scalers using the globally stitched rolling stream parameters
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
            print(f"  ⚠️ Insufficient sequence continuity for lookback {n_past}h. Skipping parameter loop.")
            continue

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
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

        # Commit weight matrix structures and isolated scaling artifacts to targeted files
        model.save_weights(os.path.join(MODELS_PATH, f'{MODEL_NAME}.h5'))

        # 🌟 UPDATED: File naming convention synced to your chronological multi_std properties
        joblib.dump(scaler_p, os.path.join(MODELS_PATH, f'{STATION_NAME}_{N_FUTURE}_scaler_gfs_only_pivot_multi_robust_{n_past}h_CHRONO.joblib'))
        joblib.dump(scaler_t, os.path.join(MODELS_PATH, f'{STATION_NAME}_{N_FUTURE}_scaler_gfs_only_pivot_target_multi_robust_{n_past}h_CHRONO.joblib'))

        with open(os.path.join(MODELS_PATH, f'{MODEL_NAME}.json'), "w") as f:
            f.write(model.to_json())

print("\n🏁 Master Chronological Stitched Training Protocol Completed Cleanly.")
