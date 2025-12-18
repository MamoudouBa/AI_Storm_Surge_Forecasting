#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppresses all but fatal errors

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
import logging

# Ensure float32 for consistency
tf.keras.mixed_precision.set_global_policy('float32')
K.set_floatx('float32')
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# --- 1. Data Loading with Gap Bridging ---
def load_and_preprocess_data(gfs_file, obs_file, obs_col, base_feature_cols):
    date_format = "%Y-%m-%d %H %M %S"
    print(f"Loading GFS features from: {gfs_file}")
    gfs_data = pd.read_csv(gfs_file)
    gfs_data.columns = gfs_data.columns.str.strip()
    gfs_data['timestamp'] = pd.to_datetime(gfs_data['timestamp'].str.strip(), format=date_format, errors='coerce')
    gfs_df = gfs_data.dropna(subset=['timestamp'])
    gfs_df['timestamp'] = gfs_df['timestamp'].dt.tz_localize(None).dt.round('h')
    
    gfs_df[base_feature_cols] = gfs_df[base_feature_cols] * -1
    gfs_df['cycle_hour'] = gfs_df['timestamp'].dt.hour % 6 * 6
    gfs_df['cycle_prefix'] = gfs_df['cycle_hour'].astype(str).str.zfill(2) + 'z'
    
    gfs_pivot = gfs_df.pivot_table(index='timestamp', columns='cycle_prefix', values=base_feature_cols, aggfunc='first')
    gfs_pivot.columns = [f'{col[0]}_{col[1]}' for col in gfs_pivot.columns]
    gfs_pivot = gfs_pivot.sort_index().ffill(limit=12)
    gfs_pivot = gfs_pivot.reset_index()

    print(f"Loading ground observations from: {obs_file}")
    obs_data = pd.read_csv(obs_file)
    obs_data.columns = obs_data.columns.str.strip()
    obs_data['timestamp'] = pd.to_datetime(obs_data['timestamp'].str.strip(), format=date_format, errors='coerce')
    obs_df = obs_data.dropna(subset=['timestamp'])
    obs_df['timestamp'] = obs_df['timestamp'].dt.tz_localize(None).dt.round('h')
    obs_df = obs_df[['timestamp', obs_col]].copy()
    obs_df[obs_col] = obs_df[obs_col] * 3.28
    obs_df = obs_df.drop_duplicates(subset=['timestamp'])

    print("Merging dataframes...")
    df = pd.merge(gfs_pivot, obs_df, on='timestamp', how='inner')
    
    full_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='h')
    df = df.set_index('timestamp').reindex(full_range)
    df = df.ffill(limit=6) 
    df = df.dropna(subset=[obs_col]) 
    
    print(f"✅ Success! {len(df)} matching hourly records found.")
    return df

# --- 2. Stable DILATE Loss Function ---
def create_dilate_loss(alpha=0.3):
    def dilate_loss(y_true, y_pred):
        # Adding 1e-6 (epsilon) prevents square roots of zero which cause NaNs
        mse = K.mean(K.square(y_true - y_pred) + 1e-6, axis=-1)
        
        y_true_shifted = K.concatenate([K.expand_dims(y_true[:, 0], -1), y_true[:, :-1]], axis=-1)
        y_pred_shifted = K.concatenate([K.expand_dims(y_pred[:, 0], -1), y_pred[:, :-1]], axis=-1)
        
        y_true_diff = y_true - y_true_shifted
        y_pred_diff = y_pred - y_pred_shifted
        
        shape_loss = K.mean(K.square(y_true_diff - y_pred_diff) + 1e-6, axis=-1)
        return alpha * mse + (1.0 - alpha) * shape_loss
    dilate_loss.__name__ = 'dilate_loss'
    return dilate_loss

def create_sequences_from_group(group_df, n_past, n_future, target_col, scaler_features, scaler_target, past_feature_cols, future_guidance_cols):
    scaled_features_np = scaler_features.transform(group_df[past_feature_cols])
    scaled_target_np = scaler_target.transform(group_df[[target_col]])[:, 0]
    num_features = len(past_feature_cols)
    X, y = [], []
    guidance_indices = [past_feature_cols.index(col) for col in future_guidance_cols]

    for i in range(n_past, len(group_df) - n_future + 1):
        past_input = scaled_features_np[i - n_past:i, :]
        future_input_padded = np.zeros((n_future, num_features), dtype=np.float32)
        future_data = scaled_features_np[i:i + n_future, :]
        for idx in guidance_indices:
            future_input_padded[:, idx] = future_data[:, idx]
        X.append(np.concatenate([past_input, future_input_padded], axis=0))
        y.append(scaled_target_np[i:i + n_future])
    return np.array(X), np.array(y)

# --- 3. Main Logic ---
if __name__ == '__main__':
    N_PAST_HOURS, N_FUTURE_HOURS = 24, 12
    GAP_THRESHOLD_HOURS, MIN_GROUP_SIZE = 24.0, 36

    GFS_FILE = '../storm_surge/sandy_hook_training_data_gfs_etss_2022_2024.csv'
    OBS_FILE = '../storm_surge/observations_sandy_hook_nj_2022_2025.csv'
    TARGET = 'water_level'
    BASE_FEATURES = ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust']
    CYCLES = ['00z', '06z', '12z', '18z']

    df = load_and_preprocess_data(GFS_FILE, OBS_FILE, TARGET, BASE_FEATURES)

    diffs = df.index.to_series().diff()
    group_ids = (diffs > pd.Timedelta(hours=GAP_THRESHOLD_HOURS)).cumsum()
    df['group'] = group_ids
    valid_group_ids = df.groupby('group').size()[df.groupby('group').size() >= MIN_GROUP_SIZE].index
    
    if len(valid_group_ids) == 0:
        print("Error: No data groups found. Exiting."); exit()

    valid_df = df[df['group'].isin(valid_group_ids)].copy()
    EXPANDED_COLS = [f'{col}_{cycle}' for cycle in CYCLES for col in BASE_FEATURES]
    PAST_FEATURE_COLS = EXPANDED_COLS + [TARGET]

    scaler_features = MinMaxScaler().fit(valid_df[PAST_FEATURE_COLS])
    scaler_target = MinMaxScaler().fit(valid_df[[TARGET]])

    all_X, all_y = [], []
    for g_id in valid_group_ids:
        X_g, y_g = create_sequences_from_group(valid_df[valid_df['group'] == g_id], N_PAST_HOURS, N_FUTURE_HOURS, TARGET, scaler_features, scaler_target, PAST_FEATURE_COLS, EXPANDED_COLS)
        if X_g.size > 0: all_X.append(X_g); all_y.append(y_g)

    X, y = np.concatenate(all_X), np.concatenate(all_y)

    # 🛑 NAN PROTECTION: Sanitize arrays before split
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --- Build Model ---
    model = Sequential([
        GRU(64, activation='tanh', input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        Dropout(0.3),
        GRU(96, activation='tanh'),
        Dropout(0.3),
        Dense(N_FUTURE_HOURS)
    ])

    # 🛑 STABILITY FIX: Gradient Clipping (clipnorm) and lower Learning Rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=create_dilate_loss(alpha=0.3))

    print("\n--- Starting Training (Check for numeric Loss values) ---")
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
    #model_json_saved('../storm_surge/sandy_hook_multi_cycle_gru.json')
    #model.save_weights('../storm_surge/sandy_hook_multi_cycle_gru.h5')

    print("\n--- Saving model to disk... ---")

    MODEL_NAME = f'sandy_hook_nj_2022_2024_multi_cycle_gru'
    model_json_saved = f'../storm_surge/{MODEL_NAME}.json'
    weights_file = f'../storm_surge/{MODEL_NAME}.h5'

    fileout = model.to_json()
    with open(model_json_saved, "w") as json_file:
        json_file.write(fileout)

    model.save_weights(weights_file)

    print(f"Saved model architecture to: {model_json_saved}")
    print(f"Saved model weights to: {weights_file}")

    print("✅ Training complete and NaN avoided!")
