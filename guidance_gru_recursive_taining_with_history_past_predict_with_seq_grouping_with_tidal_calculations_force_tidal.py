#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
# This script was built, optimized, and tuned with help of Gemini
#
# --- Imports ---
import os
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
import matplotlib
# CRITICAL FIX: Force Matplotlib to use a non-interactive backend for file saving
matplotlib.use('Agg')
# All other imports (pyplot, mdates) should be below this
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import traceback

# Set global Keras/TensorFlow data type to float32 for GPU compatibility
tf.keras.mixed_precision.set_global_policy('float32')
K.set_floatx('float32')

# Suppress TensorFlow log messages
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- GLOBAL CONSTANTS ---
OBS_TARGET_COLUMN = 'water_level'
TIDAL_FEATURE_COLS = ['tide_M2', 'tide_S2', 'tide_K1'] 
# -----------------------------------------------------------------

# --- 1. Load Data ---
def load_and_preprocess_data(gfs_file, obs_file, tidal_file, obs_col):
    """
    Loads, merges, prepares data including GFS, OBS, and TIDES.
    """
    # NOTE: date_format is handled by the explicit parsers below
    gfs_feature_name = 'gfs_' + obs_col 

    # --- Helper function for flexible GFS/OBS timestamp parsing ---
    date_format_candidates = ['%Y-%m-%d %H %M %S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H', '%Y-%m-%d %M:%S']
    def parse_timestamps(df, col_name, formats):
        for fmt in formats:
            try:
                df[col_name] = pd.to_datetime(df[col_name], format=fmt, errors='coerce')
                if df[col_name].notna().sum() > len(df) * 0.9: return df 
            except ValueError:
                continue
        df = df.dropna(subset=[col_name])
        return df
        
    # --- Helper function for rigid TIDAL timestamp parsing ---
    def parse_tidal_timestamps(df, col_name):
        df[col_name] = pd.to_datetime(df[col_name], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df = df.dropna(subset=[col_name])
        return df


    # --- 1a. Load GFS Guidance Data (Features) ---
    print(f"Loading GFS features from: {gfs_file}")
    gfs_data = pd.read_csv(gfs_file)
    gfs_df = parse_timestamps(gfs_data, 'timestamp', date_format_candidates).set_index('timestamp').sort_index()

    gfs_water_level_col = obs_col 
    if gfs_water_level_col in gfs_df.columns:
        print(f"*** Applying phase correction (sign inversion) to GFS feature: {gfs_water_level_col} ***")
        gfs_df[gfs_water_level_col] = gfs_df[gfs_water_level_col] * -1
        print(f"*** Renaming GFS water level column to: {gfs_feature_name} ***")
        gfs_df = gfs_df.rename(columns={gfs_water_level_col: gfs_feature_name})
    print(f"GFS data loaded with shape: {gfs_df.shape}")
    
    # --- 1b. Load Ground Observations (Labels) ---
    print(f"Loading ground observations from: {obs_file}")
    obs_data = pd.read_csv(obs_file)
    obs_df = parse_timestamps(obs_data, 'timestamp', date_format_candidates)
    obs_df = obs_df.dropna(subset=['timestamp'])[[obs_col, 'timestamp']].set_index('timestamp').sort_index()
    print(f"Observation data loaded with shape: {obs_df.shape}")
    
    # --- 1c. Load Tidal Data (New Feature) ---
    print(f"Loading tidal features from: {tidal_file}")
    tidal_data = pd.read_csv(tidal_file)
    tidal_df = parse_tidal_timestamps(tidal_data, 'timestamp')
    tidal_df = tidal_df.dropna(subset=['timestamp'])[TIDAL_FEATURE_COLS + ['timestamp']].set_index('timestamp').sort_index()
    print(f"Tidal data loaded with shape: {tidal_df.shape}")

    # *** CRITICAL FIX: FORCE ALL INDICES TO A CLEAN STRING FORMAT AND BACK ***
    COMMON_FORMAT = '%Y-%m-%d %H:%M:%S'
    def normalize_index(df_in):
        df_in.index = pd.to_datetime(df_in.index.strftime(COMMON_FORMAT))
        return df_in
        
    gfs_df = normalize_index(gfs_df)
    obs_df = normalize_index(obs_df)
    tidal_df = normalize_index(tidal_df)
    # *** END CRITICAL FIX ***

    # --- 1d. Merge DataFrames ---
    print("Merging GFS, Tidal, and Observation dataframes...")
    
    # 1. GFS (Input features) + OBS (Labels)
    df_aligned = gfs_df.join(obs_df, how='inner', sort=True).dropna()

    # 2. Positional Merge with Tidal features (must align by row number)
    len_tidal = len(tidal_df)
    
    if len_tidal == 0: raise ValueError("Fatal Error: Tidal data is empty (0 rows). Cannot proceed.")
        
    # Truncate aligned GFS/OBS data to match the length of Tidal data
    df_aligned_truncated = df_aligned.iloc[:len_tidal].copy()
    tidal_df_processed = tidal_df.copy()
        
    # Drop the incompatible indices and align based on position
    df_aligned_truncated = df_aligned_truncated.reset_index(names='timestamp') 
    tidal_df_processed = tidal_df_processed.reset_index(drop=True) 
    
    # Final Positional Merge
    df_combined = pd.concat([df_aligned_truncated, tidal_df_processed], axis=1)
    
    # Re-set the clean, original GFS/OBS timestamp as the final index
    df = df_combined.set_index('timestamp').sort_index()
    
    # Final check: Remove any NaN rows
    df = df.dropna(subset=TIDAL_FEATURE_COLS + [TARGET_COLUMN])

    if df.empty:
        print("Error: Merged DataFrame is empty. Check time alignment and missing data.")
        exit()

    print(f"Final Merged Data Shape: {df.shape}")
    return df

# --- 2. Define Custom DILATE-inspired Loss Function (UNCHANGED) ---
def create_dilate_loss(alpha=0.5):
    """Factory function to create the DILATE-inspired loss."""
    def dilate_loss(y_true, y_pred):
        # Time-series distance (MSE)
        mse = K.mean(K.square(y_true - y_pred), axis=-1)

        # Shape loss (difference between gradients)
        y_true_shifted = K.concatenate([K.expand_dims(y_true[:, 0], -1), y_true[:, :-1]], axis=-1)
        y_pred_shifted = K.concatenate([K.expand_dims(y_pred[:, 0], -1), y_pred[:, :-1]], axis=-1)
        y_true_diff = y_true - y_true_shifted
        y_pred_diff = y_pred - y_pred_shifted
        shape_loss = K.mean(K.square(y_true_diff - y_pred_diff), axis=-1)

        # CRITICAL: Combine MSE (magnitude) and Shape Loss (periodicity)
        loss = alpha * mse + (1.0 - alpha) * shape_loss
        return loss
    dilate_loss.__name__ = 'dilate_loss'
    return dilate_loss

# --- 3. Data Preprocessing Function (UNCHANGED) ---
def create_sequences_from_group(group_df, n_past, n_future, target_col, scaler_features, scaler_target, past_feature_cols, future_guidance_cols):
    """
    Creates input/output sequences using past data and future guidance.
    """

    # 1. Scale All Data
    scaled_features_np = scaler_features.transform(group_df[past_feature_cols])
    scaled_target_np = scaler_target.transform(group_df[[target_col]])[:, 0] 

    num_features = len(past_feature_cols)
    X, y = [], []

    # Get indices for guidance columns (GFS/Tidal variables) within the full feature set
    guidance_indices = [past_feature_cols.index(col) for col in future_guidance_cols]

    # Iterate through the data to create windows
    for i in range(n_past, len(group_df) - n_future + 1):

        # --- PART 1: Past Features (n_past x n_all_features) ---
        past_input = scaled_features_np[i - n_past:i, :] 

        # --- PART 2: Future Guidance Input (n_future x n_all_features) ---

        # Create a zero-padded matrix for the future steps
        future_input_padded = np.zeros((n_future, num_features), dtype=np.float32)

        # Get the guidance data for the future window (t to t + n_future - 1)
        future_data = scaled_features_np[i:i + n_future, :] 

        # Fill the zero-padded matrix ONLY with the scaled GFS/Tidal guidance feature values
        for idx in guidance_indices:
            future_input_padded[:, idx] = future_data[:, idx]

        # --- Combine sequence in time dimension (axis 0) ---
        X_sequence = np.concatenate([past_input, future_input_padded], axis=0)

        X.append(X_sequence)

        # Target Output y (n_future x 1)
        y.append(scaled_target_np[i:i + n_future])

    if not X:
        return np.array([]), np.array([])

    return np.array(X), np.array(y)

# --- Main Execution ---
if __name__ == '__main__':
    #
    # --- 4. Configuration ---
    #
    N_PAST_HOURS = 24
    N_FUTURE_HOURS = 1 # Recursive Model Training (Single Step)

    # --- File/Column Configuration ---
    GFS_GUIDANCE_FILE = '../storm_surge/sandy_hook_training_data_gfs_etss_2022_2024.csv'
    OBSERVATION_FILE = '../storm_surge/observations_st_petersburg_fl_2022_2024.csv'
    TIDAL_FILE = '../storm_surge/tidal_features.csv' 
    TRUE_TARGET_COLUMN = 'water_level'
    TARGET_COLUMN = TRUE_TARGET_COLUMN

    # --- FEATURE NAME ALIGNMENT ---
    GFS_WL_FEATURE = 'gfs_' + TARGET_COLUMN 
    BASE_GFS_MET_COLS = [
        'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust'
    ]

    # --- Derived Feature Lists ---
    FUTURE_GUIDANCE_COLS_TO_USE = BASE_GFS_MET_COLS + [GFS_WL_FEATURE] + TIDAL_FEATURE_COLS
    PAST_FEATURE_COLS = FUTURE_GUIDANCE_COLS_TO_USE + [TARGET_COLUMN] 

    # --- Grouping Configuration ---
    GAP_THRESHOLD_HOURS = 3.0
    MIN_GROUP_SIZE = N_PAST_HOURS + N_FUTURE_HOURS

    #
    # --- 5. Prepare Data ---
    #
    try:
        # Load and align all data using the fixed positional merge logic
        df = load_and_preprocess_data(
            GFS_GUIDANCE_FILE,
            OBSERVATION_FILE,
            TIDAL_FILE,
            TRUE_TARGET_COLUMN
        )
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed during data preparation/loading: {e}")
        traceback.print_exc()
        exit()
        
    print("\n--- Preprocessing Data ---")

    # 5.1. Group and Sequence Preparation
    diffs = df.index.to_series().diff()
    group_ids = (diffs > pd.Timedelta(hours=GAP_THRESHOLD_HOURS)).cumsum()
    df['group'] = group_ids
    group_sizes = df.groupby('group').size()
    MIN_GROUP_SIZE_ACTUAL = N_PAST_HOURS + N_FUTURE_HOURS # 25
    valid_group_ids = group_sizes[group_sizes >= MIN_GROUP_SIZE_ACTUAL].index
    print(f"Keeping {len(valid_group_ids)} groups with >= {MIN_GROUP_SIZE_ACTUAL} elements.")

    if len(valid_group_ids) == 0:
        print(f"Error: No data groups found with at least {MIN_GROUP_SIZE_ACTUAL} elements. Exiting.")
        exit()

    valid_df = df[df['group'].isin(valid_group_ids)].copy()

    # Final check on feature existence
    if not all(col in valid_df.columns for col in PAST_FEATURE_COLS):
        missing = [col for col in PAST_FEATURE_COLS if col not in valid_df.columns]
        print(f"Error: Missing required columns in merged data: {missing}. Exiting.")
        exit()

    # 5.2. Define Features and Fit Scalers
    print("Fitting scalers on all valid data...")
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    scaler_features.fit(valid_df[PAST_FEATURE_COLS])
    scaler_target.fit(valid_df[[TARGET_COLUMN]])

    # 5.3. Create sequences from each group
    all_X, all_y = [], []
    print(f"Creating sequences for {len(valid_group_ids)} groups...")

    for group_id in valid_group_ids:
        group_df = valid_df[valid_df['group'] == group_id]

        X_group, y_group = create_sequences_from_group(
            group_df,
            N_PAST_HOURS,
            N_FUTURE_HOURS,
            TARGET_COLUMN,
            scaler_features,
            scaler_target,
            PAST_FEATURE_COLS, 
            FUTURE_GUIDANCE_COLS_TO_USE
        )

        if X_group.shape[0] > 0:
            all_X.append(X_group)
            all_y.append(y_group)

    if not all_X:
        print(f"Error: No sequences were generated. Exiting.")
        exit()

    # 5.4. Final preparation for training
    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)

    num_features = X.shape[2] 
    total_sequence_length = N_PAST_HOURS + N_FUTURE_HOURS 
    print(f"\nTotal sequences generated: X={X.shape}, y={y.shape}")
    print(f"Sequence Shape: (N_Samples, {total_sequence_length} Timesteps, {num_features} Features)")

    # 5.5. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    
    # --- 6. Define and Build Model ---
    print("\n--- Building and Training Model ---")

    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        print(f"Error: X_train is empty. Shape: {X_train.shape}. Cannot build model. Exiting.")
        exit()

    model = Sequential()
    model.add(GRU(units=64, activation='tanh',
                         input_shape=(total_sequence_length, num_features),
                         return_sequences=True))
    model.add(Dropout(0.4))
    model.add(GRU(units=96, activation='tanh', return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(N_FUTURE_HOURS)) 

    # --- 7. Compile Model ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # CRITICAL FIX 1: Use the DILATE loss function to prioritize shape/periodicity
    dilate_loss_fn = create_dilate_loss(alpha=0.5) 
    model.compile(optimizer=optimizer, loss=dilate_loss_fn) 
    model.summary()

    # --- 8. CONFIGURE CALLBACKS ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # --- 9. Train Model ---
    print("\n--- Starting Training ---")
    history = model.fit(
        X_train,
        y_train,
        epochs=300,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[early_stopping]
    )

    # --- 10. Save Trained Model ---
    print("\n--- Saving model to disk... ---")

    MODEL_NAME = f'sandy_hook_nj_2022_2024_gru_recursive_single_step_WITH_TIDES_{N_PAST_HOURS}past_OBSERVATION_LABELS'
    model_json_saved = f'../storm_surge/{MODEL_NAME}.json'
    weights_file = f'../storm_surge/{MODEL_NAME}.h5'

    fileout = model.to_json()
    with open(model_json_saved, "w") as json_file:
        json_file.write(fileout)

    model.save_weights(weights_file)

    print(f"Saved model architecture to: {model_json_saved}")
    print(f"Saved model weights to: {weights_file}")
