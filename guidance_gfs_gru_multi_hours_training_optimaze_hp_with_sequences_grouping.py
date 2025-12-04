#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
# This script was built, optimized, and tuned with help of Gemini

# Import the necessary packages
import os
# Suppress TensorFlow log messages (set to ERROR level)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all but fatal errors

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

# --- Global Configuration ---
# Set global Keras/TensorFlow data type to float32 for GPU compatibility (Best practice)
tf.keras.mixed_precision.set_global_policy('float32')
K.set_floatx('float32')

# Suppress TensorFlow log messages (set to ERROR level)
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all but fatal errors

# --- 1. Define Custom DILATE-inspired Loss Function ---
def create_dilate_loss(alpha=0.5):
    """
    Factory function to create the DILATE-inspired loss.
    """
    def dilate_loss(y_true, y_pred):
        # Time-series specific loss combining MSE (Value Loss) and Shape Loss
        mse = K.mean(K.square(y_true - y_pred), axis=-1)

        # Calculate shape loss (difference of derivatives)
        y_true_shifted = K.concatenate([K.expand_dims(y_true[:, 0], -1), y_true[:, :-1]], axis=-1)
        y_pred_shifted = K.concatenate([K.expand_dims(y_pred[:, 0], -1), y_pred[:, :-1]], axis=-1)
        y_true_diff = y_true - y_true_shifted
        y_pred_diff = y_pred - y_pred_shifted
        shape_loss = K.mean(K.square(y_true_diff - y_pred_diff), axis=-1)
        
        loss = alpha * mse + (1.0 - alpha) * shape_loss
        return loss
    dilate_loss.__name__ = 'dilate_loss'
    return dilate_loss

# --- 2. Data Preprocessing Function (The Fix) ---
def create_sequences_from_group(group_df, n_past, n_future, target_col, scaler_features, scaler_target, past_feature_cols, future_guidance_cols):
    """
    Creates input/output sequences using past data (all features) and 
    future guidance data (subset of features), concatenating them along 
    the time axis (axis=0). Includes shape verification for robust execution.
    """
    X, y = [], []
    num_features = len(past_feature_cols)

    # 1. Scale All Data - Wrapped in try/except to catch column mismatches early
    try:
        scaled_features_all = scaler_features.transform(group_df[past_feature_cols])
    except ValueError as e:
        print(f"\n--- SCALER ERROR DEBUG ---")
        print(f"Error scaling features: {e}")
        print(f"Expected features (from scaler fit): {list(scaler_features.feature_names_in_)}")
        print(f"Actual features in group_df: {group_df.columns.tolist()}")
        print("--------------------------\n")
        # Re-raise the error to stop execution
        raise ValueError("Feature mismatch during scaling. Check CSV columns.")

    scaled_target_all = scaler_target.transform(group_df[[target_col]])

    # 2. Extract scaled numpy arrays
    scaled_features_df = pd.DataFrame(scaled_features_all, columns=past_feature_cols)
    scaled_past_features_np = scaled_features_df[past_feature_cols].values
    future_guidance_features = scaled_features_df[future_guidance_cols].values
    scaled_target_np = scaled_target_all

    # 3. Iterate through the data to create windows
    # Start at n_past, end at length - n_future (inclusive of the last window)
    for i in range(n_past, len(group_df) - n_future + 1):

        # 3a. Past Features (N_PAST steps, All features)
        past_input = scaled_past_features_np[i - n_past:i, :] # Shape: (N_PAST, N_FEAT)

        # 3b. Future Guidance Features (N_FUTURE steps, Guidance features only)
        future_guidance_input = future_guidance_features[i:i + n_future, :] # Shape: (N_FUTURE, N_GUIDANCE_FEAT)

        # 3c. Padding Logic: Create a full feature set for the future time steps
        future_input_padded = np.zeros((n_future, num_features), dtype=np.float32)
        
        # Find the indices of the GFS guidance columns in the full feature list
        guidance_indices = [past_feature_cols.index(col) for col in future_guidance_cols]
        
        # Fill the GFS columns in the padded matrix
        for idx_col, guidance_col_idx in enumerate(guidance_indices):
            future_input_padded[:, guidance_col_idx] = future_guidance_input[:, idx_col] # Fill the columns with guidance data

        # --- CRITICAL SHAPE CHECK before final concatenation (Axis=1/Features) ---
        if past_input.shape[1] != future_input_padded.shape[1]:
            print(f"\n--- CONCATENATION DIMENSION MISMATCH DEBUG ---")
            print(f"Past Input feature count: {past_input.shape[1]}")
            print(f"Future Padded feature count: {future_input_padded.shape[1]}")
            print(f"Expected feature count: {num_features}")
            print("Feature count mismatch. The padding logic failed.")
            raise ValueError("Feature dimension mismatch before concatenation on axis=0.")

        # 3d. Combine past and future inputs along the TIME AXIS (axis=0)
        X_sequence = np.concatenate([past_input, future_input_padded], axis=0) # Shape: (N_PAST + N_FUTURE, N_FEAT)

        X.append(X_sequence)

        # 3e. Target Output y (N_FUTURE x 1)
        y.append(scaled_target_np[i:i + n_future, 0])

    if not X:
        return np.array([]), np.array([])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# --- Main Execution ---
if __name__ == '__main__':
    
    # --- 3. Configuration ---
    npast = input("Enter N_past ")
    nfuture = input("Enter N_future ")
    N_PAST_HOURS = int(npast)
    N_FUTURE_HOURS = int(nfuture)
    TARGET_COLUMN = 'water_level'
    GAP_THRESHOLD_HOURS = 3.0
    MIN_GROUP_SIZE = N_PAST_HOURS + N_FUTURE_HOURS
    
    # --- Feature Configuration ---
    future_guidance_cols = [
        'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust'
    ]

    # --- 4. Load Data ---
    print("Loading data...")
    date_format = "%Y-%m-%d %H %M %S"

    try:
        data = pd.read_csv('../storm_surge/st_petersburg_training_data_gfs_etss_2022_2023.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'], format=date_format, errors='coerce')
        data = data.dropna(subset=['timestamp'])
        data = data.set_index('timestamp')
        df = data.sort_index()

        if df.empty:
            print(f"Error: Data is empty after parsing dates. Check CSV file and format string ('{date_format}').")
            exit()
        
        # Define all columns available in the dataset (excluding the index)
        all_cols = [col for col in df.columns]
        past_feature_cols = all_cols

        # Check if all required columns are present
        required_cols = [TARGET_COLUMN] + future_guidance_cols
        if not all(col in all_cols for col in required_cols):
            missing = [col for col in required_cols if col not in all_cols]
            print(f"Error: Missing required columns: {missing}. Check CSV headers.")
            exit()

        print("Data loaded successfully.")

    except FileNotFoundError:
        print("Error: '../storm_surge/st_petersburg_training_data_gfs_etss_2022_2023.csv' not found.")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

    # --- 5. Prepare Data Groups ---
    print("\n--- Preprocessing Data ---")
    diffs = df.index.to_series().diff()
    group_ids = (diffs > pd.Timedelta(hours=GAP_THRESHOLD_HOURS)).cumsum()
    df['group'] = group_ids
    group_sizes = df.groupby('group').size()
    valid_group_ids = group_sizes[group_sizes >= MIN_GROUP_SIZE].index
    print(f"Keeping {len(valid_group_ids)} groups with >= {MIN_GROUP_SIZE} elements.")

    if len(valid_group_ids) == 0:
        print(f"Error: No data groups found with at least {MIN_GROUP_SIZE} elements. Exiting.")
        exit()

    valid_df = df[df['group'].isin(valid_group_ids)].copy()

    # --- 6. Define and Fit Scalers ---
    print("Fitting scalers on all valid data...")

    # We must fit the feature scaler on ALL columns that are passed into the model (past_feature_cols)
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    scaler_features.fit(valid_df[past_feature_cols])
    scaler_target.fit(valid_df[[TARGET_COLUMN]])

    # --- 7. Create sequences from each group ---
    all_X, all_y = [], []
    print(f"Creating sequences for {len(valid_group_ids)} groups...")

    for group_id in valid_group_ids:
        group_df = valid_df[valid_df['group'] == group_id][all_cols]

        X_group, y_group = create_sequences_from_group(
            group_df,
            N_PAST_HOURS,
            N_FUTURE_HOURS,
            TARGET_COLUMN,
            scaler_features,
            scaler_target,
            past_feature_cols,
            future_guidance_cols
        )

        if X_group.shape[0] > 0:
            all_X.append(X_group)
            all_y.append(y_group)

    if not all_X:
        print(f"Error: No sequences were generated. Exiting.")
        exit()

    # --- 8. Combine all sequences and prepare for training ---
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print("\n--- Feature lentgh ---", len(X)," Feature lentgh ",len(y))
    # Data types already converted inside create_sequences_from_group
    
    print(f"Total sequences generated: X={X.shape}, y={y.shape}")
    print(f"New Sequence Length: {X.shape[1]} ({N_PAST_HOURS} past + {N_FUTURE_HOURS} future guidance)")
    print(f"Number of Features: {X.shape[2]}")
    print(f"Data types: X={X.dtype}, y={y.dtype}")

    # 9. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"\nTrain shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    # --- 10. Define and Build Model ---
    print("\n--- Building and Training Model ---")

    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        print(f"Error: X_train is empty or has 0 timesteps. Shape: {X_train.shape}. Cannot build model. Exiting.")
        exit()

    model = Sequential()
    sequence_length = N_PAST_HOURS + N_FUTURE_HOURS

    # Input_shape is (Time steps, Features)
    model.add(GRU(units=64, activation='tanh',
                  input_shape=(sequence_length, X_train.shape[2]),
                  return_sequences=True))

    model.add(Dropout(0.4))

    # Second GRU processes the full sequence length (36 steps)
    model.add(GRU(units=96, activation='tanh',
                  return_sequences=False))

    model.add(Dropout(0.4))

    # Output Layer: predicts N_FUTURE_HOURS water levels
    model.add(Dense(N_FUTURE_HOURS))

    # --- 11. Compile Model ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    dilate_loss_fn = create_dilate_loss(alpha=0.3)
    model.compile(optimizer=optimizer, loss=dilate_loss_fn)
    model.summary()

    # --- 12. Train Model ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
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

    # --- 13. Save Trained Model ---
    print("\n--- Saving model to disk... ---")
                        
    model_json_saved = f'../storm_surge/st_petersburg_fl_gru_multi_steps_{N_PAST_HOURS}past_plus_{N_FUTURE_HOURS}future_guidance_OPTIMAL.json'
    weights_file = f'../storm_surge/st_petersburg_fl_gru_multi_steps_{N_PAST_HOURS}past_plus_{N_FUTURE_HOURS}future_guidance_OPTIMAL.h5'

    print('model_json_saved', model_json_saved)
    fileout = model.to_json()
    with open(model_json_saved, "w") as json_file:
        json_file.write(fileout)

    model.save_weights(weights_file)

    print(f"Saved model architecture to: {model_json_saved}")
    print(f"Saved model weights to: {weights_file}")
