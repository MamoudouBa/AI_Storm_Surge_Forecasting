#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
#
# Import the necessary packages
#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disables all GPUs
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import datetime as dt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

#
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR) # Only print ERROR messages
#Suppress INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Only print ERROR messages
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#
# --- 1. Load Data (MODIFIED FOR TWO STATIONS) ---
#
print("Loading data for two stations...")

# --- Manually parse timestamp with the correct format ---
# The error "2002-01-01 00 00 00" matches "%Y-%m-%d %H %M %S"
date_format = "%Y-%m-%d %H %M %S"

def load_station_data(filepath, suffix):
    """Helper function to load, parse, and suffix a station's data."""
    try:
        # Load the dataset *without* parsing dates or setting index immediately
        data = pd.read_csv(filepath)

        if 'timestamp' not in data.columns:
            print(f"Error: 'timestamp' column not found in {filepath}. Available columns:")
            print(data.columns)
            return None

        # Convert the 'timestamp' column using the specified format
        # errors='coerce' will turn unparseable dates into NaT (which we can drop)
        data['timestamp'] = pd.to_datetime(data['timestamp'], format=date_format, errors='coerce')

        # Drop any rows that had unparseable dates
        data = data.dropna(subset=['timestamp'])

        if data.empty:
            print(f"Error: Data in {filepath} is empty after parsing dates. Check file and format.")
            return None

        # Set the 'timestamp' column as the index
        data = data.set_index('timestamp')

        # Sort by timestamp, which is crucial for gap detection
        data_sorted = data.sort_index()

        # Add suffix to all columns (e.g., 'water_level' -> 'water_level_stpete')
        data_suffixed = data_sorted.add_suffix(f'_{suffix}')

        print(f"Successfully loaded and processed {filepath}")
        return data_suffixed

    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred loading {filepath}: {e}")
        return None

# Load both datasets
df_st_petersburg = load_station_data('test_set_2022_st_petersburg_fl_for_water_level.csv', 'stpete')
df_clear_water = load_station_data('test_dataset_2022_ft_myers_fl_for_water_level.csv', 'ftmyers')

# Exit if either dataset failed to load
if df_st_petersburg is None or df_clear_water is None:
    print("Error loading one or more data files. Exiting.")
    exit()

# --- Merge the two DataFrames ---
# 'inner' merge keeps only the timestamps where *both* stations have data.
# This aligns the time series perfectly.
print("\nMerging station data...")
df = pd.merge(df_st_petersburg, df_clear_water, left_index=True, right_index=True, how='inner')

if df.empty:
    print("Error: Merged DataFrame is empty. No common timestamps found between the two files. Exiting.")
    exit()

print("Merged Data Head:\n", df.head())
print(f"Merged data shape: {df.shape}")


#
# --- 2. Define Custom DILATE-inspired Loss Function (MODIFIED FOR MULTI-TARGET) ---
#
def create_dilate_loss(alpha=0.5, n_future=12, n_targets=2):
    """
    Factory function to create the DILATE-inspired loss,
    now adapted for N_TARGETS * N_FUTURE output.
    """
    # Reshape the output back to (batch_size, n_future, n_targets)
    # The original implementation works for (batch_size, n_future)
    # We will compute the loss on the flattened array as a simplification,
    # as the model output is (batch_size, n_future * n_targets)
    
    def dilate_loss(y_true_flat, y_pred_flat):
        # The true and predicted values are flat: (batch_size, n_future * n_targets)
        
        # --- 1. MSE (Intensity Loss) ---
        mse = K.mean(K.square(y_true_flat - y_pred_flat), axis=-1)

        # --- 2. Shape Loss (Difference of Differences) ---
        # The flattening combines the targets, so the shift will mix station data.
        # This is a simplification but keeps the *spirit* of the shape loss on the overall signal.
        
        y_true_shifted = K.concatenate([K.expand_dims(y_true_flat[:, 0], -1), y_true_flat[:, :-1]], axis=-1)
        y_pred_shifted = K.concatenate([K.expand_dims(y_pred_flat[:, 0], -1), y_pred_flat[:, :-1]], axis=-1)

        y_true_diff = y_true_flat - y_true_shifted
        y_pred_diff = y_pred_flat - y_pred_shifted

        shape_loss = K.mean(K.square(y_true_diff - y_pred_diff), axis=-1)

        # --- 3. Combined Loss ---
        loss = alpha * mse + (1.0 - alpha) * shape_loss
        return loss

    dilate_loss.__name__ = 'dilate_loss_multi_target'
    return dilate_loss
#
#

#
# --- 3. Data Preprocessing Function (MODIFIED FOR MULTI-TARGET) ---
#
def create_sequences_from_group(group_df, n_past, n_future, target_cols, scaler_features, scaler_target, feature_cols):
    """
    Creates input/output sequences from a single, continuous group of data.
    Now handles multiple target columns.
    """
    # Transform data using the pre-fitted scalers
    # Note: 'target_cols' are a subset of 'feature_cols'
    scaled_features_all = scaler_features.transform(group_df[feature_cols])
    scaled_target_all = scaler_target.transform(group_df[target_cols])

    # Create a DataFrame for easier column access by name
    scaled_features_df = pd.DataFrame(scaled_features_all, columns=feature_cols, index=group_df.index)

    # Extract just the feature columns (as a numpy array)
    scaled_features_np = scaled_features_df[feature_cols].values
    # Extract the target columns (as a numpy array)
    scaled_target_np = scaled_target_all # Shape: (n_rows, n_targets)

    X, y = [], []

    # We can create (len(group_df) - n_past - n_future + 1) sequences
    for i in range(n_past, len(group_df) - n_future + 1):
        # Input: N_PAST timesteps of all features (both stations)
        X.append(scaled_features_np[i - n_past:i, :])
        
        # Output: N_FUTURE timesteps of *all* target variables, flattened
        # Target shape: (N_FUTURE, N_TARGETS) -> Flatten to (N_FUTURE * N_TARGETS,)
        future_targets = scaled_target_np[i:i + n_future, :]
        y.append(future_targets.flatten()) 

    if not X:
        return np.array([]), np.array([])

    return np.array(X), np.array(y)

# --- Main Execution ---
if __name__ == '__main__':
    #
    # --- 4. Configuration (MODIFIED) ---
    #
    N_PAST_HOURS = 24       # Use the last 24 hours of data
    N_FUTURE_HOURS = 12     # Predict the next 12 hours

    # --- NEW: Define the target columns (St. Petersburg AND Clear Water) ---
    TARGET_COLUMNS = ['water_level_stpete', 'water_level_ftmyers']
    N_TARGETS = len(TARGET_COLUMNS)

    # Check if all target columns exist after merge
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            print(f"Error: Target column '{col}' not in merged DataFrame.")
            print(f"Available columns: {df.columns.to_list()}")
            exit()

    # --- NEW: Grouping Configuration (Unchanged) ---
    GAP_THRESHOLD_HOURS = 3.0 # Gap is defined as > 3 hours
    MIN_GROUP_SIZE = N_PAST_HOURS + N_FUTURE_HOURS


    #
    # --- 5. Prepare Data (MODIFIED for Multi-Target Scaling/Sequencing) ---
    #
    # 5.1. Identify and filter groups (Unchanged)
    print("\n--- Preprocessing Merged Data ---")
    print("Identifying time series groups...")

    diffs = df.index.to_series().diff()
    group_ids = (diffs > pd.Timedelta(hours=GAP_THRESHOLD_HOURS)).cumsum()
    df['group'] = group_ids

    group_sizes = df.groupby('group').size()
    valid_group_ids = group_sizes[group_sizes >= MIN_GROUP_SIZE].index

    print(f"Found {len(group_sizes)} total groups based on >{GAP_THRESHOLD_HOURS}hr gap.")
    print(f"Keeping {len(valid_group_ids)} groups with >= {MIN_GROUP_SIZE} elements (N_PAST + N_FUTURE).")

    if len(valid_group_ids) == 0:
        print(f"Error: No data groups found with at least {MIN_GROUP_SIZE} elements. Exiting.")
        exit() # Exit the script

    valid_df = df[df['group'].isin(valid_group_ids)].copy()

    # 5.2. Fit Scalers (MODIFIED for Multi-Target)
    print("Fitting scalers on all valid data...")

    # All columns *except* the group ID will be used as features
    feature_cols = [col for col in df.columns if col not in ['group']]
    print(f"Model will be trained on {len(feature_cols)} features:")
    print(feature_cols)
    print(f"Model will predict {N_TARGETS} targets: {TARGET_COLUMNS}")


    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    # Fit on the valid data
    scaler_features.fit(valid_df[feature_cols])
    scaler_target.fit(valid_df[TARGET_COLUMNS]) # Target scaler now fits on all target columns

    # 5.3. Create sequences from each group (MODIFIED to pass TARGET_COLUMNS)
    all_X, all_y = [], []
    print(f"Creating sequences for {len(valid_group_ids)} groups...")

    for group_id in valid_group_ids:
        group_df = valid_df[valid_df['group'] == group_id][feature_cols]

        X_group, y_group = create_sequences_from_group(
            group_df,
            N_PAST_HOURS,
            N_FUTURE_HOURS,
            TARGET_COLUMNS, # Passed the list of target columns
            scaler_features,
            scaler_target,
            feature_cols
        )

        if X_group.shape[0] > 0:
            all_X.append(X_group)
            all_y.append(y_group)

    if not all_X:
           print(f"Error: No sequences were generated. Exiting.")
           exit() # Exit the script

    # 5.4. Combine all sequences
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"Total sequences generated: X={X.shape}, y={y.shape}")
    # X.shape will be (n_samples, 24, n_total_features_from_both_stations)
    # y.shape will be (n_samples, 12 * 2 = 24)

    # 5.5. Split data (Unchanged)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    #
    # --- 6. Define and Build Model (MODIFIED OUTPUT LAYER) ---
    #
    print("\n--- Building and Training Model with Optimal Hyperparameters ---")

    n_features = X_train.shape[2]
    OUTPUT_SIZE = N_FUTURE_HOURS * N_TARGETS # 12 * 2 = 24 outputs

    model = Sequential()

    # Layer 1
    model.add(GRU(units=64, activation='relu',
                  input_shape=(N_PAST_HOURS, n_features), 
                  return_sequences=True)) 

    # Layer 2
    model.add(GRU(units=96, activation='relu',
                  return_sequences=True)) 

    # Layer 3
    model.add(GRU(units=96, activation='relu',
                  return_sequences=False)) 

    # Output Layer (MODIFIED: Output is now N_FUTURE_HOURS * N_TARGETS)
    model.add(Dense(OUTPUT_SIZE)) 

    #
    # --- 7. Compile Model (MODIFIED LOSS FUNCTION CREATION) ---
    #

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Pass the required context to the loss function factory
    dilate_loss_fn = create_dilate_loss(alpha=0.5, n_future=N_FUTURE_HOURS, n_targets=N_TARGETS)
    model.compile(optimizer=optimizer, loss=dilate_loss_fn)
    model.summary()

    # --- 8. CONFIGURE CALLBACKS (Unchanged) ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #
    # --- 9. Train Model (Unchanged) ---
    #
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[early_stopping]
    )

    #
    # --- 10. Save Trained Model (MODIFIED) ---
    #
    print("\n--- Saving model to disk... ---")

    model_json_saved = f'gru_multi_target_2stations_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL.json'
    weights_file = f'gru_multi_target_2stations_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL.h5'

    fileout = model.to_json()
    with open(model_json_saved, "w") as json_file:
        json_file.write(fileout)

    model.save_weights(weights_file)

    print(f"Saved model architecture to: {model_json_saved}")
    print(f"Saved model weights to: {weights_file}")
    
    #
