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
df_st_petersburg = load_station_data('training_set_2010_2021_st_petersburg_for_water_level.csv', 'stpete')
df_clear_water = load_station_data('training_set_2003_2007_clear_water_for_water_level.csv', 'ftmyers')

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
# --- 2. Define Custom DILATE-inspired Loss Function ---
#
def create_dilate_loss(alpha=0.5):
    """
    Factory function to create the DILATE-inspired loss.
    """
    def dilate_loss(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred), axis=-1)

        y_true_shifted = K.concatenate([K.expand_dims(y_true[:, 0], -1), y_true[:, :-1]], axis=-1)
        y_pred_shifted = K.concatenate([K.expand_dims(y_pred[:, 0], -1), y_pred[:, :-1]], axis=-1)

        y_true_diff = y_true - y_true_shifted
        y_pred_diff = y_pred - y_pred_shifted

        shape_loss = K.mean(K.square(y_true_diff - y_pred_diff), axis=-1)

        loss = alpha * mse + (1.0 - alpha) * shape_loss
        return loss

    dilate_loss.__name__ = 'dilate_loss'
    return dilate_loss
#
#

#
# --- 3. Data Preprocessing Function (Unchanged) ---
#
def create_sequences_from_group(group_df, n_past, n_future, target_col, scaler_features, scaler_target, feature_cols):
    """
    Creates input/output sequences from a single, continuous group of data.
    Assumes group_df has enough data (len >= n_past + n_future).
    Uses *pre-fitted* scalers to transform the data.
    """
    # Transform data using the pre-fitted scalers
    # Note: 'target_col' is one of the 'feature_cols'
    scaled_features_all = scaler_features.transform(group_df[feature_cols])
    scaled_target_all = scaler_target.transform(group_df[[target_col]])

    # Create a DataFrame for easier column access by name
    scaled_features_df = pd.DataFrame(scaled_features_all, columns=feature_cols, index=group_df.index)

    # Extract just the feature columns (as a numpy array)
    scaled_features_np = scaled_features_df[feature_cols].values
    # Extract the target column (as a numpy array)
    scaled_target_np = scaled_target_all

    X, y = [], []

    # We can create (len(group_df) - n_past - n_future + 1) sequences
    for i in range(n_past, len(group_df) - n_future + 1):
        X.append(scaled_features_np[i - n_past:i, :])
        y.append(scaled_target_np[i:i + n_future, 0]) # Predicts N_FUTURE steps of the single target

    if not X: # Should not happen if pre-check is correct, but good safety check
        return np.array([]), np.array([])

    return np.array(X), np.array(y)

# --- Main Execution ---
if __name__ == '__main__':
    #
    # --- 4. Configuration (MODIFIED) ---
    #
    N_PAST_HOURS = 24       # Use the last 24 hours of data
    N_FUTURE_HOURS = 12     # Predict the next 12 hours
    
    # --- NEW: Define the target column (St. Petersburg's water level) ---
    # The model will use *all* columns as input features, but only predict this one.
    TARGET_COLUMN = 'water_level_stpete' 
    
    # Check if target column exists after merge
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not in merged DataFrame.")
        print(f"Available columns: {df.columns.to_list()}")
        print("Did you mean 'water_level_ftmyers'?")
        exit()

    # --- NEW: Grouping Configuration ---
    GAP_THRESHOLD_HOURS = 3.0 # Gap is defined as > 3 hours

    # To create one sequence, we need N_PAST + N_FUTURE elements.
    MIN_GROUP_SIZE = N_PAST_HOURS + N_FUTURE_HOURS


    #
    # --- 5. Prepare Data (REWRITTEN - Logic is Unchanged) ---
    #
    # This section remains the same, but now operates on the *merged* DataFrame 'df'
    #
    # 5.1. Identify and filter groups
    print("\n--- Preprocessing Merged Data ---")
    print("Identifying time series groups...")

    # This is no longer needed as the index is correctly formatted on load.
    diffs = df.index.to_series().diff()

    # cumsum() creates a new group ID every time the condition (gap > 3h) is True
    group_ids = (diffs > pd.Timedelta(hours=GAP_THRESHOLD_HOURS)).cumsum()
    df['group'] = group_ids

    group_sizes = df.groupby('group').size()
    valid_group_ids = group_sizes[group_sizes >= MIN_GROUP_SIZE].index

    print(f"Found {len(group_sizes)} total groups based on >{GAP_THRESHOLD_HOURS}hr gap.")
    print(f"Keeping {len(valid_group_ids)} groups with >= {MIN_GROUP_SIZE} elements (N_PAST + N_FUTURE).")

    if len(valid_group_ids) == 0:
        print(f"Error: No data groups found with at least {MIN_GROUP_SIZE} elements. Exiting.")
        exit() # Exit the script

    # Create the final DataFrame with only the valid groups
    valid_df = df[df['group'].isin(valid_group_ids)].copy()

    # 5.2. Fit Scalers
    # We fit scalers on *all* data we intend to use (all valid groups)
    print("Fitting scalers on all valid data...")

    # All columns *except* the group ID will be used as features
    # This now includes columns from *both* stations
    feature_cols = [col for col in df.columns if col not in ['group']]
    print(f"Model will be trained on {len(feature_cols)} features:")
    print(feature_cols)

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    # Fit on the valid data
    scaler_features.fit(valid_df[feature_cols])
    scaler_target.fit(valid_df[[TARGET_COLUMN]]) # Target scaler needs 2D array

    # 5.3. Create sequences from each group
    all_X, all_y = [], []
    print(f"Creating sequences for {len(valid_group_ids)} groups...")

    for group_id in valid_group_ids:
        # Get the DataFrame for this group, dropping the 'group' column
        group_df = valid_df[valid_df['group'] == group_id][feature_cols]

        # create_sequences_from_group will handle the windowing
        X_group, y_group = create_sequences_from_group(
            group_df,
            N_PAST_HOURS,
            N_FUTURE_HOURS,
            TARGET_COLUMN,
            scaler_features,
            scaler_target,
            feature_cols
        )

        # Add the generated sequences (if any) to our lists
        if X_group.shape[0] > 0:
            all_X.append(X_group)
            all_y.append(y_group)

    if not all_X:
         print(f"Error: No sequences were generated. (Groups >= {MIN_GROUP_SIZE} but sequence creation failed?) Exiting.")
         exit() # Exit the script

    # 5.4. Combine all sequences
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"Total sequences generated: X={X.shape}, y={y.shape}")
    # X.shape will be (n_samples, 24, n_total_features_from_both_stations)
    # y.shape will be (n_samples, 12)

    # 5.5. Split data (as before)
    # This splits the *sequences*. shuffle=False keeps chronological order of sequences.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    #
    # --- 6. Define and Build Model (Unchanged) ---
    #
    print("\n--- Building and Training Model with Optimal Hyperparameters ---")

    # Optimal parameters: {'num_layers': 3, 'units_layer_1': 64, 'units_layer_2': 96, 'units_layer_3': 96}

    # Check if X_train shape is valid
    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        print(f"Error: X_train is empty or has 0 timesteps. Shape: {X_train.shape}. Cannot build model. Exiting.")
        exit()
        
    # The model input_shape is now dynamically set by X_train.shape[2]
    # which is the total number of features from *both* stations.
    n_features = X_train.shape[2] 

    model = Sequential()

    # Layer 1
    model.add(GRU(units=64, activation='relu',
                  input_shape=(N_PAST_HOURS, n_features), # n_features is now num_cols_stpete + num_cols_ftmyers
                  return_sequences=True)) # return_sequences=True because next layer is GRU

    # Layer 2
    model.add(GRU(units=96, activation='relu',
                  return_sequences=True)) # return_sequences=True because next layer is GRU

    # Layer 3
    model.add(GRU(units=96, activation='relu',
                  return_sequences=False)) # return_sequences=False because next layer is Dense

    # Output Layer
    model.add(Dense(N_FUTURE_HOURS)) # Predicts 12 steps for the *single* target variable

    #
    # --- 7. Compile Model (Unchanged) ---
    #

    # Optimal parameter: {'learning_rate': 0.001}
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    dilate_loss_fn = create_dilate_loss(alpha=0.5)
    model.compile(optimizer=optimizer, loss=dilate_loss_fn)
    model.summary()

    # --- 8. CONFIGURE CALLBACKS (Unchanged) ---
    # Stop training if 'val_loss' doesn't improve for 10 epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #
    # --- 9. Train Model (Unchanged) ---
    #
    history = model.fit(
        X_train,
        y_train,
        epochs=100,      # Increased epochs, since EarlyStopping will find the best one
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[early_stopping] # Add the early stopping callback
    )

    #
    # --- 10. Save Trained Model (MODIFIED) ---
    #
    print("\n--- Saving model to disk... ---")

    # Updated file names to reflect the new multi-station logic
    model_json_saved = f'gru_st_petersburgs_clear_water_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL.json'
    weights_file = f'gru_st_petersburgs_clear_water_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL.h5'

    fileout = model.to_json()
    with open(model_json_saved, "w") as json_file:
            json_file.write(fileout)

    model.save_weights(weights_file)

    print(f"Saved model architecture to: {model_json_saved}")
    print(f"Saved model weights to: {weights_file}")
