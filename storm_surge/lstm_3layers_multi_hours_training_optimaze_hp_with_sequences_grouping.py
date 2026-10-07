#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
#
# Import the necessary packages
#
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import datetime as dt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping # Ensure EarlyStopping is imported
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
# --- 1. Load Data (MODIFIED) ---
#
print("Loading data...")

# --- FIX: Manually parse timestamp with the correct format ---
# The error "2002-01-01 00 00 00" matches "%Y-%m-%d %H %M %S"
date_format = "%Y-%m-%d %H %M %S"

# Load the dataset *without* parsing dates or setting index immediately
try:
    #data = pd.read_csv('surge_training_datasets.csv')
    data = pd.read_csv('training_set_2003_2007_st_petersburg_for_water_level.csv')
    #data = pd.read_csv('test_set_2022_st_petersburg_for_water_level.csv')
    # Drop any rows with nan
    #data = data.dropna()


    # Check if 'timestamp' column exists
    if 'timestamp' not in data.columns:
        print("Error: 'timestamp' column not found in CSV. Available columns:")
        print(data.columns)
        exit()

    # Convert the 'timestamp' column using the specified format
    # errors='coerce' will turn unparseable dates into NaT (which we can drop)
    data['timestamp'] = pd.to_datetime(data['timestamp'], format=date_format, errors='coerce')

    # Drop any rows that had unparseable dates
    data = data.dropna(subset=['timestamp'])

    # Set the 'timestamp' column as the index
    data = data.set_index('timestamp')

    # Sort by timestamp, which is crucial for gap detection
    df = data.sort_index()

    if df.empty:
        print(f"Error: Data is empty after parsing dates. Check CSV file and format string ('{date_format}').")
        exit()

    print("Data Head:\n", df.head())

except FileNotFoundError:
    print("Error: 'surge_training_datasets.csv' not found.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

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
# --- 3. Data Preprocessing Function (NEW) ---
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
    scaled_features_df = pd.DataFrame(scaled_features_all, columns=feature_cols)

    # Extract just the feature columns (as a numpy array)
    scaled_features_np = scaled_features_df[feature_cols].values
    # Extract the target column (as a numpy array)
    scaled_target_np = scaled_target_all

    X, y = [], []

    # We can create (len(group_df) - n_past - n_future + 1) sequences
    for i in range(n_past, len(group_df) - n_future + 1):
        X.append(scaled_features_np[i - n_past:i, :])
        y.append(scaled_target_np[i:i + n_future, 0])

    if not X: # Should not happen if pre-check is correct, but good safety check
        return np.array([]), np.array([])

    return np.array(X), np.array(y)

# --- Main Execution ---
if __name__ == '__main__':
    #
    # --- 4. Configuration ---
    #
    N_PAST_HOURS = 24        # Use the last 24 hours of data
    N_FUTURE_HOURS = 12      # Predict the next 12 hours
    TARGET_COLUMN = 'water_level'

    # --- NEW: Grouping Configuration ---
    GAP_THRESHOLD_HOURS = 3.0 # Gap is defined as > 3 hours

    # To create one sequence, we need N_PAST + N_FUTURE elements.
    # This is the *true* minimum group size, superseding the '20'
    # (e.g., 24 + 12 = 36).
    MIN_GROUP_SIZE = N_PAST_HOURS + N_FUTURE_HOURS


    #
    # --- 5. Prepare Data (REWRITTEN) ---
    #

    # 5.1. Identify and filter groups
    print("\n--- Preprocessing Data ---")
    print("Identifying time series groups...")

    # --- FIX 2: Removed redundant pd.to_datetime(df.index) ---
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
    # This ensures a consistent scaling transformation
    print("Fitting scalers on all valid data...")

    # All columns *except* the group ID will be used as features
    # The original code scaled all columns, so we keep that behavior
    feature_cols = [col for col in df.columns if col not in ['group']]

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

    # 5.5. Split data (as before)
    # This splits the *sequences*. shuffle=False keeps chronological order of sequences.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    #
    # --- 6. Define and Build Model with OPTIMIZED HYPERPARAMETERS ---
    #
    print("\n--- Building and Training Model with Optimal Hyperparameters ---")

    # Optimal parameters: {'learning_rate': 0.0001, 'num_layers': 2, 'dropout_rate': 0.4,
    # 'lstm_activation': 'tanh', 'dilate_alpha': 0.3, 'units_layer_1': 64, 'units_layer_2': 96}

    # Check if X_train shape is valid
    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        print(f"Error: X_train is empty or has 0 timesteps. Shape: {X_train.shape}. Cannot build model. Exiting.")
        exit()

    model = Sequential()

    # Layer 1 (units_layer_1, lstm_activation, dropout_rate)
    model.add(LSTM(units=64, activation='tanh', # units_layer_1, lstm_activation
                    input_shape=(N_PAST_HOURS, X_train.shape[2]),
                    return_sequences=True if 2 > 1 else False)) # num_layers = 2, so return_sequences is True for L1

    model.add(Dropout(0.4)) # dropout_rate

    # Layer 2 (units_layer_2, lstm_activation, dropout_rate)
    model.add(LSTM(units=96, activation='tanh', # units_layer_2, lstm_activation
                    return_sequences=False)) # Last LSTM layer, so return_sequences is False

    model.add(Dropout(0.4)) # dropout_rate

    # Output Layer
    model.add(Dense(N_FUTURE_HOURS))

    #
    # --- 7. Compile Model with OPTIMIZED HYPERPARAMETERS ---
    #

    # Optimal parameter: {'learning_rate': 0.0001, 'dilate_alpha': 0.3}
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    dilate_loss_fn = create_dilate_loss(alpha=0.3) # dilate_alpha
    model.compile(optimizer=optimizer, loss=dilate_loss_fn)
    model.summary()

    # --- 8. CONFIGURE CALLBACKS (Early Stopping Incorporated) ---
    # Stop training if 'val_loss' doesn't improve for 10 epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #
    # --- 9. Train Model ---
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
    # --- 10. Save Trained Model ---
    #
    print("\n--- Saving model to disk... ---")

    # Updated file names to reflect the new grouping logic
    model_json_saved = f'st_petersburg_2003_2007_lstm_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL.json'
    weights_file = f'st_petersburg_2003_2007_lstm_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL.h5'

    fileout = model.to_json()
    with open(model_json_saved, "w") as json_file:
            json_file.write(fileout)

    model.save_weights(weights_file)

    print(f"Saved model architecture to: {model_json_saved}")
    print(f"Saved model weights to: {weights_file}")
