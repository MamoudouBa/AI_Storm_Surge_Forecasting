#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
#
#This script was built, optimized, and tuned with help of Gemini
#
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Required for date formatting
from datetime import datetime
import datetime as dt
import tensorflow.keras.backend as K
import traceback
import os

# --- Define GFS Guidance Columns (MUST MATCH TRAINING SCRIPT) ---
# These are the columns used for the N_FUTURE steps
FUTURE_GUIDANCE_COLS = [
    'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust'
]
# -----------------------------------------------------------------

try:
    print("Forcing TensorFlow to use CPU-only to avoid hardware conflict...")
    tf.config.set_visible_devices([], 'GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("CPU forced. Proceeding with prediction script...")
except Exception as e:
    print(f"Warning: Could not force CPU: {e}")

#
# --- 1. Load Data (Unchanged) ---
#
station_name = 'St Petersburg FL' # Corrected station name for context
print("Loading data...")
data = pd.read_csv('st_petersburg_training_gfs_wl_data.csv')
obs = pd.read_csv('observations_st_petersburg_fl.csv')
print("Loading data...")
# Drop any rows with nan
data = data.dropna()

data = data.sort_values(by='timestamp')

try:
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H %M %S')
except ValueError as e:
    try:
        print("First format failed, trying '%Y-%m-%d %H:%M:%S'...")
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    except Exception as e2:
        print(f"Error parsing datetimes: {e2}")
        exit()
data.set_index('timestamp', inplace=True)
df = data.sort_values(by='timestamp')
if df.index.has_duplicates:
    print("Warning: Duplicate timestamps found. Aggregating with mean().")
    df = df.groupby(df.index).mean()
print("Data Head (with correct DatetimeIndex):\n", df.head())
print(f"Data index type: {type(df.index)}")

#
# --- 2. Define Custom DILATE-inspired Loss Function (Unchanged) ---
#
def create_dilate_loss(alpha=0.5):
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
# --- 3. Data Preprocessing Function (MODIFIED for Future Guidance) ---
#
def prepare_data(df_sequence, n_past, n_future, target_col, scaler_features, scaler_target, future_guidance_cols):
    """
    Creates windows (X, y) for a single data sequence, incorporating future guidance in X.
    """
    print(f"  > Preprocessing sequence of length {len(df_sequence)}...")
    
    # All features in the DataFrame (used for scaling and past input)
    past_feature_cols = df_sequence.columns.tolist() 
    
    # --- Use pre-fitted scalers to transform this sequence ---
    scaled_features_df = pd.DataFrame(scaler_features.transform(df_sequence[past_feature_cols]),
                                      columns=past_feature_cols)
    scaled_target = scaler_target.transform(df_sequence[target_col].values.reshape(-1, 1))

    # Extract scaled numpy arrays
    scaled_past_features_np = scaled_features_df[past_feature_cols].values
    scaled_guidance_features_np = scaled_features_df[future_guidance_cols].values 
    
    X, y = [], []
    y_start_timestamps = []
    
    # Get indices for padding logic
    num_features = len(past_feature_cols)
    guidance_indices = [past_feature_cols.index(col) for col in future_guidance_cols]

    # Windowing logic must account for the full sequence length (N_PAST + N_FUTURE)
    for i in range(n_past, len(df_sequence) - n_future + 1):
        
        # 1. Past Input: [t - N_PAST, t - 1]. Shape: (N_PAST, N_ALL_FEATURES)
        past_input = scaled_past_features_np[i - n_past:i, :]
        
        # 2. Future Guidance Data: [t, t + N_FUTURE - 1]. Shape: (N_FUTURE, N_GUIDANCE_FEATURES)
        future_guidance_data = scaled_guidance_features_np[i:i + n_future, :]
        
        # 3. Create zero-padded matrix for Future Input (N_FUTURE, N_ALL_FEATURES)
        future_input_padded = np.zeros((n_future, num_features), dtype=np.float32) 
        
        # 4. Fill the GFS columns in the padded matrix
        for idx_col, guidance_col_idx in enumerate(guidance_indices):
            future_input_padded[:, guidance_col_idx] = future_guidance_data[:, idx_col]
        
        # 5. Concatenate Past and Future sequences along the TIME AXIS (axis=0)
        # Final shape: (N_PAST + N_FUTURE, N_ALL_FEATURES)
        X_sequence = np.concatenate([past_input, future_input_padded], axis=0) 
        
        X.append(X_sequence)
        y.append(scaled_target[i:i + n_future, 0])
        y_start_timestamps.append(df_sequence.index[i])

    X = np.array(X, dtype=np.float32) # Ensure X is float32 for model
    y = np.array(y, dtype=np.float32)
    y_start_times = np.array(y_start_timestamps)

    print(f"    ...Created {X.shape[0]} windows with final input shape {X.shape[1:]}.")

    return X, y, y_start_times


# --- Main Execution ---
if __name__ == '__main__':
    # Configuration -
    N_PAST_HOURS = 24
    N_FUTURE_HOURS = 12       # Must match the model
    TARGET_COLUMN = 'water_level'
    SAMPLE_INDEX = 0          # Which sample from the test set to predict

    # =========================================================================
    # --- NEW SECTION: Grouping and Preprocessing ---
    # =========================================================================

    # --- 1. Fit Scalers ONCE on all data ---
    print("Fitting scalers on entire dataset...")
    features = df.columns
    target = df[TARGET_COLUMN]
    target_col_index = df.columns.get_loc(TARGET_COLUMN)
    
    # Check for required columns
    if not all(col in features for col in FUTURE_GUIDANCE_COLS):
        missing = [col for col in FUTURE_GUIDANCE_COLS if col not in features]
        print(f"Error: Missing GFS guidance columns in test data: {missing}. Exiting.")
        exit()

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_features.fit(df[features])

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(target.values.reshape(-1, 1))
    print("Scalers fitted.")

    # --- 2. Group Data into Sequences ---
    print("Grouping data into sequences based on 1-hour gaps...")
    df_with_col = df.reset_index()
    time_diff = df_with_col['timestamp'].diff()
    sequence_id = (time_diff > pd.Timedelta('1 hour')).cumsum()
    groups = df_with_col.groupby(sequence_id)
    print(f"Found {len(groups)} total sequences.")

    # --- 3. Create Windows for each valid sequence ---
    all_X, all_y, all_y_start_times = [], [], []
    MIN_SEQ_LEN = N_PAST_HOURS + N_FUTURE_HOURS

    print(f"Windowing sequences (min length {MIN_SEQ_LEN} required)...")

    for seq_id, group_df in groups:
        if len(group_df) >= MIN_SEQ_LEN:
            group_df_indexed = group_df.set_index('timestamp')

            # Pass the guidance column list to the modified function
            X_seq, y_seq, y_times_seq = prepare_data(
                group_df_indexed,
                N_PAST_HOURS,
                N_FUTURE_HOURS,
                TARGET_COLUMN,
                scaler_features,
                scaler_target,
                FUTURE_GUIDANCE_COLS # NEW: Pass Future Guidance Columns
            )

            if X_seq.shape[0] > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)
                all_y_start_times.append(y_times_seq)
            else:
                print(f"  > Skipping sequence {seq_id} (length {len(group_df)} < {MIN_SEQ_LEN})")

    # --- 4. Stack all windows into one dataset ---
    if not all_X:
        print("\n--- CRITICAL ERROR ---")
        print(f"No valid sequences found with the minimum required length of {MIN_SEQ_LEN}.")
        exit()

    X = np.vstack(all_X)
    y = np.vstack(all_y)
    y_start_times = np.concatenate(all_y_start_times)

    print("\n--- Preprocessing Complete ---")
    print(f"Final training data shapes:")
    print(f"  X shape: {X.shape}") # X.shape[1] should now be 36 (24+12)
    print(f"  y shape: {y.shape}")
    print(f"  y_start_times shape: {y_start_times.shape}")

    # =========================================================================
    # --- END OF NEW SECTION ---
    # =========================================================================

    # 2. Split Data 
    X_train, X_test, y_train, y_test, y_times_train, y_times_test = train_test_split(
        X, y, y_start_times, test_size=0.2, random_state=42, shuffle=False
    )
    # Ensure X_test has the correct sequence length (N_PAST + N_FUTURE)
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}, Sequence Length: {X_test.shape[1]}")


    #
    # --- 3. Get Timestamps for Plotting (Adjusted to use the full sequence length in logic) ---
    #
    print("\n--- Getting corrected timestamps ---")
    
    # We need the last N_PAST hours from the original data, and the N_FUTURE hours (the target)
    if SAMPLE_INDEX < 0:
        # Calculate index into the final combined X array
        sample_full_index = len(X) + SAMPLE_INDEX
    else:
        # Calculate index for X_test (which starts after X_train)
        sample_full_index = len(X_train) + SAMPLE_INDEX 
    
    # Find the corresponding starting time in the full sequence array
    forecast_start_time = y_start_times[sample_full_index]
    print(f"Correct forecast start time for sample {SAMPLE_INDEX} is: {forecast_start_time}")
    
    try:
        forecast_start_index_in_df = df.index.get_loc(forecast_start_time)
    except KeyError:
        print(f"CRITICAL ERROR: The timestamp {forecast_start_time} was not found.")
        exit()

    # The history must span N_PAST hours before the forecast starts
    history_start_index_in_df = forecast_start_index_in_df - N_PAST_HOURS
    history_datetimes = df.index[history_start_index_in_df : forecast_start_index_in_df]
    
    # The forecast must span N_FUTURE hours starting at forecast_start_index_in_df
    forecast_datetimes = df.index[forecast_start_index_in_df : forecast_start_index_in_df + N_FUTURE_HOURS]

    if len(history_datetimes) != N_PAST_HOURS or len(forecast_datetimes) != N_FUTURE_HOURS:
        print("Error: Could not retrieve valid history or forecast timestamps.")
        exit()

    print(f"History timestamps from: {history_datetimes[0]} to {history_datetimes[-1]}")
    print(f"Forecast timestamps from: {forecast_datetimes[0]} to {forecast_datetimes[-1]}")
    
    #
    # --- 4. Load Pre-Trained Model (Unchanged) ---
    #
    print("\n--- Loading model from disk... ---")
    # Updated model names to reflect the guidance training scheme
    model_json_saved = f'ft_myers_fl_2010_2021_gru_multi_steps_{N_PAST_HOURS}past_plus_{N_FUTURE_HOURS}future_guidance_OPTIMAL.json'
    weights_file = f'ft_myers_fl_2010_2021_gru_multi_steps_{N_PAST_HOURS}past_plus_{N_FUTURE_HOURS}future_guidance_OPTIMAL.h5'

    dilate_loss_fn = create_dilate_loss(alpha=0.5)
    try:
        json_file = open(model_json_saved, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, custom_objects={'dilate_loss': dilate_loss_fn})
        loaded_model.load_weights(weights_file)
        print("Loaded model from disk")
        loaded_model.compile(optimizer='adam', loss=dilate_loss_fn)
    except FileNotFoundError:
        print(f"Error: Model files not found: {model_json_saved}, {weights_file}")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()
    
    #
    # --- 5. Make and Interpret Predictions (MODIFIED INPUT SHAPE) ---
    #
    print(f"\n--- Making prediction for sample {SAMPLE_INDEX} ---")
    
    # The input shape must be (1, N_PAST + N_FUTURE, N_FEATURES)
    # X_test already contains the correctly constructed sequence (past data + padded future guidance)
    
    try:
        print("--- Making FUTURE prediction ---")
        # X_test[SAMPLE_INDEX] is the full, 36-hour sequence with guidance
        sample_input = X_test[SAMPLE_INDEX].reshape(1, N_PAST_HOURS + N_FUTURE_HOURS, X_test.shape[2]) 
        
        predicted_scaled = loaded_model.predict(sample_input)
        predicted_water_levels = scaler_target.inverse_transform(predicted_scaled) * 3.28
        
        actual_scaled = y_test[SAMPLE_INDEX].reshape(1, N_FUTURE_HOURS)
        actual_water_levels = scaler_target.inverse_transform(actual_scaled) * 3.28
        
        print(f"\nForecast for the next {N_FUTURE_HOURS} hours (starting {forecast_datetimes[0]}):")
        for i in range(N_FUTURE_HOURS):
            print(f"  - {forecast_datetimes[i]}: Predicted={predicted_water_levels[0][i]:.2f}ft, Actual={actual_water_levels[0][i]:.2f}ft")
    except IndexError:
        print(f"Error: Sample index {SAMPLE_INDEX} is out of bounds for the test set (size {len(X_test)}).")
        exit()
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc()
        exit()

    # The rest of the plotting logic remains the same, as it relies on X_test and the generated arrays.
    
    #
    # --- 6. Plot the Sample Prediction (MODIFIED FOR COLOR SEGMENTS - REMAINS UNCHANGED) ---
    #
    print("\n--- Plotting sample prediction with history ---")
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")

    plot_filename = f'FINAL_PLOT_idx{SAMPLE_INDEX}_{timestamp_str}_gru_st_petersburg_guidance.png'

    try:
        # --- GET HISTORY DATA ---
        # History is the first N_PAST steps of the input sequence
        sample_input_scaled_features = X_test[SAMPLE_INDEX][:N_PAST_HOURS, :] 
        
        sample_input_UNSCALED_features = scaler_features.inverse_transform(sample_input_scaled_features) * 3.28
        target_col_index = df.columns.get_loc(TARGET_COLUMN)
        history_levels = sample_input_UNSCALED_features[:, target_col_index]

        # --- "Close the gap" fix for ACTUAL history ---
        forecast_start_actual_level = actual_water_levels[0][0]
        plot_history_datetimes = history_datetimes.union([forecast_datetimes[0]])
        plot_history_levels = np.append(history_levels, forecast_start_actual_level)

        # Skip the complex "past hindcast" logic from the original script as it often fails
        can_predict_past = False # Resetting this to False for cleaner output

        # --- Y-Axis Limits ---
        all_y_data_list = [
            plot_history_levels,
            actual_water_levels[0],
            predicted_water_levels[0]
        ]
        
        all_y_data = np.concatenate(all_y_data_list)

        y_min = np.nanmin(all_y_data)
        y_max = np.nanmax(all_y_data)
        y_padding = (y_max - y_min) * 0.1
        if y_padding == 0: y_padding = 0.5
        plot_y_min = y_min - y_padding
        plot_y_max = y_max + y_padding

        print(f"Forcing Y-Axis limits from {plot_y_min:.2f} to {plot_y_max:.2f}")

        # --- Plotting ---
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot the Actual History
        ax.plot(plot_history_datetimes,
                plot_history_levels,
                marker='o', markersize=4, linestyle='--',
                label='Actual History (Model Input)', color='blue')

        # Plot the Actual Future (Ground Truth)
        ax.plot(forecast_datetimes,
                actual_water_levels[0],
                marker='o', markersize=4, linestyle='--',
                label='Actual Future (Ground Truth)', color='cyan')

        # Plot the Predicted Future (Model Output)
        ax.plot(forecast_datetimes,
                predicted_water_levels[0],
                marker='x', markersize=6, linestyle='-',
                label='Predicted (Model Output)', color='red')

        # Vertical line for forecast start
        ax.axvline(forecast_datetimes[0], color='red', linestyle='--', label='Forecast Start')

        # --- Formatting ---
        all_datetimes = plot_history_datetimes.union(forecast_datetimes)
        ax.set_xlim(all_datetimes[0] - dt.timedelta(hours=1),
                     all_datetimes[-1] + dt.timedelta(hours=1))

        plot_ticks = pd.date_range(start=all_datetimes[0], end=all_datetimes[-1], freq='4h')
        ax.set_xticks(plot_ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()

        ax.set_title(f'Water Level Forecast with GFS Guidance (Station: {station_name}) - {timestamp_str}')
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Water Level (feet)')
        ax.set_ylim(plot_y_min, plot_y_max)
        ax.legend()
        ax.grid(True)

        fig.savefig(plot_filename)

        print(f"\n********************************************************")
        print(f"Plot saved as a NEW FILE: '{plot_filename}'")
        print(f"********************************************************")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        traceback.print_exc()

