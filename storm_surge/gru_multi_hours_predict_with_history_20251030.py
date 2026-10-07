#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
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

#
# --- 1. Load Data ---
#
print("Loading data...")
# Load the data *without* setting the index first
data = pd.read_csv('202209_27_28_ft_myers.surge_test.csv')

# Manually convert the 'timestamp' column using the exact format
try:
    # This format "%Y-%m-%d %H %M %S" (with spaces) was in your original file.
    # If your data uses colons, change it to: "%Y-%m-%d %H:%M:%S"
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H %M %S')
except ValueError as e:
    try:
        print("First format failed, trying '%Y-%m-%d %H:%M:%S'...")
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    except Exception as e2:
        print(f"Error parsing datetimes: {e2}")
        print("Please check that your timestamp format matches the one in the script.")
        exit()

# Now, set the *correct* DatetimeIndex
data.set_index('timestamp', inplace=True)
df = data.sort_values(by='timestamp')
# --- NEW: Handle potential duplicate timestamps ---
if df.index.has_duplicates:
    print("Warning: Duplicate timestamps found. Aggregating with mean().")
    df = df.groupby(df.index).mean()

print("Data Head (with correct DatetimeIndex):\n", df.head())
print(f"Data index type: {type(df.index)}") # This should now be DatetimeIndex

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
# --- 3. Data Preprocessing Function (MODIFIED) ---
#
def prepare_data(df, n_past, n_future, target_col):
    """Prepares data for the LSTM model."""
    print("Preprocessing data...")
    features = df.columns
    target = df[target_col]

    target_col_index = df.columns.get_loc(target_col)
    print(f"Target column '{target_col}' is at index {target_col_index}")

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(df[features])

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

    X, y = [], []

    # --- NEW: We must also store the timestamps ---
    y_start_timestamps = [] # This will store the start time of each Y (forecast)
    # --- END NEW ---

    for i in range(n_past, len(scaled_features) - n_future + 1):
        X.append(scaled_features[i - n_past:i, 0:df.shape[1]])
        y.append(scaled_target[i:i + n_future, 0])

        # --- NEW: Store the timestamp corresponding to the start of y ---
        # y starts at index i in the scaled data, which corresponds
        # to index i in the *original* dataframe's index.
        y_start_timestamps.append(df.index[i])
        # --- END NEW ---

    X, y = np.array(X), np.array(y)

    # --- NEW: Convert timestamps to a numpy array as well ---
    y_start_times = np.array(y_start_timestamps)
    # --- END NEW ---

    print(f"Data shape: X={X.shape}, y={y.shape}, y_start_times={y_start_times.shape}")

    # Return the target index AND the new timestamp array
    return X, y, y_start_times, scaler_features, scaler_target, target_col_index


# --- Main Execution ---
if __name__ == '__main__':
    # Configuration -
    N_PAST_HOURS = 24
    N_FUTURE_HOURS = 5   # Must match the '5future' model
    TARGET_COLUMN = 'water_level'
    SAMPLE_INDEX = 0      # Which sample from the test set to predict

    # 1. Preprocess Data (MODIFIED)
    X, y, y_start_times, scaler_features, scaler_target, target_col_index = prepare_data(df, N_PAST_HOURS, N_FUTURE_HOURS, TARGET_COLUMN)

    # 2. Split Data (MODIFIED)
    # We must split X, y, and y_start_times together so they stay in sync
    X_train, X_test, y_train, y_test, y_times_train, y_times_test = train_test_split(
        X, y, y_start_times, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}, y_times_test={y_times_test.shape}")

    #
    # --- 3. Get Timestamps for Plotting (RE-WRITTEN) ---
    #
    print("\n--- Getting corrected timestamps ---")

    # Get the *actual* starting timestamp for our chosen sample
    # This is the whole reason for the v6 fix.
    forecast_start_time = y_times_test[SAMPLE_INDEX]

    print(f"Correct forecast start time for sample {SAMPLE_INDEX} is: {forecast_start_time}")

    # Find this exact timestamp in the original dataframe
    try:
        forecast_start_index_in_df = df.index.get_loc(forecast_start_time)
    except KeyError:
        print(f"CRITICAL ERROR: The timestamp {forecast_start_time} from the split array")
        print("was not found in the original dataframe index. This shouldn't be possible.")
        print("Check for data-mangling or duplicate timestamps.")
        exit()

    # Get the 5 future forecast timestamps (from the known-good start)
    forecast_datetimes = df.index[forecast_start_index_in_df : forecast_start_index_in_df + N_FUTURE_HOURS]

    # Get the 24 past history timestamps (from the known-good start)
    history_start_index_in_df = forecast_start_index_in_df - N_PAST_HOURS
    history_datetimes = df.index[history_start_index_in_df : forecast_start_index_in_df]

    # Final sanity check
    if len(history_datetimes) != N_PAST_HOURS or len(forecast_datetimes) != N_FUTURE_HOURS:
        print("Error: Could not retrieve valid history or forecast timestamps even with fix.")
        print(f"  History start/end: {history_datetimes[0]} / {history_datetimes[-1]}")
        print(f"  Forecast start/end: {forecast_datetimes[0]} / {forecast_datetimes[-1]}")
        exit()

    print(f"History timestamps from: {history_datetimes[0]} to {history_datetimes[-1]}")
    print(f"Forecast timestamps from: {forecast_datetimes[0]} to {forecast_datetimes[-1]}")
    print(f"Time continuity check: History ends at {history_datetimes[-1]}. Forecast starts 1 step later at {forecast_datetimes[0]}.")

    #
    # --- 4. Load Pre-Trained Model ---
    #
    print("\n--- Loading model from disk... ---")
    model_json_saved = f'gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model.json'
    weights_file = f'gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model.h5'
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
        print(f"Error: Model files not found. Make sure these files are in the same directory:")
        print(f" - {model_json_saved}")
        print(f" - {weights_file}")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()

    #
    # --- 5. Make and Interpret a Prediction ---
    #
    print(f"\n--- Making prediction for sample {SAMPLE_INDEX} ---")

    if len(X_test) > SAMPLE_INDEX:
        # NOTE: We are plotting X_test[SAMPLE_INDEX] against the
        # timestamps we just derived from y_times_test[SAMPLE_INDEX].
        # They correspond to the same sample, so this is correct.
        sample_input = X_test[SAMPLE_INDEX].reshape(1, N_PAST_HOURS, X_test.shape[2])
        predicted_scaled = loaded_model.predict(sample_input)
        predicted_water_levels = scaler_target.inverse_transform(predicted_scaled)

        actual_scaled = y_test[SAMPLE_INDEX].reshape(1, N_FUTURE_HOURS)
        actual_water_levels = scaler_target.inverse_transform(actual_scaled)

        print(f"Forecast for the next {N_FUTURE_HOURS} hours (starting {forecast_datetimes[0]}):")
        for i in range(N_FUTURE_HOURS):
            print(f"  - {forecast_datetimes[i]}: Predicted={predicted_water_levels[0][i]:.2f}m, Actual={actual_water_levels[0][i]:.2f}m")
    else:
        print(f"Error: Sample index {SAMPLE_INDEX} is out of bounds for the test set (size {len(X_test)}).")
        exit()

    #
    # --- 6. Plot the Sample Prediction (with Past Trend) ---
    #
    print("\n--- Plotting sample prediction with history ---")
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    plot_filename = f'sample_forecast_20220927-28_idx{SAMPLE_INDEX}_with_history_{timestamp_str}.png'

    try:
        # --- GET HISTORY DATA (Correct Scaler Method from v4) ---
        sample_input_scaled_features = X_test[SAMPLE_INDEX]
        sample_input_UNSCALED_features = scaler_features.inverse_transform(sample_input_scaled_features)
        history_levels = sample_input_UNSCALED_features[:, target_col_index]

        # --- PRE-PLOT DEBUG INFO (from v5) ---
        print("\n--- PRE-PLOT DEBUG INFO ---")
        print(f"History X (Times)  | type: {type(history_datetimes)} | len: {len(history_datetimes)}")
        print(f"History Y (Levels) | type: {type(history_levels)} | shape: {history_levels.shape}")
        print(f"History Y (Values) | min: {np.nanmin(history_levels):.2f} | max: {np.nanmax(history_levels):.2f} | has_nan: {np.isnan(history_levels).any()}")

        print(f"Forecast X (Times)| type: {type(forecast_datetimes)} | len: {len(forecast_datetimes)}")
        print(f"Forecast Y (Actual)| type: {type(actual_water_levels[0])} | shape: {actual_water_levels[0].shape} | min: {np.nanmin(actual_water_levels[0]):.2f} | max: {np.nanmax(actual_water_levels[0]):.2f}")
        print(f"Forecast Y (Pred)  | type: {type(predicted_water_levels[0])} | shape: {predicted_water_levels[0].shape} | min: {np.nanmin(predicted_water_levels[0]):.2f} | max: {np.nanmax(predicted_water_levels[0]):.2f}")

        # --- Manually Calculate Y-Axis Limits (from v5) ---
        all_y_data = np.concatenate([
            history_levels,
            actual_water_levels[0],
            predicted_water_levels[0]
        ])

        y_min = np.nanmin(all_y_data)
        y_max = np.nanmax(all_y_data)
        y_padding = (y_max - y_min) * 0.1
        if y_padding == 0:
            y_padding = 0.5

        plot_y_min = y_min - y_padding
        plot_y_max = y_max + y_padding

        print(f"Forcing Y-Axis limits from {plot_y_min:.2f} to {plot_y_max:.2f}")
        print("-----------------------------\n")

        plt.close('all')
        fig, ax = plt.subplots(figsize=(15, 7))
        # Plot the Past History
        ax.plot(history_datetimes, # Now the CORRECT timestamps
                history_levels,    # The CORRECT data
                marker='o', markersize=4, linestyle='--',
                label='Actual History (Model Input)', color='blue')

        # Plot the Future Actuals
        ax.plot(forecast_datetimes, # The CORRECT timestamps
                actual_water_levels[0],
                marker='o', markersize=4, linestyle='--',
                label='Actual Future (Ground Truth)', color='cyan')
        # Plot the Future Predictions
        ax.plot(forecast_datetimes, # Now the CORRECT timestamps
                predicted_water_levels[0],
                marker='x', markersize=6, linestyle='-',
                label='Predicted (Model Output)', color='red')

        # Add vertical line
        ax.axvline(forecast_datetimes[0], color='red', linestyle='--', label='Forecast Start')

        all_datetimes = history_datetimes.union(forecast_datetimes)
        ax.set_xlim(all_datetimes[0] - dt.timedelta(hours=1),
                    all_datetimes[-1] + dt.timedelta(hours=1))
        plot_ticks = pd.date_range(start=all_datetimes[0], end=all_datetimes[-1], freq='4h')
        ax.set_xticks(plot_ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()


        ax.set_title(f'Water Level Forecast (Sample {SAMPLE_INDEX})')
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Water Level (m)')

        # Apply the forced Y-Axis limits
        ax.set_ylim(plot_y_min, plot_y_max)

        ax.legend()
        ax.grid(True)
        fig.savefig(plot_filename)

        print(f"Plot saved as '{plot_filename}'")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        print("Check the PRE-PLOT DEBUG INFO above for shape mismatches or NaN values.")

