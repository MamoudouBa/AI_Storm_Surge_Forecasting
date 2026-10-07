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
import traceback
import os

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
station_name = 'Clear Water Beach'
print("Loading data...")
data = pd.read_csv('test_dataset_2022_ft_myers_fl_for_water_level.csv')
print("Loading data...")
#data = pd.read_csv('test_dataset_2022_ft_myers_fl_for_water_level.csv')
#data = pd.read_csv('training_set_2022_ft_myers_fl_for_water_level.csv')
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
#

#
# --- 3. Data Preprocessing Function (MODIFIED) ---
#
def prepare_data(df_sequence, n_past, n_future, target_col, scaler_features, scaler_target):
    """
    Creates windows (X, y) for a single data sequence using pre-fitted scalers.
    df_sequence is expected to have a DatetimeIndex.
    """
    print(f"  > Preprocessing sequence of length {len(df_sequence)}...")
    features = df_sequence.columns
    target = df_sequence[target_col]

    # --- Use pre-fitted scalers to transform this sequence ---
    scaled_features = scaler_features.transform(df_sequence[features])
    scaled_target = scaler_target.transform(target.values.reshape(-1, 1))

    X, y = [], []
    y_start_timestamps = []

    # Windowing logic is unchanged
    for i in range(n_past, len(scaled_features) - n_future + 1):
        X.append(scaled_features[i - n_past:i, 0:df_sequence.shape[1]])
        y.append(scaled_target[i:i + n_future, 0])
        y_start_timestamps.append(df_sequence.index[i]) # Uses DatetimeIndex

    X, y = np.array(X), np.array(y)
    y_start_times = np.array(y_start_timestamps)

    print(f"    ...Created {X.shape[0]} windows from this sequence.")

    # --- Return only X, y, and times for this sequence ---
    return X, y, y_start_times


# --- Main Execution ---
if __name__ == '__main__':
    # Configuration -
    N_PAST_HOURS = 24
    N_FUTURE_HOURS = 12       # Must match the '12future' model
    TARGET_COLUMN = 'water_level'
    
    # --- MODIFICATION: Set the target prediction start time ---
    TARGET_PREDICT_TIME = datetime(2022, 11, 3, 2, 0, 0)
    SAMPLE_INDEX = 0          # Temporary value, will be calculated later

    # =========================================================================
    # --- NEW SECTION: Grouping and Preprocessing ---
    # =========================================================================

    # --- 1. Fit Scalers ONCE on all data ---
    print("Fitting scalers on entire dataset...")
    features = df.columns
    target = df[TARGET_COLUMN]
    # Get index for plotting later
    target_col_index = df.columns.get_loc(TARGET_COLUMN)

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_features.fit(df[features])

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(target.values.reshape(-1, 1))
    print("Scalers fitted.")

    # --- 2. Group Data into Sequences ---
    print("Grouping data into sequences based on 1-hour gaps...")
    # Reset index to use 'timestamp' as a column for diff()
    df_with_col = df.reset_index()
    time_diff = df_with_col['timestamp'].diff()
    # Create a unique ID for each contiguous block of data
    sequence_id = (time_diff > pd.Timedelta('1 hour')).cumsum()
    groups = df_with_col.groupby(sequence_id)
    print(f"Found {len(groups)} total sequences.")

    # --- 3. Create Windows for each valid sequence ---
    all_X, all_y, all_y_start_times = [], [], []

    # The minimum length MUST be n_past + n_future to create even ONE window
    MIN_SEQ_LEN = N_PAST_HOURS + N_FUTURE_HOURS

    print(f"Windowing sequences (min length {MIN_SEQ_LEN} required)...")

    for seq_id, group_df in groups:
        if len(group_df) >= MIN_SEQ_LEN:
            # prepare_data expects 'timestamp' as index
            group_df_indexed = group_df.set_index('timestamp')

            # Call modified prepare_data
            X_seq, y_seq, y_times_seq = prepare_data(
                group_df_indexed,
                N_PAST_HOURS,
                N_FUTURE_HOURS,
                TARGET_COLUMN,
                scaler_features,
                scaler_target
            )

            # Only append if windows were actually created
            if X_seq.shape[0] > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)
                all_y_start_times.append(y_times_seq)
        else:
            print(f"  > Skipping sequence {seq_id} (length {len(group_df)} < {MIN_SEQ_LEN})")

    # --- 4. Stack all windows into one dataset ---

    # =========================================================================
    # NEW FIX: Check if any windows were created before stacking
    # =========================================================================
    if not all_X:
        print("\n--- CRITICAL ERROR ---")
        print(f"No valid sequences found with the minimum required length of {MIN_SEQ_LEN}.")
        print("The dataset does not contain any continuous data blocks long enough to create training windows.")
        print("Try lowering N_PAST_HOURS or N_FUTURE_HOURS, or check your input data file.")
        exit()
    # =========================================================================

    X = np.vstack(all_X)
    y = np.vstack(all_y)
    y_start_times = np.concatenate(all_y_start_times)

    print("\n--- Preprocessing Complete ---")
    print(f"Final training data shapes:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y_start_times shape: {y_start_times.shape}")

    # =========================================================================
    # --- END OF NEW SECTION ---
    # =========================================================================

    # 2. Split Data (Unchanged logic, but uses new X, y)
    X_train, X_test, y_train, y_test, y_times_train, y_times_test = train_test_split(
        X, y, y_start_times, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}, y_times_test={y_times_test.shape}")

    # =========================================================================
    # 🚨 DYNAMIC SAMPLE_INDEX FINDER 🚨
    # =========================================================================
    print("\n--- Searching for Target Prediction Time in Test Set ---")
    # Convert the NumPy array of Timestamps to a pandas DatetimeIndex for easier searching
    y_times_test_dtindex = pd.to_datetime(y_times_test)

    try:
        # Get the index of the target time in the test set's start times
        found_index = y_times_test_dtindex.get_loc(TARGET_PREDICT_TIME)

        if isinstance(found_index, int):
            SAMPLE_INDEX = found_index
            print(f"✅ Found target time {TARGET_PREDICT_TIME} at SAMPLE_INDEX = {SAMPLE_INDEX}")
        else:
            raise KeyError("Found multiple matches or index type unexpected.")
            
    except KeyError:
        print(f"\n❌ CRITICAL ERROR: The target prediction start time {TARGET_PREDICT_TIME} was NOT found in the test set.")
        print("This means the data for this window was either split into the training set, or is missing/discontinuous.")
        exit()
    except Exception as e:
        print(f"\nAn error occurred while searching for the index: {e}")
        exit()
    # =========================================================================
    # 🚨 END DYNAMIC SAMPLE_INDEX FINDER 🚨
    # =========================================================================

    #
    # --- 3. Get Timestamps for Plotting (Unchanged) ---
    #
    print("\n--- Getting corrected timestamps ---")
    forecast_start_time = y_times_test[SAMPLE_INDEX]
    print(f"Correct forecast start time for sample {SAMPLE_INDEX} is: {forecast_start_time}")
    try:
        # Use the original 'df' (with DatetimeIndex) for timestamp lookups
        forecast_start_index_in_df = df.index.get_loc(forecast_start_time)
    except KeyError:
        print(f"CRITICAL ERROR: The timestamp {forecast_start_time} was not found.")
        exit()

    forecast_datetimes = df.index[forecast_start_index_in_df : forecast_start_index_in_df + N_FUTURE_HOURS]
    history_start_index_in_df = forecast_start_index_in_df - N_PAST_HOURS
    history_datetimes = df.index[history_start_index_in_df : forecast_start_index_in_df]

    if len(history_datetimes) != N_PAST_HOURS or len(forecast_datetimes) != N_FUTURE_HOURS:
        print("Error: Could not retrieve valid history or forecast timestamps even with fix.")
        print("This might happen if the test sample is right next to a data gap.")
        # We can still proceed, but the plot's 'Actual History' might be from a different sequence
        # For this script, we'll exit to be safe.
        print("Exiting.")
        exit()

    print(f"History timestamps from: {history_datetimes[0]} to {history_datetimes[-1]}")
    print(f"Forecast timestamps from: {forecast_datetimes[0]} to {forecast_datetimes[-1]}")
    print(f"Time continuity check: History ends at {history_datetimes[-1]}. Forecast starts 1 step later at {forecast_datetimes[0]}.")
    #
    # --- 4. Load Pre-Trained Model (Unchanged) ---
    #
    print("\n--- Loading model from disk... ---")
    model_json_saved = f'ft_myers_fl_2010_2021_gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL.json'
    weights_file = f'ft_myers_fl_2010_2021_gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL.h5'
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
    # --- 5. Make and Interpret Predictions (Unchanged) ---
    #
    print(f"\n--- Making prediction for sample {SAMPLE_INDEX} ---")
    can_predict_past = False
    predicted_past_levels = None # Initialize as None
    predicted_past_levels_list = []

    # Use the concatenated 'X' (from all sequences) for this logic
    if SAMPLE_INDEX < 0:
        sample_full_index = len(X) + SAMPLE_INDEX
    else:
        # Calculate the full index based on the train/test split size
        sample_full_index = len(X_train) + SAMPLE_INDEX 

    print("\n--- PAST PREDICTION DEBUG ---")
    print(f"  Total samples in X (full dataset): {len(X)}")
    print(f"  Calculated 'sample_full_index' (in X): {sample_full_index}")
    print(f"Generating {N_PAST_HOURS}-hour past 'hindcast' prediction...")
    n_loops_needed = int(np.ceil(N_PAST_HOURS / N_FUTURE_HOURS))
    print(f"  Model predicts {N_FUTURE_HOURS} steps. Need {n_loops_needed} loops to fill {N_PAST_HOURS} past hours.")
    try:
        for i in range(n_loops_needed):
            steps_back = N_PAST_HOURS - (i * N_FUTURE_HOURS)
            input_index = sample_full_index - steps_back
            print(f"\n  Loop {i+1}/{n_loops_needed}:")
            print(f"    Calculated input_index = {input_index}")
            if input_index < 0:
                print(f"    ERROR: Input index ({input_index}) is less than 0.")
                break
            print(f"    Accessing X[{input_index}] from full dataset...")
            # Use the global, concatenated 'X'
            input_data = X[input_index].reshape(1, N_PAST_HOURS, X.shape[2])
            print(f"    Running model.predict()...")
            pred_scaled = loaded_model.predict(input_data)
            # The scaling is correct: inverse_transform and convert to feet (*3.28)
            pred_levels = scaler_target.inverse_transform(pred_scaled) * 3.28 
            predicted_past_levels_list.append(pred_levels.flatten())
            print(f"    ...Success. Stored {len(pred_levels.flatten())} predicted points.")

        if predicted_past_levels_list:
            predicted_past_levels_raw = np.concatenate(predicted_past_levels_list)
            predicted_past_levels = predicted_past_levels_raw[:N_PAST_HOURS]
            if len(predicted_past_levels) == N_PAST_HOURS:
                can_predict_past = True
                print(f"\nSuccessfully generated past prediction. Final shape: {predicted_past_levels.shape}")
            else:
                print(f"\nWARNING: Past prediction shape mismatch. Got {predicted_past_levels.shape}, expected {N_PAST_HOURS}")
        else:
            print("\nWARNING: No past predictions were generated (list is empty).")
    except Exception as e:
        print(f"\n\n--- !!! CRITICAL ERROR !!! ---")
        print(f"An exception occurred during the past-prediction loop.")
        print(f"Error Type: {type(e)}")
        print(f"Error Message: {e}")
        traceback.print_exc()
    print("--- END PAST PREDICTION DEBUG ---\n")
    try:
        print("--- Making FUTURE prediction ---")
        sample_input = X_test[SAMPLE_INDEX].reshape(1, N_PAST_HOURS, X_test.shape[2])
        predicted_scaled = loaded_model.predict(sample_input)
        predicted_water_levels = scaler_target.inverse_transform(predicted_scaled)*3.28
        actual_scaled = y_test[SAMPLE_INDEX].reshape(1, N_FUTURE_HOURS)
        actual_water_levels = scaler_target.inverse_transform(actual_scaled)*3.28
        print(f"\nForecast for the next {N_FUTURE_HOURS} hours (starting {forecast_datetimes[0]}):")
        for i in range(N_FUTURE_HOURS):
            print(f"  - {forecast_datetimes[i]}: Predicted={predicted_water_levels[0][i]:.2f}m, Actual={actual_water_levels[0][i]:.2f}m")
    except IndexError:
        print(f"Error: Sample index {SAMPLE_INDEX} is out of bounds for the test set (size {len(X_test)}).")
        exit()
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        exit()

    #
    # --- 6. Plot the Sample Prediction (Unchanged) ---
    #
    print("\n--- Plotting sample prediction with history ---")
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")

    # Using a unique filename to avoid image cache
    plot_filename = f'ft_myers_fl_gru_idx{SAMPLE_INDEX}_{timestamp_str}_ft_myers_fl_gru.png'

    try:
        # --- GET HISTORY DATA ---
        sample_input_scaled_features = X_test[SAMPLE_INDEX]
        # Use the scaler_features we defined in the main block
        sample_input_UNSCALED_features = scaler_features.inverse_transform(sample_input_scaled_features)*3.28
        # Use the target_col_index we defined in the main block
        history_levels = sample_input_UNSCALED_features[:, target_col_index]

        # --- "Close the gap" fix for ACTUAL history ---
        forecast_start_actual_level = actual_water_levels[0][0]
        plot_history_datetimes = history_datetimes.union([forecast_datetimes[0]])
        plot_history_levels = np.append(history_levels, forecast_start_actual_level)

        # --- NEW: "Close the gap" fix for PREDICTED past ---
        plot_past_pred_levels = None # Initialize
        if can_predict_past:
            forecast_start_predicted_level = predicted_water_levels[0][0]
            plot_past_pred_levels = np.append(predicted_past_levels, forecast_start_predicted_level)
            print(f"Created {N_PAST_HOURS+1}-point 'plot_past_pred_levels' array. Shape: {plot_past_pred_levels.shape}")
            print('predicted_past_levels ',predicted_past_levels)
        # --- END NEW ---
        # --- PRE-PLOT DEBUG INFO ---
        print("\n--- PRE-PLOT DEBUG INFO ---")
        print(f"History Y (Levels) | shape: {plot_history_levels.shape} | min: {np.nanmin(plot_history_levels):.2f} | max: {np.nanmax(plot_history_levels):.2f}")
        print(f"Forecast Y (Actual)| shape: {actual_water_levels[0].shape} | min: {np.nanmin(actual_water_levels[0]):.2f} | max: {np.nanmax(actual_water_levels[0]):.2f}")
        print(f"Forecast Y (Pred)  | shape: {predicted_water_levels[0].shape} | min: {np.nanmin(predicted_water_levels[0]):.2f} | max: {np.nanmax(predicted_water_levels[0]):.2f}")

        # --- Y-Axis Limits (MODIFIED) ---
        all_y_data_list = [
            plot_history_levels,
            actual_water_levels[0],
            predicted_water_levels[0]
        ]

        if can_predict_past:
            print(f"PastPred Y (Levels)| shape: {plot_past_pred_levels.shape} | min: {np.nanmin(plot_past_pred_levels):.2f} | max: {np.nanmax(plot_past_pred_levels):.2f}")
            all_y_data_list.append(plot_past_pred_levels)
        else:
            print("PastPred Y (Levels)| NOT PLOTTING (can_predict_past=False)")
        all_y_data = np.concatenate(all_y_data_list)

        y_min = np.nanmin(all_y_data)
        y_max = np.nanmax(all_y_data)
        y_padding = (y_max - y_min) * 0.1
        if y_padding == 0: y_padding = 0.5
        plot_y_min = y_min - y_padding
        plot_y_max = y_max + y_padding

        print(f"Forcing Y-Axis limits from {plot_y_min:.2f} to {plot_y_max:.2f}")
        print("-----------------------------\n")

        # --- Plotting ---
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15, 7))

        print(f"--- FINAL PLOT CHECK: 'can_predict_past' is {can_predict_past} ---")
        if can_predict_past:
            print(f"--- Plotting 'Predicted Past' line FIRST ---")
            ax.plot(plot_history_datetimes,
                    plot_past_pred_levels,
                    marker='x',
                    linestyle=':',
                    linewidth=2.5,
                    label='Predicted Past (Hindcast)',
                    color='magenta',
                    zorder=10)
            print(f"\nForecast for the past hours:")
            print(plot_history_datetimes, plot_past_pred_levels)

        ax.plot(plot_history_datetimes,
                plot_history_levels,
                marker='o', markersize=4, linestyle='--',
                label='Actual History (Model Input)', color='blue')

        ax.plot(forecast_datetimes,
                actual_water_levels[0],
                marker='o', markersize=4, linestyle='--',
                label='Actual Future (Ground Truth)', color='cyan')

        ax.plot(forecast_datetimes,
                predicted_water_levels[0],
                marker='x', markersize=6, linestyle='-',
                label='Predicted (Model Output)', color='red')

        ax.axvline(forecast_datetimes[0], color='red', linestyle='--', label='Forecast Start')

        # --- Formatting ---
        all_datetimes = plot_history_datetimes.union(forecast_datetimes)
        ax.set_xlim(all_datetimes[0] - dt.timedelta(hours=1),
                    all_datetimes[-1] + dt.timedelta(hours=1))

        plot_ticks = pd.date_range(start=all_datetimes[0], end=all_datetimes[-1], freq='4h')
        ax.set_xticks(plot_ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()

        #ax.set_title(f'FINAL PLOT - Water Level Forecast (Sample {SAMPLE_INDEX}) - {timestamp_str}')
        ax.set_title(f'FINAL PLOT - Water Level Forecast (Station: {station_name}) - {timestamp_str}')
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Water Level (feet)')
        ax.set_ylim(plot_y_min, plot_y_max)
        ax.legend()
        ax.grid(True)

        fig.savefig(plot_filename)

        print(f"\n********************************************************")
        print(f"Plot saved as a NEW FILE: '{plot_filename}'")
        print(f"Please look for this *exact* file in your directory.")
        print(f"********************************************************")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        print("Check the PRE-PLOT DEBUG INFO above for shape mismatches or NaN values.")
