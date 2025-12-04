#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
#
# This script was built, optimized, and tuned with help of Gemini
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all but fatal errors
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import datetime as dt
import tensorflow.keras.backend as K
import traceback
import logging
#
# Suppress TensorFlow log messages (set to ERROR level)
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# --- Define GFS Guidance Columns (MUST MATCH TRAINING SCRIPT) ---
FUTURE_GUIDANCE_COLS = [
    'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust'
]
OBS_TARGET_COLUMN = 'water_level'
OBS_TEMP_COLUMN = 'air_temp' # Column to check for Kelvin conversion
# -----------------------------------------------------------------

try:
    print("Forcing TensorFlow to use CPU-only to avoid hardware conflict...")
    tf.config.set_visible_devices([], 'GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("CPU forced. Proceeding with prediction script...")
except Exception as e:
    print(f"Warning: Could not force CPU: {e}")

# --- Helper function for timestamp parsing ---
# Add common formats that might be in your CSV files
date_format_candidates = ['%Y-%m-%d %H %M %S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H', '%Y-%m-%d %H:%M']

def parse_timestamps(df, col_name, formats):
    for fmt in formats:
        try:
            # Use errors='raise' in the final parsing step to confirm the working format
            df[col_name] = pd.to_datetime(df[col_name], format=fmt)
            print(f"Timestamp column '{col_name}' successfully parsed using format: {fmt}")
            return df
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp column '{col_name}' with any of the provided formats: {formats}.")

#
# --- 1. Load Data and Process Timestamps (ENHANCED DEBUGGING) ---
#
station_name = 'St Petersburg FL'
print("Loading data...")

try:
    # Attempt to load the raw data files
    data = pd.read_csv('../storm_surge/st_petersburg_test_data_gfs_etss_2024.csv')
    obs = pd.read_csv('../storm_surge/observations_st_petersburg_fl_2022_2024.csv')

except FileNotFoundError as e:
    print("\n--- CRITICAL FILE NOT FOUND ERROR ---")
    print(f"Error: {e}")
    print("Please check the relative paths (`../storm_surge/`) to your CSV files.")
    print("-------------------------------------")
    exit()
except Exception as e:
    print(f"Unexpected error during file loading: {e}")
    traceback.print_exc()
    exit()


# Process GFS/Training Data (df_raw)
try:
    # Check for timestamp column existence
    if 'timestamp' not in data.columns:
        raise ValueError(f"Required column 'timestamp' not found in GFS data. Available columns: {data.columns.tolist()}")

    data = parse_timestamps(data, 'timestamp', date_format_candidates)
    data.set_index('timestamp', inplace=True)
    df_raw = data.sort_values(by='timestamp')
    if df_raw.index.has_duplicates:
        df_raw = df_raw.groupby(df_raw.index).mean()

except Exception as e:
    print(f"\n--- CRITICAL GFS DATA PARSING ERROR ---")
    print(f"Error details: {e}")
    traceback.print_exc() # Print the full stack trace
    print("---------------------------------------")
    exit()


# --- Process Observation Data (obs_df) with Kelvin to Celsius Conversion ---
try:
    # Check for timestamp column existence
    if 'timestamp' not in obs.columns:
        raise ValueError(f"Required column 'timestamp' not found in Observation data. Available columns: {obs.columns.tolist()}")

    obs = parse_timestamps(obs, 'timestamp', date_format_candidates)
    obs.set_index('timestamp', inplace=True)
    obs_df_raw = obs.sort_values(by='timestamp')
    if obs_df_raw.index.has_duplicates:
        obs_df_raw = obs_df_raw.groupby(obs_df_raw.index).mean()

    # MODIFICATION: Kelvin to Celsius Conversion (T_C = T_K - 273.15)
    if OBS_TEMP_COLUMN in obs_df_raw.columns:
        print(f"Converting '{OBS_TEMP_COLUMN}' in observation data from Kelvin to Celsius...")
        # NOTE: The conversion is commented out in your original code, leaving it commented here.
        # obs_df_raw[OBS_TEMP_COLUMN] = obs_df_raw[OBS_TEMP_COLUMN] - 273.15

except Exception as e:
    print(f"\n--- CRITICAL OBSERVATION DATA PARSING ERROR ---")
    print(f"Error details: {e}")
    traceback.print_exc()
    print("---------------------------------------")
    exit()

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
# --- 3. Data Preprocessing Function (Unchanged) ---
#
def prepare_data(df_sequence, n_past, n_future, target_col, scaler_features, scaler_target, future_guidance_cols):
    """
    Creates windows (X, y) for a single data sequence, incorporating future guidance in X.
    """
    past_feature_cols = df_sequence.columns.tolist()
    scaled_features_df = pd.DataFrame(scaler_features.transform(df_sequence[past_feature_cols]),
                                      columns=past_feature_cols)
    scaled_target = scaler_target.transform(df_sequence[target_col].values.reshape(-1, 1))

    scaled_past_features_np = scaled_features_df[past_feature_cols].values
    scaled_guidance_features_np = scaled_features_df[future_guidance_cols].values

    X, y = [], []
    y_start_timestamps = []

    num_features = len(past_feature_cols)
    
    try:
        guidance_indices = [past_feature_cols.index(col) for col in future_guidance_cols]
    except ValueError as e:
        print(f"\nFATAL: Guidance column missing from feature list. Error: {e}")
        print(f"Features: {past_feature_cols}")
        print(f"Guidance: {future_guidance_cols}")
        raise # Stop execution

    for i in range(n_past, len(df_sequence) - n_future + 1):
        past_input = scaled_past_features_np[i - n_past:i, :]
        future_guidance_data = scaled_guidance_features_np[i:i + n_future, :]
        future_input_padded = np.zeros((n_future, num_features), dtype=np.float32)

        for idx_col, guidance_col_idx in enumerate(guidance_indices):
            future_input_padded[:, guidance_col_idx] = future_guidance_data[:, idx_col]

        X_sequence = np.concatenate([past_input, future_input_padded], axis=0)

        X.append(X_sequence)
        y.append(scaled_target[i:i + n_future, 0])
        y_start_timestamps.append(df_sequence.index[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    y_start_times = np.array(y_start_timestamps)

    return X, y, y_start_times


# --- Main Execution ---
if __name__ == '__main__':
    # Configuration -
    npast = input("Enter N_past ")
    nfuture = input("Enter N_future ")
    N_PAST_HOURS = int(npast)
    N_FUTURE_HOURS = int(nfuture)
    TARGET_COLUMN = 'water_level'

    # --- Prediction Start Mode Selection ---
    DEFAULT_CUTOFF = '2022-08-28 00:00:00'
    print("\nSelect Prediction Start Mode:")
    print(f"  [1] Fixed Date Cutoff (Use data up to: {DEFAULT_CUTOFF})")
    print("  [2] Last Available Hour (Use ALL available training data)")
    mode_choice = input("Enter choice (1 or 2): ")

    if mode_choice == '1':
        MAX_DATA_TIMESTAMP = DEFAULT_CUTOFF
    elif mode_choice == '2':
        # Find the absolute last timestamp in the raw, parsed data
        if not df_raw.empty:
            MAX_DATA_TIMESTAMP = df_raw.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        else:
            print("Error: Raw GFS data is empty. Cannot determine last hour.")
            exit()
    else:
        print("Invalid choice. Exiting.")
        exit()

    print(f"Prediction will use data up to: {MAX_DATA_TIMESTAMP}")
    # ----------------------------------------------------

    # --- NEW FILTERING STEP: Apply Cutoff to GFS/Training Data (df) and Observations (obs_df) ---
    df = df_raw.copy()
    obs_df = obs_df_raw.copy()

    df = df[df.index <= MAX_DATA_TIMESTAMP]
    obs_df = obs_df[obs_df.index <= MAX_DATA_TIMESTAMP]
    df = df.dropna() # Re-drop NaNs after filtering

    print(f"Filtered GFS data size: {len(df)}")
    print(f"Filtered OBS data size: {len(obs_df)}")

    # We always want the LAST possible sample for the prediction
    SAMPLE_INDEX = -1

    if N_PAST_HOURS % N_FUTURE_HOURS != 0 or N_PAST_HOURS // N_FUTURE_HOURS != 2:
        print(f"ERROR: Multi-color plot logic requires N_PAST_HOURS ({N_PAST_HOURS}) to be exactly 2x N_FUTURE_HOURS ({N_FUTURE_HOURS}).")
        exit()

    # --- Preprocessing ---
    # 1. Fit Scalers ONCE on all data
    print("Fitting scalers on entire filtered dataset...")
    features = df.columns
    target = df[TARGET_COLUMN]
    target_col_index = df.columns.get_loc(TARGET_COLUMN)

    if not all(col in features for col in FUTURE_GUIDANCE_COLS):
        missing = [col for col in FUTURE_GUIDANCE_COLS if col not in features]
        print(f"Error: Missing GFS guidance columns in test data: {missing}. Exiting.")
        exit()

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_features.fit(df[features])

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(target.values.reshape(-1, 1))
    print("Scalers fitted.")

    # 2. Group and Window Data
    df_with_col = df.reset_index()
    time_diff = df_with_col['timestamp'].diff()
    sequence_id = (time_diff > pd.Timedelta('1 hour')).cumsum()
    groups = df_with_col.groupby(sequence_id)

    all_X, all_y, all_y_start_times = [], [], []
    MIN_SEQ_LEN = N_PAST_HOURS + N_FUTURE_HOURS

    for seq_id, group_df in groups:
        if len(group_df) >= MIN_SEQ_LEN:
            group_df_indexed = group_df.set_index('timestamp')
            X_seq, y_seq, y_times_seq = prepare_data(
                group_df_indexed, N_PAST_HOURS, N_FUTURE_HOURS, TARGET_COLUMN,
                scaler_features, scaler_target, FUTURE_GUIDANCE_COLS
            )

            if X_seq.shape[0] > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)
                all_y_start_times.append(y_times_seq)

    if not all_X:
        print("\n--- CRITICAL ERROR ---")
        print(f"No valid sequences found with the minimum required length of {MIN_SEQ_LEN} before cutoff date.")
        exit()

    X = np.vstack(all_X)
    y = np.vstack(all_y)
    y_start_times = np.concatenate(all_y_start_times)

    print("\n--- Feature lentgh ---", len(X)," Feature lentgh ",len(y))

    # 3. Split Data (Needed for the consistent model input format, even if test_size is arbitrary for this use)
    test_size_val = max(0.2, MIN_SEQ_LEN / len(X)) if len(X) > MIN_SEQ_LEN else 0.5

    X_train, X_test, y_train, y_test, y_times_train, y_times_test = train_test_split(
        X, y, y_start_times, test_size=test_size_val, random_state=42, shuffle=False
    )

    sample_input = X_test[SAMPLE_INDEX].reshape(1, N_PAST_HOURS + N_FUTURE_HOURS, X_test.shape[2])

    # 4. Get Timestamps and Load Model

    sample_full_index = len(X_train) + SAMPLE_INDEX

    forecast_start_time = y_start_times[sample_full_index]

    try:
        forecast_start_index_in_df = df.index.get_loc(forecast_start_time)
    except KeyError:
        print(f"CRITICAL ERROR: The forecast start timestamp {forecast_start_time} was not found in the filtered GFS data.")
        exit()

    history_start_index_in_df = forecast_start_index_in_df - N_PAST_HOURS
    history_datetimes = df.index[history_start_index_in_df : forecast_start_index_in_df]
    forecast_datetimes = df.index[forecast_start_index_in_df : forecast_start_index_in_df + N_FUTURE_HOURS]

    all_plot_datetimes = history_datetimes.union(forecast_datetimes)
    print(f"\n--- Prediction Window Selected ---")
    print(f"Forecast Start Time: {forecast_start_time}")
    print(f"Last available time point for this forecast: {forecast_datetimes[-1]}")

    # --- Load Pre-Trained Model ---
    model_json_saved = f'../storm_surge/st_petersburg_fl_gru_multi_steps_{N_PAST_HOURS}past_plus_{N_FUTURE_HOURS}future_guidance_OPTIMAL.json'
    weights_file = f'../storm_surge/st_petersburg_fl_gru_multi_steps_{N_PAST_HOURS}past_plus_{N_FUTURE_HOURS}future_guidance_OPTIMAL.h5'

    print('model_json_saved ',model_json_saved)
    dilate_loss_fn = create_dilate_loss(alpha=0.5)
    try:
        json_file = open(model_json_saved, 'r')
        loaded_model = model_from_json(json_file.read(), custom_objects={'dilate_loss': dilate_loss_fn})
        json_file.close()
        loaded_model.load_weights(weights_file)
        loaded_model.compile(optimizer='adam', loss=dilate_loss_fn)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\n--- CRITICAL MODEL LOADING ERROR ---")
        print(f"Error loading model: {e}")
        print(f"Check if files exist: {model_json_saved} and {weights_file}")
        traceback.print_exc()
        print("------------------------------------")
        exit()

    # --- 5. Make and Interpret Predictions ---

    # Past Prediction (Hindcast) setup
    can_predict_past = False
    predicted_past_levels = None

    hindcast_start_index = sample_full_index - N_PAST_HOURS
    if hindcast_start_index >= 0:
        can_predict_past = True
        predicted_past_levels_list = []

        num_hindcast_segments = N_PAST_HOURS // N_FUTURE_HOURS

        for i in range(num_hindcast_segments):
            input_index = hindcast_start_index + (i * N_FUTURE_HOURS)

            if input_index < 0 or input_index >= len(X):
                print(f"Warning: Cannot generate hindcast segment {i+1} due to out-of-bounds index.")
                continue

            input_data_segment = X[input_index].reshape(1, N_PAST_HOURS + N_FUTURE_HOURS, X.shape[2])

            pred_scaled = loaded_model.predict(input_data_segment, verbose=0)
            pred_levels = scaler_target.inverse_transform(pred_scaled)
            predicted_past_levels_list.append(pred_levels.flatten())

        if predicted_past_levels_list:
             predicted_past_levels = np.concatenate(predicted_past_levels_list)[:N_PAST_HOURS]
        else:
            can_predict_past = False


    # Making FUTURE prediction (Main forecast)
    try:
        predicted_scaled = loaded_model.predict(sample_input, verbose=0)
        predicted_water_levels = scaler_target.inverse_transform(predicted_scaled)

        actual_scaled = y_test[SAMPLE_INDEX].reshape(1, N_FUTURE_HOURS)
        actual_water_levels = scaler_target.inverse_transform(actual_scaled)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc()
        exit()

    # --- 6. Retrieve Observation and GFS Guidance Data for Plotting ---

    # 6A. Observations (Fixed: Reindex, but do NOT drop NaNs here)
    try:
        # Reindex the full OBS data to the plotting window (introduces NaNs for missing hours)
        obs_window = obs_df[OBS_TARGET_COLUMN].reindex(all_plot_datetimes)

        # We plot the full reindexed data, including NaNs where data is missing
        obs_levels = obs_window.values * 3.28 # Convert to feet
        obs_datetimes = obs_window.index # This is now identical to all_plot_datetimes

    except KeyError:
        print(f"Warning: '{OBS_TARGET_COLUMN}' not found in observation data. Plotting observations will be skipped.")
        # Set to NaN-filled arrays of correct length to prevent errors
        obs_levels = np.full(len(all_plot_datetimes), np.nan)
        obs_datetimes = all_plot_datetimes


    # 6B. ETSS Water Level Guidance (from the training/input data)
    try:
        # Reindex the filtered GFS data (df) to match the plotting window
        gfs_guidance_window = df[TARGET_COLUMN].reindex(all_plot_datetimes)
        gfs_guidance_levels = gfs_guidance_window.values # Convert to feet
    except Exception as e:
        print(f"Warning: Could not retrieve ETSS Water Level Guidance: {e}")
        gfs_guidance_levels = np.full(len(all_plot_datetimes), np.nan)


    #
    # --- 7. Plot the Sample Prediction ---
    #
    print("\n--- Plotting sample prediction with Observations and GFS Guidance ---")
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    plot_filename = f'../storm_surge/FINAL_PLOT_idx_LATEST_{timestamp_str}_gru_st_petersburg_guidance_OBS.png'


    try:
        # --- GET HISTORY DATA (GFS Input Data) ---
        history_levels = gfs_guidance_levels[:N_PAST_HOURS]

        # --- "Close the gap" fix for ACTUAL history ---
        forecast_start_actual_level = actual_water_levels[0][0]
        plot_history_datetimes = history_datetimes.union([forecast_datetimes[0]])
        plot_history_levels = np.append(history_levels, forecast_start_actual_level)

        # --- "Close the gap" fix for PREDICTED past ---
        plot_past_pred_levels = None
        if can_predict_past:
            forecast_start_predicted_level = predicted_water_levels[0][0]
            plot_past_pred_levels = np.append(predicted_past_levels, forecast_start_predicted_level)

        # --- Dynamic Y-Axis Limits Calculation ---
        all_y_data_list = [
            plot_history_levels,
            actual_water_levels[0],
            predicted_water_levels[0],
            obs_levels,
        ]

        if can_predict_past:
            all_y_data_list.append(plot_past_pred_levels)

        # Concatenate and filter NaNs for robust min/max calculation
        all_y_data = np.concatenate(all_y_data_list)
        valid_y_data = all_y_data[~np.isnan(all_y_data)]

        # Calculate min/max based ONLY on the primary (non-outlier) data
        if len(valid_y_data) > 0:
            y_min = np.min(valid_y_data)
            y_max = np.max(valid_y_data)
        else:
            # Fallback if all data points are NaN
            y_min = 0.0
            y_max = 2.0

        y_padding = (y_max - y_min) * 0.1
        if y_padding == 0 or np.isnan(y_padding): y_padding = 0.25 # Use a fixed, small padding

        plot_y_min = y_min - y_padding
        plot_y_max = y_max + y_padding

        # ENSURE HIGH ETSS VALUES ARE VISIBLE OR CAPPED
        max_gfs = np.nanmax(gfs_guidance_levels)
        min_gfs = np.nanmin(gfs_guidance_levels)
        max_obs = np.nanmax(obs_levels) if np.any(~np.isnan(obs_levels)) else 0.0
        min_obs = np.nanmin(obs_levels) if np.any(~np.isnan(obs_levels)) else 100.0

        max_water_level = max(max_gfs, max_obs) if not np.isnan(max_gfs) and not np.isnan(max_obs) else np.nanmax([max_gfs, max_obs])
        min_water_level = min(min_gfs, min_obs) if not np.isnan(min_gfs) and not np.isnan(min_obs) else np.nanmin([min_gfs, min_obs])

        if np.isnan(max_water_level): max_water_level = 0.0
        if np.isnan(min_water_level): min_water_level = 0.0 # Safety default

        # Round and pad for final limits
        max_water_level = np.round(max_water_level + 0.5)
        min_water_level = np.round(min_water_level - 0.5)

        if plot_y_max < max_water_level:
            plot_y_max = max_water_level

        if plot_y_min > min_water_level:
            plot_y_min = min_water_level


        # --- Plotting ---
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15, 7))

        # --- 7A: Plot Observation Data ---

        # 1. Isolate the history observations (the first N_PAST_HOURS = 24 data points)
        obs_history_datetimes = obs_datetimes[:N_PAST_HOURS]
        obs_history_levels = obs_levels[:N_PAST_HOURS]

        # 2. Define the split index for the two 12-hour periods (24 / 2 = 12)
        split_idx = N_PAST_HOURS // 2

        # 3. Plot the first 12-hour period (H-24 to H-12)
        ax.plot(obs_history_datetimes[:split_idx],
                obs_history_levels[:split_idx],
                marker='o', markersize=7, linestyle='dashdot',linewidth=1.5,
                label='Observed Water Level (H-12 to H-6)', color='teal', zorder=15)

        # 4. Plot the second 12-hour period (H-12 to H-0)
        ax.plot(obs_history_datetimes[split_idx:],
                obs_history_levels[split_idx:],
                marker='o', markersize=7, linestyle='dashdot',linewidth=1.5,
                label='Observed Water Level (H-6 to H-0)', color='darkgreen', zorder=15)

        # 5. Plot the future observations (H-0 to H+12)
        ax.plot(obs_datetimes[N_PAST_HOURS:],
                obs_levels[N_PAST_HOURS:],
                marker='o', markersize=7, linestyle='dashdot',linewidth=1.5,
                label='Observed Water Level (Future)', color='black', zorder=15)

        # --- 7B: Plot ETSS Water Level Guidance ---

        # Define the split points
        P1_END_IDX = N_FUTURE_HOURS       # 12
        P2_END_IDX = N_PAST_HOURS        # 24

        # 1. Plot History Guidance: First 12 hours (H-24 to H-12)
        ax.plot(all_plot_datetimes[:P1_END_IDX],
                gfs_guidance_levels[:P1_END_IDX],
                linestyle='--', linewidth=1.5, marker='.', markersize=8,
                label='ETSS Guidance (H-12 to H-6)', color='darkcyan', zorder=1)
        
        # 2. Plot History Guidance: Second 12 hours (H-12 to H-0)
        ax.plot(all_plot_datetimes[P1_END_IDX:P2_END_IDX],
                gfs_guidance_levels[P1_END_IDX:P2_END_IDX],
                linestyle='--', linewidth=1.5, marker='.', markersize=8,
                label='ETSS Guidance (H-6 to H-0)', color='navy', zorder=1)
        
        # 3. Plot Future Guidance (H-0 to H+12)
        ax.plot(all_plot_datetimes[P2_END_IDX:],
                gfs_guidance_levels[P2_END_IDX:],
                linestyle='--', linewidth=2.5, marker='.', markersize=8,
                label='ETSS Guidance (Future)', color='dodgerblue', zorder=1)
        

        # --- 7C: Multi-color Hindcast Plotting (Model's Hindcast) ---
        if can_predict_past:
            segment_colors = ['magenta', 'orange']
            num_segments = N_PAST_HOURS // N_FUTURE_HOURS

            for i in range(num_segments):
                start_idx_data = i * N_FUTURE_HOURS
                end_idx_data = (i + 1) * N_FUTURE_HOURS

                segment_levels = predicted_past_levels[start_idx_data:end_idx_data]
                segment_datetimes = history_datetimes[start_idx_data:end_idx_data]

                is_last_segment = (i == num_segments - 1)

                if is_last_segment:
                    segment_levels = np.append(segment_levels, forecast_start_predicted_level)
                    segment_datetimes = history_datetimes[start_idx_data:].union([forecast_datetimes[0]])

                label = f'Hindcast Segment {i+1}'

                ax.plot(segment_datetimes,
                        segment_levels,
                        marker='d',
                        markersize=8,
                        linestyle=':',
                        linewidth=3.0, # Increased line width for visibility
                        label=label,
                        color=segment_colors[i],
                        zorder=10)

        # --- 7D: Plot the Predicted Future (Model Output) ---
        ax.plot(forecast_datetimes,
                predicted_water_levels[0],
                marker='d', markersize=8, linestyle='-',
                label='Predicted Future (Model Output)', color='red', zorder=10)

        # --- Vertical line for forecast start ---
        ax.axvline(forecast_datetimes[0], color='red', linestyle='--', label='Forecast Start', zorder=1)

        # --- Formatting ---
        ax.set_xlim(all_plot_datetimes[0] - dt.timedelta(hours=1), all_plot_datetimes[-1] + dt.timedelta(hours=1))

        plot_ticks = pd.date_range(start=all_plot_datetimes[0], end=all_plot_datetimes[-1], freq='4h')
        ax.set_xticks(plot_ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()

        # Set dynamic Y-axis limits
        ax.set_ylim(plot_y_min, plot_y_max)

        ax.set_title(f'Water Level Forecast vs. Observation (Station: {station_name}) - {timestamp_str}\nData Cutoff: {MAX_DATA_TIMESTAMP}')
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Water Level (feet)')
        ax.legend(loc='best')
        ax.grid(True)

        fig.savefig(plot_filename)

        print(f"\n********************************************************")
        print(f"Plot saved as a NEW FILE: '{plot_filename}'")
        print(f"Dynamic Y-Axis range: {plot_y_min:.2f} to {plot_y_max:.2f} feet.")
        print(f"Prediction Start Time: {forecast_start_time}")
        print(f"********************************************************")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        traceback.print_exc()
