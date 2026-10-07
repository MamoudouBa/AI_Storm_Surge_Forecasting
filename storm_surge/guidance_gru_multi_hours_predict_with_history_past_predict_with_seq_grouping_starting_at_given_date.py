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
import matplotlib.dates as mdates 
from datetime import datetime
import datetime as dt
import tensorflow.keras.backend as K
import traceback
import os

# --- Define GFS Guidance Columns (MUST MATCH TRAINING SCRIPT) ---
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
# --- 1. Load Data and Filter (MODIFIED for Cutoff Date) ---
#
station_name = 'St Petersburg FL' 
print("Loading data...")
data = pd.read_csv('st_petersburg_training_gfs_wl_data.csv')
obs = pd.read_csv('observations_st_petersburg_fl.csv') 

# --- CONFIGURATION: Define Max Data Timestamp ---
MAX_DATA_TIMESTAMP = '2022-08-28 00:00:00' # Setting a cutoff slightly after your target forecast start (2022-11-30 00:00:00) 
# Note: The *actual* forecast start time depends on the length of N_PAST and data availability.
print(f"Applying data cutoff: All GFS and OBS data restricted to <= {MAX_DATA_TIMESTAMP}")
# ----------------------------------------------------

# --- Process GFS/Training Data (df) ---
data = data.dropna()
data = data.sort_values(by='timestamp')

date_format_candidates = ['%Y-%m-%d %H %M %S', '%Y-%m-%d %H:%M:%S']

def parse_timestamps(df, col_name, formats):
    for fmt in formats:
        try:
            df[col_name] = pd.to_datetime(df[col_name], format=fmt)
            return df
        except ValueError:
            continue
    raise ValueError("Could not parse timestamp column with provided formats.")

try:
    data = parse_timestamps(data, 'timestamp', date_format_candidates)
except Exception as e:
    print(f"Error parsing GFS datetimes: {e}")
    exit()

data.set_index('timestamp', inplace=True)
df = data.sort_values(by='timestamp')
if df.index.has_duplicates:
    df = df.groupby(df.index).mean()

# --- Process Observation Data (obs_df) ---
OBS_TARGET_COLUMN = 'water_level'
try:
    obs = parse_timestamps(obs, 'timestamp', date_format_candidates)
    obs.set_index('timestamp', inplace=True)
    obs_df = obs.sort_values(by='timestamp')
    if obs_df.index.has_duplicates:
        obs_df = obs_df.groupby(obs_df.index).mean()
except Exception as e:
    print(f"Error processing observation data: {e}.")
    exit()

# --- NEW FILTERING STEP: Apply Cutoff ---
df = df[df.index <= MAX_DATA_TIMESTAMP]
obs_df = obs_df[obs_df.index <= MAX_DATA_TIMESTAMP]
print(f"Filtered GFS data size: {len(df)}")
print(f"Filtered OBS data size: {len(obs_df)}")

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
    guidance_indices = [past_feature_cols.index(col) for col in future_guidance_cols]

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
    N_PAST_HOURS = 24
    N_FUTURE_HOURS = 12       
    TARGET_COLUMN = 'water_level'
    
    # --- IMPORTANT CHANGE: Set SAMPLE_INDEX to -1 to select the *last* available sample ---
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

    # 3. Split Data
    X_train, X_test, y_train, y_test, y_times_train, y_times_test = train_test_split(
        X, y, y_start_times, test_size=0.2, random_state=42, shuffle=False
    )
    
    # 4. Get Timestamps and Load Model
    # Since SAMPLE_INDEX = -1, we select the last sample from the X_test set.
    sample_full_index = len(X_train) + SAMPLE_INDEX
    
    forecast_start_time = y_start_times[sample_full_index]
    
    try:
        forecast_start_index_in_df = df.index.get_loc(forecast_start_time)
    except KeyError:
        print(f"CRITICAL ERROR: The timestamp {forecast_start_time} was not found.")
        exit()

    history_start_index_in_df = forecast_start_index_in_df - N_PAST_HOURS
    history_datetimes = df.index[history_start_index_in_df : forecast_start_index_in_df]
    forecast_datetimes = df.index[forecast_start_index_in_df : forecast_start_index_in_df + N_FUTURE_HOURS]

    all_plot_datetimes = history_datetimes.union(forecast_datetimes) 
    print(f"\n--- Prediction Window Selected ---")
    print(f"Forecast Start Time: {forecast_start_time}")
    print(f"Last available time point for this forecast: {forecast_datetimes[-1]}")
    
    # --- Load Pre-Trained Model ---
    model_json_saved = f'ft_myers_fl_2010_2021_gru_multi_steps_{N_PAST_HOURS}past_plus_{N_FUTURE_HOURS}future_guidance_OPTIMAL.json'
    weights_file = f'ft_myers_fl_2010_2021_gru_multi_steps_{N_PAST_HOURS}past_plus_{N_FUTURE_HOURS}future_guidance_OPTIMAL.h5'

    dilate_loss_fn = create_dilate_loss(alpha=0.5)
    try:
        json_file = open(model_json_saved, 'r')
        loaded_model = model_from_json(json_file.read(), custom_objects={'dilate_loss': dilate_loss_fn})
        json_file.close()
        loaded_model.load_weights(weights_file)
        loaded_model.compile(optimizer='adam', loss=dilate_loss_fn)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # --- 5. Make and Interpret Predictions ---

    # Past Prediction (Hindcast) setup
    can_predict_past = False
    predicted_past_levels = None

    hindcast_start_index = sample_full_index - N_PAST_HOURS
    if hindcast_start_index >= 0:
        can_predict_past = True
        predicted_past_levels_list = []

        for i in range(N_PAST_HOURS // N_FUTURE_HOURS):
            input_index = hindcast_start_index + (i * N_FUTURE_HOURS)
            input_data = X[input_index].reshape(1, N_PAST_HOURS + N_FUTURE_HOURS, X.shape[2])

            pred_scaled = loaded_model.predict(input_data)
            pred_levels = scaler_target.inverse_transform(pred_scaled) * 3.28
            predicted_past_levels_list.append(pred_levels.flatten())

        predicted_past_levels = np.concatenate(predicted_past_levels_list)[:N_PAST_HOURS]

    # Making FUTURE prediction (Main forecast)
    try:
        sample_input = X_test[SAMPLE_INDEX].reshape(1, N_PAST_HOURS + N_FUTURE_HOURS, X_test.shape[2])
        predicted_scaled = loaded_model.predict(sample_input)
        predicted_water_levels = scaler_target.inverse_transform(predicted_scaled) * 3.28
        actual_scaled = y_test[SAMPLE_INDEX].reshape(1, N_FUTURE_HOURS)
        actual_water_levels = scaler_target.inverse_transform(actual_scaled) * 3.28

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc()
        exit()

    # --- 6. Retrieve Observation Data for Plotting ---
    try:
        obs_window = obs_df[OBS_TARGET_COLUMN].reindex(all_plot_datetimes)
        obs_levels = obs_window.values * 3.28 # Convert to feet
        obs_datetimes = obs_window.index

    except KeyError:
        print(f"Warning: '{OBS_TARGET_COLUMN}' not found in observation data. Plotting observations will be skipped.")
        obs_levels = np.full(len(all_plot_datetimes), np.nan)
        obs_datetimes = all_plot_datetimes

    #
    # --- 7. Plot the Sample Prediction ---
    #
    print("\n--- Plotting sample prediction with Observations ---")
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    plot_filename = f'FINAL_PLOT_idx_LATEST_{timestamp_str}_gru_st_petersburg_guidance_OBS.png'

    try:
        # --- GET HISTORY DATA (GFS Input Data) ---
        sample_input_scaled_features = X_test[SAMPLE_INDEX][:N_PAST_HOURS, :]
        sample_input_UNSCALED_features = scaler_features.inverse_transform(sample_input_scaled_features) * 3.28
        history_levels = sample_input_UNSCALED_features[:, target_col_index]

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
            obs_levels # Include observations in dynamic range calculation
        ]

        if can_predict_past:
            all_y_data_list.append(plot_past_pred_levels)

        all_y_data = np.concatenate(all_y_data_list)

        y_min = np.nanmin(all_y_data)
        y_max = np.nanmax(all_y_data)
        y_padding = (y_max - y_min) * 0.1
        if y_padding == 0 or np.isnan(y_padding): y_padding = 0.5 # Default padding
        plot_y_min = y_min - y_padding
        plot_y_max = y_max + y_padding

        # --- Plotting ---
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15, 7))

        # --- 7A: Plot Observation Data ---
        ax.plot(obs_datetimes,
                obs_levels,
                marker='.', markersize=4, linestyle='-',
                label='Observed Water Level (Ground Truth)', color='black', zorder=2)

        # --- 7B: Multi-color Hindcast Plotting (Model's Hindcast) ---
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
                        marker='x',
                        linestyle=':',
                        linewidth=2.5,
                        label=label,
                        color=segment_colors[i],
                        zorder=10)
            
            # Add a generic dashed line to represent the overall hindcast
            ax.plot(plot_history_datetimes, plot_past_pred_levels, linestyle=':', color='black', linewidth=1, label='_nolegend_', zorder=9)


        # --- 7C: Plot the Predicted Future (Model Output) ---
        ax.plot(forecast_datetimes,
                predicted_water_levels[0],
                marker='x', markersize=6, linestyle='-',
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

        ax.set_title(f'Water Level Forecast vs. Observation (Station: {station_name}) - {timestamp_str}')
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Water Level (feet)')
        ax.legend()
        ax.grid(True)

        fig.savefig(plot_filename)

        print(f"\n********************************************************")
        print(f"Plot saved as a NEW FILE: '{plot_filename}'")
        print(f"Dynamic Y-Axis range: {plot_y_min:.2f} to {plot_y_max:.2f} feet.")
        print(f"********************************************************")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        traceback.print_exc()
