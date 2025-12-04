#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disables all GPUs
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')
    print("CPU forced. Proceeding with prediction script...")
except Exception as e:
    print(f"Warning: Could not force CPU: {e}")

#
# --- 1. Load Data (UNCHANGED) ---
#
station_name = 'St_petersburg_clear_water' # Used for plot title
print("Loading data for two stations...")

# --- Manually parse timestamp with the correct format ---
date_format_1 = '%Y-%m-%d %H %M %S'
date_format_2 = '%Y-%m-%d %H:%M:%S' # Fallback format

def load_station_data(filepath, suffix):
    """Helper function to load, parse, and suffix a station's data."""
    try:
        data = pd.read_csv(filepath)
        data = data.dropna()
        data = data.sort_values(by='timestamp')

        # Try multiple datetime formats
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'], format=date_format_1)
        except ValueError:
            print(f"Format 1 failed for {filepath}, trying format 2...")
            data['timestamp'] = pd.to_datetime(data['timestamp'], format=date_format_2)

        data.set_index('timestamp', inplace=True)
        data_sorted = data.sort_index()

        # Handle duplicates
        if data_sorted.index.has_duplicates:
            print(f"Warning: Duplicate timestamps found in {filepath}. Aggregating with mean().")
            data_sorted = data_sorted.groupby(data_sorted.index).mean()

        # Add suffix to all columns (e.g., 'water_level' -> 'water_level_oldporttmp')
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
df_st_petersburg = load_station_data('test_dataset_2022_old_port_tampa_for_water_level.csv', 'oldporttmp')
df_clear_water = load_station_data('test_set_2022_ft_myers_for_water_level.csv', 'ftmyers')

# Exit if either dataset failed to load
if df_st_petersburg is None or df_clear_water is None:
    print("Error loading one or more data files. Exiting.")
    exit()

# --- Merge the two DataFrames ---
print("\nMerging station data...")
df = pd.merge(df_st_petersburg, df_clear_water, left_index=True, right_index=True, how='inner')

if df.empty:
    print("Error: Merged DataFrame is empty. No common timestamps found. Exiting.")
    exit()

print("Merged Data Head:\n", df.head())
print(f"Merged data shape: {df.shape}")


#
# --- 2. Define Custom DILATE-inspired Loss Function (MODIFIED FOR MULTI-TARGET) ---
#
def create_dilate_loss(alpha=0.5):
    """
    Retains the DILATE structure, needed for model loading.
    """
    def dilate_loss(y_true_flat, y_pred_flat):
        mse = K.mean(K.square(y_true_flat - y_pred_flat), axis=-1)

        y_true_shifted = K.concatenate([K.expand_dims(y_true_flat[:, 0], -1), y_true_flat[:, :-1]], axis=-1)
        y_pred_shifted = K.concatenate([K.expand_dims(y_pred_flat[:, 0], -1), y_pred_flat[:, :-1]], axis=-1)

        y_true_diff = y_true_flat - y_true_shifted
        y_pred_diff = y_pred_flat - y_pred_shifted

        shape_loss = K.mean(K.square(y_true_diff - y_pred_diff), axis=-1)

        loss = alpha * mse + (1.0 - alpha) * shape_loss
        return loss

    dilate_loss.__name__ = 'dilate_loss_multi_target'
    return dilate_loss
#
#

#
# --- 3. Data Preprocessing Function (MODIFIED for MULTI-TARGET) ---
#
def prepare_data(df_sequence, n_past, n_future, target_cols, scaler_features, scaler_target):
    """
    Creates windows (X, y) for a single data sequence using pre-fitted scalers.
    Now handles multiple targets in y, flattened to (n_future * n_targets,).
    """
    print(f"  > Preprocessing sequence of length {len(df_sequence)}...")

    features = df_sequence.columns
    targets = df_sequence[target_cols]

    scaled_features = scaler_features.transform(df_sequence[features])
    scaled_targets = scaler_target.transform(targets) # Shape: (n_rows, n_targets)

    X, y = [], []
    y_start_timestamps = []

    for i in range(n_past, len(scaled_features) - n_future + 1):
        X.append(scaled_features[i - n_past:i, 0:df_sequence.shape[1]])
        
        # Output (y): N_FUTURE timesteps of ALL targets, flattened
        future_targets = scaled_targets[i:i + n_future, :]
        y.append(future_targets.flatten()) 
        
        y_start_timestamps.append(df_sequence.index[i])

    X, y = np.array(X), np.array(y)
    y_start_times = np.array(y_start_timestamps)

    print(f"    ...Created {X.shape[0]} windows from this sequence.")

    return X, y, y_start_times


# --- Main Execution ---
if __name__ == '__main__':
    #
    # --- 4. Configuration (MODIFIED) ---
    #
    N_PAST_HOURS = 24
    N_FUTURE_HOURS = 12       # Must match the '12future' model

    # --- KEY CHANGE: Define BOTH target columns ---
    TARGET_COLUMNS = ['water_level_oldporttmp', 'water_level_ftmyers']
    N_TARGETS = len(TARGET_COLUMNS)
    
    TARGET_COLUMN_STPETE = 'water_level_oldporttmp' 

    SAMPLE_INDEX = 0          # Which sample from the test set to predict

    # Check if target columns exist
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            print(f"Error: Target column '{col}' not in merged DataFrame.")
            print(f"Available columns: {df.columns.to_list()}")
            exit()

    # =========================================================================
    # --- 5. Grouping and Preprocessing (MODIFIED for Multi-Target Scaling) ---
    # =========================================================================

    # --- 5.1. Fit Scalers ONCE on all data ---
    print("Fitting scalers on entire dataset...")

    feature_cols = [col for col in df.columns if col not in ['group']] 

    print(f"Using {len(feature_cols)} features (from both stations).")
    print(f"Using {N_TARGETS} targets: {TARGET_COLUMNS}")

    target_col_index_oldporttmp = df.columns.get_loc(TARGET_COLUMN_STPETE)
    target_col_index_ftmyers = df.columns.get_loc('water_level_ftmyers')


    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_features.fit(df[feature_cols])

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(df[TARGET_COLUMNS])
    print("Scalers fitted.")

    # --- 5.2. Group Data into Sequences (Unchanged) ---
    print("Grouping data into sequences based on 1-hour gaps...")
    df_with_col = df.reset_index()
    time_diff = df_with_col['timestamp'].diff()
    sequence_id = (time_diff > pd.Timedelta('1 hour')).cumsum()
    groups = df_with_col.groupby(sequence_id)
    print(f"Found {len(groups)} total sequences.")

    # --- 5.3. Create Windows for each valid sequence (MODIFIED to pass TARGET_COLUMNS) ---
    all_X, all_y, all_y_start_times = [], [], []
    MIN_SEQ_LEN = N_PAST_HOURS + N_FUTURE_HOURS

    print(f"Windowing sequences (min length {MIN_SEQ_LEN} required)...")

    for seq_id, group_df in groups:
        if len(group_df) >= MIN_SEQ_LEN:
            group_df_indexed = group_df.set_index('timestamp')

            X_seq, y_seq, y_times_seq = prepare_data(
                group_df_indexed,
                N_PAST_HOURS,
                N_FUTURE_HOURS,
                TARGET_COLUMNS, 
                scaler_features,
                scaler_target
            )

            if X_seq.shape[0] > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)
                all_y_start_times.append(y_times_seq)
        else:
            print(f"  > Skipping sequence {seq_id} (length {len(group_df)} < {MIN_SEQ_LEN})")

    # --- 5.4. Stack all windows into one dataset ---
    if not all_X:
        print("\n--- CRITICAL ERROR ---")
        print(f"No valid sequences found with the minimum required length of {MIN_SEQ_LEN}. Exiting.")
        exit()

    X = np.vstack(all_X)
    y = np.vstack(all_y) 
    y_start_times = np.concatenate(all_y_start_times)

    print("\n--- Preprocessing Complete ---")
    print(f"Final data shapes:")
    print(f"  X shape: {X.shape}") 
    print(f"  y shape: {y.shape}") 
    print(f"  y_start_times shape: {y_start_times.shape}")

    # --- 5.5. Split Data (Unchanged logic) ---
    X_train, X_test, y_train, y_test, y_times_train, y_times_test = train_test_split(
        X, y, y_start_times, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}, y_times_test={y_times_test.shape}")


    #
    # --- 6. Get Timestamps for Plotting (Unchanged) ---
    #
    print("\n--- Getting corrected timestamps ---")
    forecast_start_time = y_times_test[SAMPLE_INDEX]
    print(f"Correct forecast start time for sample {SAMPLE_INDEX} is: {forecast_start_time}")
    try:
        forecast_start_index_in_df = df.index.get_loc(forecast_start_time)
    except KeyError:
        print(f"CRITICAL ERROR: The timestamp {forecast_start_time} was not found.")
        exit()

    forecast_datetimes = df.index[forecast_start_index_in_df : forecast_start_index_in_df + N_FUTURE_HOURS]
    history_start_index_in_df = forecast_start_index_in_df - N_PAST_HOURS
    history_datetimes = df.index[history_start_index_in_df : forecast_start_index_in_df]

    if len(history_datetimes) != N_PAST_HOURS or len(forecast_datetimes) != N_FUTURE_HOURS:
        print("Error: Could not retrieve valid history or forecast timestamps. Exiting.")
        exit()

    print(f"History timestamps from: {history_datetimes[0]} to {history_datetimes[-1]}")
    print(f"Forecast timestamps from: {forecast_datetimes[0]} to {forecast_datetimes[-1]}")

    #
    # --- 7. Load Pre-Trained Model (UNCHANGED) ---
    #
    print("\n--- Loading model from disk... ---")
    model_json_saved = f'gru_multi_target_2stations_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL_myers_oldport.json'
    weights_file = f'gru_multi_target_2stations_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL_myers_oldport.h5'

    dilate_loss_fn = create_dilate_loss(alpha=0.5) 
    try:
        json_file = open(model_json_saved, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, custom_objects={'dilate_loss_multi_target': dilate_loss_fn})
        loaded_model.load_weights(weights_file)
        loaded_model.compile(optimizer='adam', loss=dilate_loss_fn)
        print("Loaded model from disk successfully.")
    except FileNotFoundError:
        print(f"Error: Model files not found: {model_json_saved}, {weights_file}")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        traceback.print_exc()
        exit()

    #
    # --- 8. Make and Interpret Predictions (FIXED + HINDCAST ADDED) ---
    #
    print(f"\n--- Making prediction for sample {SAMPLE_INDEX} ---")
    
    sample_full_index = len(X_train) + SAMPLE_INDEX 

    # --- FUTURE prediction (Multi-Target) ---
    try:
        sample_input = X_test[SAMPLE_INDEX].reshape(1, N_PAST_HOURS, X_test.shape[2])
        predicted_scaled = loaded_model.predict(sample_input) # Shape: (1, 24)

        # FIX: Reshape the flattened output (1, 24) to the sequence shape (12, 2) 
        predicted_reshaped_temp = predicted_scaled.reshape(N_FUTURE_HOURS, N_TARGETS)
        predicted_unscaled = scaler_target.inverse_transform(predicted_reshaped_temp) * 3.28
        predicted_levels = predicted_unscaled # Shape: (12, 2)
        
        # Reshape and inverse transform the actual (true) values
        actual_reshaped_temp = y_test[SAMPLE_INDEX].reshape(N_FUTURE_HOURS, N_TARGETS)
        actual_unscaled = scaler_target.inverse_transform(actual_reshaped_temp) * 3.28
        actual_levels = actual_unscaled # Shape: (12, 2)
        
        for j, target_name in enumerate(TARGET_COLUMNS):
            print(f"\nForecast for {target_name} (starting {forecast_datetimes[0]}):")
            for i in range(N_FUTURE_HOURS):
                 print(f"  - {forecast_datetimes[i]}: Predicted={predicted_levels[i, j]:.2f}ft, Actual={actual_levels[i, j]:.2f}ft")

    except IndexError:
        print(f"Error: Sample index {SAMPLE_INDEX} is out of bounds for the test set (size {len(X_test)}).")
        exit()
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc()
        exit()
        
    # --- HINDCAST (PAST PREDICTION) LOGIC ---
    predicted_past_levels_seg1 = None
    predicted_past_levels_seg2 = None
    can_predict_past = False
    
    print("\n--- Generating Color-Coded Hindcast (2 segments) ---")
    try:
        idx_input_end = sample_full_index - 1 
        
        # 1. Hindcast Segment 2 (t-12 to t-1). Input X ends at t-13.
        idx_input_seg2 = idx_input_end - N_FUTURE_HOURS 

        if idx_input_seg2 >= 0:
            input_seg2 = X[idx_input_seg2].reshape(1, N_PAST_HOURS, X.shape[2])
            pred_scaled_seg2 = loaded_model.predict(input_seg2)
            
            pred_reshaped_temp = pred_scaled_seg2.reshape(N_FUTURE_HOURS, N_TARGETS)
            predicted_past_levels_seg2 = scaler_target.inverse_transform(pred_reshaped_temp) * 3.28
            can_predict_past = True
        
        # 2. Hindcast Segment 1 (t-24 to t-13). Input X ends at t-25.
        idx_input_seg1 = idx_input_end - N_PAST_HOURS 

        if idx_input_seg1 >= 0:
            input_seg1 = X[idx_input_seg1].reshape(1, N_PAST_HOURS, X.shape[2])
            pred_scaled_seg1 = loaded_model.predict(input_seg1)
            
            pred_reshaped_temp = pred_scaled_seg1.reshape(N_FUTURE_HOURS, N_TARGETS)
            predicted_past_levels_seg1 = scaler_target.inverse_transform(pred_reshaped_temp) * 3.28
            can_predict_past = True
            
    except Exception as e:
        print(f"\nCRITICAL ERROR during hindcast generation: {e}")
        traceback.print_exc()
        can_predict_past = False


    # --- GET ACTUAL HISTORY DATA FOR BOTH TARGETS ---
    sample_input_scaled_features = X_test[SAMPLE_INDEX]
    sample_input_UNSCALED_features = scaler_features.inverse_transform(sample_input_scaled_features) * 3.28
    
    history_levels_oldporttmp = sample_input_UNSCALED_features[:, target_col_index_oldporttmp]
    history_levels_ftmyers = sample_input_UNSCALED_features[:, target_col_index_ftmyers]
    
    #
    # --- 9. Plot the Sample Prediction (MODIFIED FOR SIMPLIFIED FUTURE PLOT) ---
    #
    print("\n--- Plotting sample prediction with history for BOTH stations ---")
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    
    # --- Define Plotting Segments ---
    SEGMENT_LENGTH = N_FUTURE_HOURS # 12 hours (N_PAST_HOURS = 2 * N_FUTURE_HOURS)

    # --- History Datetime Segments (ALWAYS available) ---
    hindcast_datetimes_seg1 = history_datetimes[:SEGMENT_LENGTH] # t-24 to t-13
    hindcast_datetimes_seg2 = history_datetimes[SEGMENT_LENGTH:] # t-12 to t-1

    # --- Y-Axis Limits Calculation (Unchanged) ---
    all_y_data_list = [
        history_levels_oldporttmp, history_levels_ftmyers,
        predicted_levels[:, 0], predicted_levels[:, 1],
        actual_levels[:, 0], actual_levels[:, 1]
    ]
    if predicted_past_levels_seg1 is not None:
         all_y_data_list.extend([predicted_past_levels_seg1[:, 0], predicted_past_levels_seg1[:, 1]])
    if predicted_past_levels_seg2 is not None:
         all_y_data_list.extend([predicted_past_levels_seg2[:, 0], predicted_past_levels_seg2[:, 1]])

    all_y_data = np.concatenate(all_y_data_list)
    y_min = np.nanmin(all_y_data)
    y_max = np.nanmax(all_y_data)
    y_padding = (y_max - y_min) * 0.1
    if y_padding == 0: y_padding = 0.5
    plot_y_min = y_min - y_padding
    plot_y_max = y_max + y_padding
    
    # --- Common X-Axis Formatting ---
    all_datetimes = history_datetimes.union(forecast_datetimes)
    plot_ticks = pd.date_range(start=all_datetimes[0], end=all_datetimes[-1], freq='4h')

    def plot_single_station(ax, station_index, title_suffix):
        """Helper function to draw the forecast/hindcast for one station."""
        
        station = TARGET_COLUMNS[station_index]
        current_history_levels = history_levels_oldporttmp if station_index == 0 else history_levels_ftmyers
        
        # --- History Level Segments ---
        history_levels_seg1 = current_history_levels[:SEGMENT_LENGTH]
        history_levels_seg2 = current_history_levels[SEGMENT_LENGTH:]
        
        # --- Gap closing point (Actual value at t=0) ---
        gap_closing_actual_level = actual_levels[0, station_index]

        # --- Plotting 1: Actual History (t-24 to t-13) ---
        ax.plot(hindcast_datetimes_seg1, 
                history_levels_seg1,
                marker='o', markersize=4, linestyle='--',
                label='Actual History (t-24 to t-13)', color='#1f77b4', zorder=5) # Default blue

        # --- Plotting 2: Actual History (t-12 to t-1) + Gap Closer ---
        ax.plot(hindcast_datetimes_seg2.union([forecast_datetimes[0]]), 
                np.append(history_levels_seg2, gap_closing_actual_level),
                marker='o', markersize=4, linestyle='--',
                label='Actual History (t-12 to t-1)', color='#ff7f0e', zorder=6) # Default orange/second color

        
        # --- Color-Coded Hindcast (Past Prediction) ---
        if predicted_past_levels_seg1 is not None and predicted_past_levels_seg2 is not None:
            # Hindcast Segment 1 (t-24 to t-13)
            ax.plot(hindcast_datetimes_seg1,
                    predicted_past_levels_seg1[:, station_index],
                    marker='x', markersize=5, linestyle=':',
                    linewidth=2,
                    label='Hindcast (t-24 to t-13)', color='magenta', zorder=10)
                        
            # Hindcast Segment 2 (t-12 to t-1)
            ax.plot(hindcast_datetimes_seg2,
                    predicted_past_levels_seg2[:, station_index],
                    marker='x', markersize=5, linestyle=':',
                    linewidth=2,
                    label='Hindcast (t-12 to t-1)', color='green', zorder=10)


        # --- 🛑 Plotting 3: Actual Future (t+0 to t+11) - SINGLE COLOR (CYAN) ---
        ax.plot(forecast_datetimes,
                actual_levels[:, station_index],
                marker='o', markersize=4, linestyle='--',
                label='Actual Future (t+0 to t+11)', color='cyan', zorder=7)

        # --- 🛑 Plotting 4: Predicted Future (t+0 to t+11) - SINGLE COLOR (RED) ---
        ax.plot(forecast_datetimes,
                predicted_levels[:, station_index],
                marker='x', markersize=6, linestyle='-',
                label='Predicted Future (t+0 to t+11)', color='red', zorder=15)


        ax.axvline(forecast_datetimes[0], color='black', linestyle='--', label='Forecast Start', zorder=20)

        # --- Formatting ---
        ax.set_xlim(all_datetimes[0] - dt.timedelta(hours=1),
                    all_datetimes[-1] + dt.timedelta(hours=1))

        ax.set_xticks(plot_ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        ax.set_title(f'Water Level Forecast for {station} {title_suffix}', fontsize=14)
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Water Level (feet)')
        ax.set_ylim(plot_y_min, plot_y_max)
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True)
        return ax

    try:
        # ------------------------------------
        # 1. SAVE COMBINED PLOT
        # ------------------------------------
        
        plt.close('all')
        fig, axes = plt.subplots(N_TARGETS, 1, figsize=(15, 7 * N_TARGETS)) 
        
        for j, ax in enumerate(axes):
            plot_single_station(ax, j, f'(Sample {SAMPLE_INDEX})')
        
        fig.autofmt_xdate()
        fig.suptitle(f'Multi-Station GRU Forecast and Hindcast - {station_name} - Start: {forecast_start_time}', 
                     fontsize=16, y=1.02)
        plt.tight_layout()
        combined_plot_filename = f'COMBINED_PLOT_idx{SAMPLE_INDEX}_{timestamp_str}_FINAL_VIEW_SIMPLE.png'
        fig.savefig(combined_plot_filename)
        print(f"Saved Combined Plot to: '{combined_plot_filename}'")
        
        # ------------------------------------
        # 2. SAVE SEPARATE PLOTS
        # ------------------------------------
        
        for j, station in enumerate(TARGET_COLUMNS):
            plt.close('all')
            fig_single, ax_single = plt.subplots(figsize=(15, 7))
            
            # Draw the plot using the helper function
            plot_single_station(ax_single, j, f'(Sample {SAMPLE_INDEX})')
            
            fig_single.autofmt_xdate()
            plt.tight_layout()
            
            single_plot_filename = f'{station}_PLOT_idx{SAMPLE_INDEX}_{timestamp_str}_FINAL_VIEW_SIMPLE.png'
            fig_single.savefig(single_plot_filename)
            print(f"Saved {station} Plot to: '{single_plot_filename}'")

        print(f"\n********************************************************")
        print(f"All 3 simplified plot files have been saved.")
        print(f"********************************************************")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        traceback.print_exc()
