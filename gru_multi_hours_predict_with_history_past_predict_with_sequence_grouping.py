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
# --- 1. Load Data ---
#
station_name = 'St Petersburg FL' 
print("Loading data...")
data = pd.read_csv('../storm_surge/test_set_2022_st_petersburg_fl_for_water_level.csv')
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
# --- 2. Define Custom DILATE-inspired Loss Function ---
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
    N_FUTURE_HOURS = 12       
    TARGET_COLUMN = 'water_level'
    SAMPLE_INDEX = 0          # Which sample from the test set to predict
    
    # Check assumptions for multi-color plot
    if N_PAST_HOURS % N_FUTURE_HOURS != 0 or N_PAST_HOURS // N_FUTURE_HOURS != 2:
        print(f"ERROR: Multi-color plot logic requires N_PAST_HOURS ({N_PAST_HOURS}) to be exactly 2x N_FUTURE_HOURS ({N_FUTURE_HOURS}).")
        exit()

    # =========================================================================
    # --- Preprocessing ---
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

    # --- 2. Group and Window Data ---
    print("Grouping data into sequences based on 1-hour gaps...")
    df_with_col = df.reset_index()
    time_diff = df_with_col['timestamp'].diff()
    sequence_id = (time_diff > pd.Timedelta('1 hour')).cumsum()
    groups = df_with_col.groupby(sequence_id)

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
                TARGET_COLUMN,
                scaler_features,
                scaler_target,
                FUTURE_GUIDANCE_COLS
            )

            if X_seq.shape[0] > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)
                all_y_start_times.append(y_times_seq)

    # --- 3. Stack all windows into one dataset ---
    if not all_X:
        print("\n--- CRITICAL ERROR ---")
        print(f"No valid sequences found with the minimum required length of {MIN_SEQ_LEN}.")
        exit()

    X = np.vstack(all_X)
    y = np.vstack(all_y)
    y_start_times = np.concatenate(all_y_start_times)

    print("\n--- Preprocessing Complete ---")
    print(f"Final training data shapes: X={X.shape}, y={y.shape}") 

    # 4. Split Data 
    X_train, X_test, y_train, y_test, y_times_train, y_times_test = train_test_split(
        X, y, y_start_times, test_size=0.2, random_state=42, shuffle=False
    )
    
    #
    # --- 5. Get Timestamps and Load Model ---
    #
    print("\n--- Getting corrected timestamps ---")
    
    # Calculate index into the final combined X array for SAMPLE_INDEX
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

    # --- Load Pre-Trained Model ---
    model_json_saved = f'../storm_surge/ft_myers_fl_2010_2021_gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL.json'
    weights_file = f'../storm_surge/ft_myers_fl_2010_2021_gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_grouped_OPTIMAL.h5'

    dilate_loss_fn = create_dilate_loss(alpha=0.5)
    try:
        json_file = open(model_json_saved, 'r')
        loaded_model = model_from_json(json_file.read(), custom_objects={'dilate_loss': dilate_loss_fn})
        json_file.close()
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
    # --- 6. Make and Interpret Predictions ---
    #
    print(f"\n--- Making prediction for sample {SAMPLE_INDEX} ---")
    
    # Past Prediction (Hindcast) setup
    can_predict_past = False
    predicted_past_levels = None 
    
    # We can only perform the multi-step hindcast if there are enough samples BEFORE the test sample.
    hindcast_start_index = sample_full_index - N_PAST_HOURS
    if hindcast_start_index >= 0:
        can_predict_past = True
        predicted_past_levels_list = []
        
        # We need 2 loops to cover 24 hours (24/12 = 2)
        for i in range(N_PAST_HOURS // N_FUTURE_HOURS):
            
            # The input index moves back 12 hours for each loop
            input_index = hindcast_start_index + (i * N_FUTURE_HOURS)
            
            # Input data is X[input_index]. X already has the correct N_PAST+N_FUTURE shape
            input_data = X[input_index].reshape(1, N_PAST_HOURS + N_FUTURE_HOURS, X.shape[2])
            
            pred_scaled = loaded_model.predict(input_data)
            pred_levels = scaler_target.inverse_transform(pred_scaled) * 3.28
            predicted_past_levels_list.append(pred_levels.flatten())
            
        predicted_past_levels = np.concatenate(predicted_past_levels_list)[:N_PAST_HOURS]
        print(f"Successfully generated past prediction. Final shape: {predicted_past_levels.shape}")

    else:
        print("Warning: Skipping multi-step hindcast as not enough historical data samples exist before the test sample.")


    # --- Making FUTURE prediction (Main forecast) ---
    try:
        print("--- Making FUTURE prediction (Main forecast) ---")
        # X_test[SAMPLE_INDEX] is the full, N_PAST + N_FUTURE sequence with guidance
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

    #
    # --- 7. Plot the Sample Prediction (MODIFIED for two-color hindcast) ---
    #
    print("\n--- Plotting sample prediction with history and two-color hindcast ---")
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    plot_filename = f'../storm_surge/FINAL_PLOT_idx{SAMPLE_INDEX}_{timestamp_str}_gru_st_petersburg.png'

    try:
        # --- GET HISTORY DATA ---
        sample_input_scaled_features = X_test[SAMPLE_INDEX][:N_PAST_HOURS, :] 
        sample_input_UNSCALED_features = scaler_features.inverse_transform(sample_input_scaled_features) * 3.28
        history_levels = sample_input_UNSCALED_features[:, target_col_index]

        # --- "Close the gap" fix for ACTUAL history ---
        forecast_start_actual_level = actual_water_levels[0][0]
        plot_history_datetimes = history_datetimes.union([forecast_datetimes[0]])
        plot_history_levels = np.append(history_levels, forecast_start_actual_level)

        # --- "Close the gap" fix for PREDICTED past (connecting the last hindcast point to the forecast start) ---
        plot_past_pred_levels = None 
        if can_predict_past:
            forecast_start_predicted_level = predicted_water_levels[0][0]
            # Append the first forecast point to the hindcast array for plotting the connecting line
            plot_past_pred_levels = np.append(predicted_past_levels, forecast_start_predicted_level) 
            print(f"Created {N_PAST_HOURS+1}-point 'plot_past_pred_levels' array.")

        # --- Y-Axis Limits (Calculation omitted for brevity, logic remains sound) ---
        all_y_data_list = [plot_history_levels, actual_water_levels[0], predicted_water_levels[0]]
        if can_predict_past: all_y_data_list.append(plot_past_pred_levels)
        all_y_data = np.concatenate(all_y_data_list)
        #y_min = -0.2
        #y_max = 9.0
        y_min, y_max = np.nanmin(all_y_data), np.nanmax(all_y_data)
        y_padding = max((y_max - y_min) * 0.1, 0.5)
        plot_y_min, plot_y_max = y_min - y_padding, y_max + y_padding

        # --- Plotting ---
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15, 7))

        # --- Multi-color Hindcast Plotting (The requested feature) ---
        if can_predict_past:
            segment_colors = ['magenta', 'orange'] 
            num_segments = N_PAST_HOURS // N_FUTURE_HOURS # Should be 2

            for i in range(num_segments):
                start_idx_data = i * N_FUTURE_HOURS
                end_idx_data = (i + 1) * N_FUTURE_HOURS
                
                # Get the 12 predicted hindcast points
                segment_levels = predicted_past_levels[start_idx_data:end_idx_data]
                
                # Get the corresponding 12 history datetimes
                segment_datetimes = history_datetimes[start_idx_data:end_idx_data]

                is_last_segment = (i == num_segments - 1)
                
                if is_last_segment:
                    # Append the connecting point to the final segment's data
                    segment_levels = np.append(segment_levels, forecast_start_predicted_level)
                    # Extend the datetimes to include the forecast start time
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
            
            # Add a generic dashed line to represent the overall hindcast, avoiding excessive legend entries
            ax.plot(plot_history_datetimes, plot_past_pred_levels, linestyle=':', color='black', linewidth=1, label='_nolegend_')


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
        ax.set_xlim(all_datetimes[0] - dt.timedelta(hours=1), all_datetimes[-1] + dt.timedelta(hours=1))

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

