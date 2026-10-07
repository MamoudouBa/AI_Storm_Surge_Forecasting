#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import numpy as np
import pandas as pd
import tensorflow as tf
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
# --- 1. Load Data (*** THIS SECTION IS NOW FIXED ***) ---
#
print("Loading data...")
# Load the data *without* setting the index first
#data = pd.read_csv('surge_training_datasets.csv')
data = pd.read_csv('test_set_2022_ft_myers_for_water_level.csv')

# --- THIS IS THE FIX ---
# Manually convert the 'timestamp' column using the exact format
# This will handle '2007-08-05 03 00 00' correctly
try:
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H %M %S')
except ValueError as e:
    print(f"Error parsing datetimes: {e}")
    print("Please check that your timestamp format matches '%Y-%m-%d %H %M %S'")
    exit()

# Now, set the *correct* DatetimeIndex
data.set_index('timestamp', inplace=True)
df = data.sort_values(by='timestamp')
print("Data Head (with correct DatetimeIndex):\n", df.head())
print(f"Data index type: {type(df.index)}") # This should now be <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
# --- END OF FIX ---


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
# --- 3. Data Preprocessing Function ---
#
def prepare_data(df, n_past, n_future, target_col):
    """Prepares data for the LSTM model."""
    print("Preprocessing data...")
    features = df.columns
    target = df[target_col]

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(df[features])

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

    X, y = [], []
    for i in range(n_past, len(scaled_features) - n_future + 1):
        X.append(scaled_features[i - n_past:i, 0:df.shape[1]])
        y.append(scaled_target[i:i + n_future, 0])

    X, y = np.array(X), np.array(y)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    return X, y, scaler_features, scaler_target


# --- Main Execution ---
if __name__ == '__main__':
    # Configuration -
    N_PAST_HOURS = 24
    N_FUTURE_HOURS = 12   # Must match the '5future' model
    TARGET_COLUMN = 'water_level'
    SAMPLE_INDEX = -7 

    # 1. Preprocess Data
    X, y, scaler_features, scaler_target = prepare_data(df, N_PAST_HOURS, N_FUTURE_HOURS, TARGET_COLUMN)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    # Get the timestamps for the plot
    pred_start_index_in_df = len(X_train) + N_PAST_HOURS + SAMPLE_INDEX
    # This 'forecast_datetimes' will now be a DatetimeIndex
    forecast_datetimes = df.index[pred_start_index_in_df : pred_start_index_in_df + N_FUTURE_HOURS]
    
    # 3. Load Pre-Trained Model
    print("\n--- Loading model from disk... ---")

    #model_json_saved = f'gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model.json'
    #weights_file = f'gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model.h5'
    model_json_saved = f'gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_OPTIMAL.json'
    weights_file = f'gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_OPTIMAL.h5'

    dilate_loss_fn = create_dilate_loss(alpha=0.5)

    try:
        json_file = open(model_json_saved, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json,
                                      custom_objects={'dilate_loss': dilate_loss_fn})
        loaded_model.load_weights(weights_file)
        print("Loaded model from disk")
        loaded_model.compile(optimizer='adam', loss=dilate_loss_fn)

    except FileNotFoundError:
        print(f"Error: Model files not found.")
        print(f"Tried to load: '{model_json_saved}' and '{weights_file}'")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()


    # 4. Make and Interpret a Prediction
    print(f"\n--- Making prediction for sample {SAMPLE_INDEX} ---")
    
    if len(X_test) > SAMPLE_INDEX:
        sample_input = X_test[SAMPLE_INDEX].reshape(1, N_PAST_HOURS, X_test.shape[2]) 
        predicted_scaled = loaded_model.predict(sample_input)
        predicted_water_levels = scaler_target.inverse_transform(predicted_scaled)*3.28
        
        actual_scaled = y_test[SAMPLE_INDEX].reshape(1, N_FUTURE_HOURS)
        actual_water_levels = scaler_target.inverse_transform(actual_scaled)*3.28

        print(f"Forecast for the next {N_FUTURE_HOURS} hours (starting {forecast_datetimes[0]}):")
        for i in range(N_FUTURE_HOURS):
            print(f"  - {forecast_datetimes[i]}: Predicted={predicted_water_levels[0][i]:.2f}m, Actual={actual_water_levels[0][i]:.2f}m")
    else:
        print(f"Error: Sample index {SAMPLE_INDEX} is out of bounds for the test set (size {len(X_test)}).")
        exit()


    # --- 5. Plot the Sample Prediction (with Datetimes) ---
    print("\n--- Plotting sample prediction ---")
    plot_filename = f'sample_forecast_comparison_idx{SAMPLE_INDEX}.png'
    
    print(f"Plotting with these x-axis datetimes (type: {type(forecast_datetimes)}):\n{forecast_datetimes}")

    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # This .to_pydatetime() call will now work because forecast_datetimes is a DatetimeIndex
        ax.plot(forecast_datetimes.to_pydatetime(), actual_water_levels[0], marker='o', linestyle='--', label='Actual')
        ax.plot(forecast_datetimes.to_pydatetime(), predicted_water_levels[0], marker='x', linestyle='-', label='Predicted')
        
        # Format the x-axis to show dates and times
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        
        # Auto-rotate date labels
        plt.gcf().autofmt_xdate() 
        
        ax.set_title(f'Water Level Forecast (Sample {SAMPLE_INDEX})')
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Water Level (feet)') 
        ax.legend()
        ax.grid(True)
        
        plt.savefig(plot_filename)
        print(f"Plot saved as '{plot_filename}'")
        
        # Optionally, uncomment to display
        # plt.show()
    
    except Exception as e:
        print(f"An error occurred while plotting: {e}")
