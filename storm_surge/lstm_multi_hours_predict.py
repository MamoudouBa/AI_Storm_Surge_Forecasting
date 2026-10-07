#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from datetime import datetime
import datetime as dt

# --- 1. Data Loading and Parsing ---
# Load the dataset
#data = pd.read_csv('surge_training_datasets.csv')
data = pd.read_csv('202209_ft_myers_2022_09_surge.csv')
# Convert timestamp to datetime objects
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H %M %S')
# Set timestamp as the index
data.set_index('timestamp', inplace=True)
# Sort the dataframe by the index (time)
df = data.sort_index()

data_points = len(data)
print("Data Head:\n", df.head())


# Define custom loss function
def dilate_loss(y_true, y_pred):
    """
    Custom loss function (DILATE).
    Combines shape loss (MSE) and time loss (MAE).
    """
    shape_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    time_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    total_loss = shape_loss + time_loss
    return total_loss
###################

# --- 2. Data Preprocessing ---
def prepare_data(df, n_past, n_future, target_col):
    """
    Prepares data for the LSTM model.
    Scales features and target, then creates sequences.
    """
    print("Preprocessing data...")
    features = df.columns
    target = df[target_col]
    
    # Scale features
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(df[features])
    
    # Scale target
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))
    
    X, y = [], []
    # Create sequences
    for i in range(n_past, len(scaled_features) - n_future + 1):
        X.append(scaled_features[i - n_past:i, 0:df.shape[1]])
        y.append(scaled_target[i:i + n_future, 0])
        
    X, y = np.array(X), np.array(y)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    return X, y, scaler_features, scaler_target

if __name__ == '__main__':
    # --- Configuration ---
    N_PAST_HOURS = 24     # Number of past hours to use as input
    N_FUTURE_HOURS = 24   # Number of future hours to predict
    TARGET_COLUMN = 'water_level' # The column we are trying to predict

    # --- 2. Preprocess Data ---
    # Note: This scales the *entire* dataset, which is standard for model training.
    # For prediction, we only need the scalers.
    X, y, scaler_features, scaler_target = prepare_data(df, N_PAST_HOURS, N_FUTURE_HOURS, TARGET_COLUMN)

    # --- 3. Load Model ---
    model_json_saved = 'lstm_multi_steps_' + str(N_PAST_HOURS) + 'hours_model.json'
    weights_file = 'lstm_multi_steps_' + str(N_PAST_HOURS) + 'hours_model.h5'

    print(f"Loading model from {model_json_saved}...")
    try:
        json_file = open(model_json_saved, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        # Load model architecture with custom loss function
        loaded_model = model_from_json(loaded_model_json, custom_objects={'dilate_loss': dilate_loss})
        # Load model weights
        loaded_model.load_weights(weights_file)
        print(f"Loaded model '{model_json_saved}' and weights '{weights_file}'.")
    except FileNotFoundError:
        print(f"Error: Could not find model files ('{model_json_saved}' or '{weights_file}').")
        print("Please make sure the model .json and .h5 files are in the same directory.")
        exit()


    # --- 4. Make a Back-Test Prediction ---
    print(f"\n--- Making a back-test prediction on the last {N_FUTURE_HOURS} hours ---")

    # This is the 24-hour block of data we'll use as input
    # It takes data from (24+24)=48 hours ago up to 24 hours ago
    total_window_size = N_PAST_HOURS + N_FUTURE_HOURS
    input_data_slice = df.iloc[-total_window_size : -N_FUTURE_HOURS]

    # This is the 24-hour "ground truth" block we are trying to predict
    # It takes data from 24 hours ago to the most recent data point
    ground_truth_df = df.tail(N_FUTURE_HOURS)

    # 2. Check if you have enough data
    if len(input_data_slice) < N_PAST_HOURS:
        print(f"Error: Need at least {total_window_size} total hours to run this comparison.")
    else:
        print(f"Using data from {input_data_slice.index[0]} to {input_data_slice.index[-1]} to predict.")

        # 3. Scale the input data using the *already fitted* scaler
        scaled_input_data = scaler_features.transform(input_data_slice)

        # 4. Reshape the data for the model (1 sample, N_PAST_HOURS steps, n_features)
        n_features = scaled_input_data.shape[1]
        sample_input = scaled_input_data.reshape(1, N_PAST_HOURS, n_features)

        # 5. Predict
        predicted_scaled = loaded_model.predict(sample_input)

        # 6. Inverse transform the prediction using the *target scaler*
        predicted_water_levels = scaler_target.inverse_transform(predicted_scaled)

        # 7. Create pandas dataframe for the predictions
        print('Predictions shape (scaled): ', predicted_scaled.shape)
        print('Predictions shape (inverse transformed): ', predicted_water_levels.shape)


        # 8. Set timestamps for the prediction
        # We use the *actual* timestamps from the ground truth data
        prediction_timestamps = ground_truth_df.index
        
        # 9. Flatten the prediction array (from 2D to 1D)
        predicted_values_1d = predicted_water_levels.flatten()

        # 10. Create the final prediction DataFrame
        df_prediction = pd.DataFrame({
            'timestamp': prediction_timestamps,
            'Prediction': predicted_values_1d
        })

        print("\n--- Predicted Water Levels (Back-test) ---")
        print(df_prediction)

        # --- 5. Prepare Data for Plotting ---
        # Get all observations
        df_observations = df[['water_level']].reset_index()
        # Note: 'inplace=True' is modifying df_observations directly
        df_observations.rename(columns={'index': 'timestamp'}, inplace=True)

        # Convert water level to feet for plotting
        df_observations['water_level'] = df_observations['water_level'] * 3.28084
        print("\n--- Obs (in feet) ---")
        print(df_observations)

        # --- UPDATED ---
        # number of hours to plot (ONLY the last N_FUTURE_HOURS)
        num_hours_to_plot = N_FUTURE_HOURS
        
        # This gets ONLY the last 24 hours of actual observations
        last_n_past_obs_values = df_observations.tail(num_hours_to_plot)
        
        # Convert prediction to feet for plotting
        df_prediction['Prediction'] = df_prediction['Prediction'] * 3.28084
        print("\n--- Predictions (in feet) ---")
        print(df_prediction)

        # --- 6. Visualize Results ---
        # --- UPDATED ---
        # The blue line will show the last 24 hours of observations.
        # The red dashed line will plot on top of it, showing the prediction for that same 24-hour window.

        print("Generating plot...")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the historical observations for the last 24 hours
        ax.plot(last_n_past_obs_values['timestamp'],
                last_n_past_obs_values['water_level'],
                label='Observations (Last 24h)',
                color='blue')

        # Plot the back-tested predictions
        ax.plot(df_prediction['timestamp'],
                df_prediction['Prediction'],
                label=f'Prediction (Last {N_FUTURE_HOURS} hours)',
                color='red',
                linestyle='--')

        # Add labels and a title
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Water Level (feet)')
        ax.set_title('Water Level: Actual Observations vs. Back-Tested Prediction')

        ax.legend()
        plt.grid(True)
        plt.tight_layout() # Adjusts plot to prevent labels from overlapping

        # Save the plot
        fig.savefig('water_level_BACKTEST_24hr_overlay.png')
        print("Plot saved to 'water_level_BACKTEST_24hr_overlay.png'")


