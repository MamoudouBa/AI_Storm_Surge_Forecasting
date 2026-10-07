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
# <--- FIX: Corrected the data loading and parsing ---
# Load the data *without* parsing dates or setting the index immediately
data = pd.read_csv('surge_training_datasets.csv')

# EXPLICITLY convert the timestamp column using the correct format.
# This format was found in your plotting code later in the script.
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H %M %S')

# NOW, set the index
data.set_index('timestamp', inplace=True)

# Sort by the index (not by values)
df = data.sort_index()
# --- End of FIX ---

data_points = len(data)
print("Data Head:\n", df.head())


# Define loss function
# This function is not from Gemini
def dilate_loss(y_true, y_pred):
    # Calculate shape distortion loss (e.g., using Dynamic Time Warping or a similar method)
    shape_loss = tf.reduce_mean(tf.square(y_true - y_pred))  # Example: Mean Squared Error

    # Calculate temporal localisation loss (e.g., using a time-based error metric)
    time_loss = tf.reduce_mean(tf.abs(y_true - y_pred))  # Example: Mean Absolute Error

    # Combine the losses
    total_loss = shape_loss + time_loss  # Adjust weights as needed

    return total_loss
###################

# Gemini template
# --- 2. Data Preprocessing ---
def prepare_data(df, n_past, n_future, target_col):
    """Prepares data for the LSTM model."""
    print("Preprocessing data...")
    # Select features and target
    features = df.columns
    target = df[target_col]

    # Scale the data
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
    # Configuration
    N_PAST_HOURS = 24  # Use the last 24 hours of data to predict
    N_FUTURE_HOURS = 24   # Predict the next 5 hours
    TARGET_COLUMN = 'water_level' # The column we want to predict

    # 1. Get Data (Done above)

    # 2. Preprocess Data
    X, y, scaler_features, scaler_target = prepare_data(df, N_PAST_HOURS, N_FUTURE_HOURS, TARGET_COLUMN)

    # 3. Load Model
    model_json_saved = 'lstm_multi_steps_' + str(N_PAST_HOURS) + 'hours_model.json'
    weights_file = 'lstm_multi_steps_' + str(N_PAST_HOURS) + 'hours_model.h5'

    json_file = open(model_json_saved, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # <--- FIX: Added custom_objects to load your 'dilate_loss' function ---
    loaded_model = model_from_json(loaded_model_json, custom_objects={'dilate_loss': dilate_loss})

    # <--- FIX: Added loading the model weights. This is essential! ---
    loaded_model.load_weights(weights_file)
    print(f"Loaded model '{model_json_saved}' and weights '{weights_file}'.")


    # --- 4. Make a New Prediction from Latest Data ---
    print("\n--- Making a new prediction from the latest data ---")

    # 1. Get the last N_PAST_HOURS rows from your dataframe
    latest_data_df = df.tail(N_PAST_HOURS)

    # 2. Check if you have enough data
    if len(latest_data_df) < N_PAST_HOURS:
        print(f"Error: Need at least {N_PAST_HOURS} hours of data, but only found {len(latest_data_df)}.")
    else:
        # 3. Scale the input data using the *original* scaler_features
        scaled_input_data = scaler_features.transform(latest_data_df)

        # 4. Reshape the data for the model: (1, n_past_hours, n_features)
        n_features = scaled_input_data.shape[1] # Get number of features
        sample_input = scaled_input_data.reshape(1, N_PAST_HOURS, n_features)
        
        # 5. Predict
        predicted_scaled = loaded_model.predict(sample_input) # Shape (1, 5)

        # 6. Inverse transform the prediction
        predicted_water_levels = scaler_target.inverse_transform(predicted_scaled)

        # 7. Create pandas dataframe for the predictions
        print('predictions shape ', predicted_scaled.shape) # (1, 5)

        # Get the timestamp of the last *actual* observation from 'df'
        # <--- FIX: This will now be a proper Timestamp object, not a string ---
        last_obs_timestamp = df.index[-1]

        # 8. Get the number of prediction steps (e.g., 5)
        n_predictions = predicted_water_levels.shape[1]

        # 9. Create the future timestamps for the prediction
        # <--- FIX: This line will now work correctly ---
        prediction_timestamps = pd.date_range(
            start=last_obs_timestamp + pd.Timedelta(hours=1),
            periods=n_predictions,
            freq='h'
        )

        # 10. Flatten the prediction array (1, 5) -> (5,)
        predicted_values_1d = predicted_water_levels.flatten()

        # 11. Create the final prediction DataFrame
        df_prediction = pd.DataFrame({
            'timestamp': prediction_timestamps,
            'Prediction': predicted_values_1d
        })

        print("\n--- Predicted Water Levels ---")
        print(df_prediction)

        # --- 5. Prepare Data for Plotting ---
        df_observations = df[['water_level']].reset_index()
        # Note: df_observations['timestamp'] is already a datetime object,
        # but we'll rename the column just to be safe (as in your code)
        df_observations.rename(columns={'index': 'timestamp'}, inplace=True)
        
        # Convert water level to feet
        df_observations['water_level'] = df_observations['water_level'] * 3.28084
        
        # number of hours to plot
        num_hours_to_plot = N_FUTURE_HOURS + 72
        last_n_past_obs_values = df_observations.tail(num_hours_to_plot)

        print(f'The last past {num_hours_to_plot} observations')
        print(last_n_past_obs_values)

        # Convert prediction to feet
        df_prediction['Prediction'] = df_prediction['Prediction'] * 3.28084
        print("\n--- Predictions (in feet) ---")
        print(df_prediction)
        
        # --- 6. Visualize Results ---
        print("Generating plot...")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the historical observations
        ax.plot(last_n_past_obs_values['timestamp'],
                last_n_past_obs_values['water_level'],
                label='Observations',
                color='blue')

        # Plot the future predictions
        ax.plot(df_prediction['timestamp'],
                df_prediction['Prediction'],
                label=f'Prediction ({N_FUTURE_HOURS} hours)',
                color='red',
                linestyle='--') # make predictions a dashed line

        # Add labels and a title
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Water Level (feet)')
        ax.set_title('Water Level Observations and Predictions')

        # Add a legend to tell the lines apart
        ax.legend()

        # Improve layout
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        fig.savefig('water_level_prediction.png')
        print("Plot saved to 'water_level_prediction.png'")
