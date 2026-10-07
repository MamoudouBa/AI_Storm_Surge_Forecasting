#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python

#https://www.geeksforgeeks.org/gated-recurrent-unit-networks/

#Import libraries
#
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Required for date formatting

# --- START: Modified Data Loading and Preprocessing ---

# Opening input data
# Original observation data (contains actual water level and observed met features)
df_obs = pd.read_csv('/contrib/Mamoudou.Ba/storm_surge/st_petersburg_training_gfs_wl_data.csv', index_col='timestamp', parse_dates=True)
df_obs = df_obs.sort_values(by='timestamp')
# **Explicitly convert index to datetime with the correct format**
df_obs.index = pd.to_datetime(df_obs.index, format='%Y-%m-%d %H %M %S')


# NWP forecast data (contains forecast met features, water_level column will be ignored for its values here)
df_nwp_raw = pd.read_csv('/contrib/Mamoudou.Ba/storm_surge/st_petersburg_gfs_data.csv', index_col='timestamp', parse_dates=True)
df_nwp_raw = df_nwp_raw.sort_values(by='timestamp')
# **Explicitly convert index to datetime with the correct format**
df_nwp_raw.index = pd.to_datetime(df_nwp_raw.index, format='%Y-%m-%d %H %M %S') # Apply format here

# Ensure both dataframes cover the same time range for merging and alignment
common_timestamps = df_obs.index.intersection(df_nwp_raw.index)
df_obs = df_obs.loc[common_timestamps]
df_nwp_raw = df_nwp_raw.loc[common_timestamps]

# Define the features to be used for the model
# These are the *observed* meteorological features from df_obs for training
# and will be replaced by NWP *forecasts* during future prediction.
# water_level is the target.
features_for_model = ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust']
target_feature = 'water_level'

# Create the combined DataFrame for training (input features + target)
# Ensure the order of features is consistent for scaling and model input
df_training = df_obs[features_for_model + [target_feature]]

# Preprocessing the Data
# Label(true) data
y_data = df_training[target_feature].values.reshape(-1, 1) # Target is water_level

# Input features (X data)
x_data = df_training[features_for_model].values

# Splitting: train set 80% and test data 20%
train_ind = int(0.8 * x_data.shape[0])

x_train = x_data[:train_ind]
x_test = x_data[train_ind:]

y_train = y_data[:train_ind]
y_test = y_data[train_ind:]

# scalers
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# scaling
x_train_sc = scaler_x.fit_transform(x_train)
x_test_sc = scaler_x.transform(x_test)
y_train_sc = scaler_y.fit_transform(y_train)
y_test_sc = scaler_y.transform(y_test)

# Functions for data transformation and loss (unchanged)
def lstm_gru_data_transform(x_data_input, y_data_input, num_steps=100):
    """ Changes data to the format for LSTM training
    for sliding window approach """
    X, y = list(), list()
    for i in range(x_data_input.shape[0]):
        end_ix = i + num_steps
        if end_ix >= x_data_input.shape[0]:
            break
        seq_X = x_data_input[i:end_ix]
        seq_y = y_data_input[end_ix-1] # Predict the next step's water level based on the sequence
        X.append(seq_X)
        y.append(seq_y)
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array

def dilate_loss(y_true, y_pred):
    shape_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    time_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    total_loss = shape_loss + time_loss
    return total_loss

# Reshape the original 2D data into 3D “sliding window” shape
# Adjusted num_steps
num_steps = 48 # Changed from 102 to 48
num_features = len(features_for_model) # 5 features (air_temp, air_pressure, wind_speed, wind_direction, wind_gust)

# Training set
(x_train_transformed,
 y_train_transformed) = lstm_gru_data_transform(x_train_sc, y_train_sc, num_steps=num_steps)
assert x_train_transformed.shape[0] == y_train_transformed.shape[0]

# Test set
(x_test_transformed,
 y_test_transformed) = lstm_gru_data_transform(x_test_sc, y_test_sc, num_steps=num_steps)
assert x_test_transformed.shape[0] == y_test_transformed.shape[0]

# 5. Building the GRU Model
model = Sequential()
# Input shape adjusted to `num_features`
# Reduced GRU units and increased dropout
model.add(GRU(30, activation='tanh', input_shape=(num_steps, num_features), dropout=0.3, recurrent_dropout=0.3, return_sequences=False)) # GRU units 30, dropout 0.3
# Reduced Dense layer units
model.add(Dense(units=30, activation='relu')) # Dense units 30
model.add(Dense(units=1, activation='linear'))

#Compile the model
# Reduced learning rate and temporarily switched loss to 'mse'
model.compile(optimizer=Adam(learning_rate=0.00005), loss='mse') # Changed learning rate and loss

# Training the Model
# Added EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) # Patience increased to 15
history = model.fit(x_train_transformed, y_train_transformed, epochs=100, # Reduced epochs for faster testing
                    batch_size=32, validation_data=(x_test_transformed, y_test_transformed),
                    callbacks=[early_stopping]) # Added early stopping

# Saving trained model
np.save('/contrib/Mamoudou.Ba/loss_dropout_history.npy', history.history)

model_json = model.to_json()
with open("/contrib/Mamoudou.Ba/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("/contrib/Mamoudou.Ba/gru_dropout_model.h5")
print("Saved model to disk")

# Load json and create model (for immediate use or later)
json_file = open('/contrib/Mamoudou.Ba/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/contrib/Mamoudou.Ba/gru_dropout_model.h5")
print("Loaded model from disk")

# Compile the loaded model with the new learning rate and loss
loaded_model.compile(loss='mse', optimizer=Adam(learning_rate=0.00005), metrics=['accuracy']) # Changed loss and learning rate

# --- END: Modified Data Loading and Preprocessing ---

# --- START: New Prediction Function (No changes needed here based on overfitting adjustments) ---

def predict_future_water_level(model, initial_observations_df, nwp_future_forecast_df,
                               scaler_x, scaler_y, num_steps, features_for_model, target_feature):
    """
    Predicts future water levels using the trained GRU model and NWP forecasts.

    Args:
        model: The trained Keras GRU model.
        initial_observations_df: DataFrame of the last `num_steps` of actual observations
                                 (including water_level and met features) used to start the forecast.
                                 Must contain `features_for_model` and `target_feature` columns.
        nwp_future_forecast_df: DataFrame of NWP forecasts for the future horizon.
                                 Must contain `features_for_model` columns for the forecast period.
                                 The index should be the forecast timestamps.
        scaler_x: The StandardScaler fitted on the input features (meteorological).
        scaler_y: The StandardScaler fitted on the water level target.
        num_steps: The number of time steps used for input sequences.
        features_for_model: List of meteorological feature names used by the model.
        target_feature: Name of the water level target feature.

    Returns:
        A list of predicted water levels for the future horizon, and a DataFrame
        containing the full predicted sequence.
    """
    if len(initial_observations_df) < num_steps:
        raise ValueError(f"initial_observations_df must contain at least {num_steps} steps.")

    future_predictions_values = []
    # Initialize the current_input_sequence with the last `num_steps` of observed meteorological data
    current_input_sequence = initial_observations_df[features_for_model].values[-num_steps:].copy()

    # Create a DataFrame to store the full forecasted sequence (for visualization/debugging)
    forecast_timestamps = nwp_future_forecast_df.index
    full_forecast_df = pd.DataFrame(index=forecast_timestamps, columns=features_for_model + [target_feature])

    # Loop through each future forecast horizon
    for i in range(len(nwp_future_forecast_df)):
        # 1. Get NWP forecast for the current future step's meteorological features
        nwp_met_forecast_for_current_step = nwp_future_forecast_df.iloc[i][features_for_model].values

        # 2. Scale the current input sequence (meteorological features only)
        # The GRU model was trained on `num_features` (met features) sequences.
        scaled_input_sequence = scaler_x.transform(current_input_sequence)

        # 3. Reshape for model prediction
        x_input = scaled_input_sequence.reshape(1, num_steps, len(features_for_model))

        # 4. Predict the next water level (scaled)
        predicted_water_level_scaled = model.predict(x_input, verbose=0)[0, 0]

        # 5. Inverse transform the predicted water level
        predicted_water_level = scaler_y.inverse_transform([[predicted_water_level_scaled]])[0, 0]
        future_predictions_values.append(predicted_water_level)

        # 6. Prepare the input sequence for the next prediction
        #    Shift the sequence by one step.
        #    The *newest* entry in `current_input_sequence` will be the NWP met forecast for this *just predicted* time step.
        current_input_sequence = np.append(current_input_sequence[1:], [nwp_met_forecast_for_current_step], axis=0)

        # Store the full predicted row for the current timestamp
        full_forecast_df.loc[forecast_timestamps[i], features_for_model] = nwp_met_forecast_for_current_step
        full_forecast_df.loc[forecast_timestamps[i], target_feature] = predicted_water_level

    return future_predictions_values, full_forecast_df

# Example of how to use the prediction function:

# 1. Define the forecast horizon (e.g., next 24 hours)
N_future_horizons = 24
# The starting point for the forecast will be the end of your training/observation data.
# Find the latest timestamp in your observation data
last_obs_timestamp = df_obs.index[-1]
print(f"Last observation timestamp from df_obs: {last_obs_timestamp}") # Added more clarity

# The forecasts should ideally start immediately after `last_obs_timestamp`
nwp_future_forecast_start_time = last_obs_timestamp + pd.Timedelta(hours=1) # Assuming hourly data
print(f"Intended NWP future forecast start time: {nwp_future_forecast_start_time}") # Added more clarity

# Add checks for df_nwp_raw's time range
print(f"df_nwp_raw first timestamp: {df_nwp_raw.index.min()}")
print(f"df_nwp_raw last timestamp: {df_nwp_raw.index.max()}")

# Determine the actual number of horizons we can forecast based on available NWP data
# We are taking the last N_future_horizons from the df_nwp_raw for *simulation* purposes
# This ensures we always get available data and avoid KeyError.
actual_forecast_length = min(N_future_horizons, len(df_nwp_raw))

if actual_forecast_length == 0:
    print("Error: No NWP forecast data available in df_nwp_raw to simulate future forecasts.")
    print("Please ensure your df_nwp_raw file has data.")
    exit() # Exit if no data for simulation

nwp_future_forecast_df = df_nwp_raw.iloc[-actual_forecast_length:]

print(f"Simulating future forecasts using {actual_forecast_length} entries from df_nwp_raw.")
print(f"Simulated NWP forecast data range: {nwp_future_forecast_df.index.min()} to {nwp_future_forecast_df.index.max()}")

# --- START: Missing line re-added ---
# 2. Extract the initial observations needed for the first `num_steps` input sequence
# We need the last `num_steps` of observed data (both met features and water level)
initial_observations_for_forecast = df_obs.loc[df_obs.index <= last_obs_timestamp].tail(num_steps)
# --- END: Missing line re-added ---


# Perform the future prediction
print(f"\nStarting {actual_forecast_length}-hour water level forecast simulation...")
predicted_water_levels, full_predicted_sequence_df = predict_future_water_level(
    loaded_model,
    initial_observations_for_forecast,
    nwp_future_forecast_df,
    scaler_x,
    scaler_y,
    num_steps,
    features_for_model,
    target_feature
)

print("\nFuture Water Level Predictions:")
# Combine predicted values with their corresponding timestamps
forecast_output_df = pd.DataFrame({
    'timestamp': nwp_future_forecast_df.index[:len(predicted_water_levels)], # Use the actual timestamps from the simulated NWP data
    'predicted_water_level': predicted_water_levels
})
print(forecast_output_df)

# You can also inspect the full predicted sequence including met features
# print("\nFull Predicted Sequence (including NWP meteorological forecasts and predicted water levels):")
# print(full_predicted_sequence_df)

# Example: Plotting (if you have matplotlib)
# import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df_obs.index[-200:], df_obs[target_feature].tail(200), label='Observed Water Level (Last 200 hours)')
plt.plot(forecast_output_df['timestamp'], forecast_output_df['predicted_water_level'], label=f'Predicted Water Level (Next {actual_forecast_length} hours)', linestyle='--')
plt.xlabel('Timestamp')
plt.ylabel('Water Level')
plt.title('Water Level Forecast')
plt.legend()
plt.grid(True)
# plt.show()
filename = 'gru_gfs_obs_prediction_ft_myers_num_steps_' +str(num_steps) + 'hours.png'
 # Set the name of the variable to plot
plt.savefig(filename) # Set the output file name

