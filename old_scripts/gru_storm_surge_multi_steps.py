# Developed with the assistance of the Gemini AI code assistant.
#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python

#Import libraries
#
import os
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU, Dense,Dropout
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import datetime as dt


##################################
# Configuration of GPU Options (MODERNIZED)
#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- NEW TF2 GPU Configuration ---
# Try to find all physical GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to True for all GPUs
        # This prevents TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Optionally, you can still limit memory, but memory_growth is usually preferred.
        # This mimics your original 0.8 fraction, assuming one GPU
        # v_memory_limit = int(1024 * 8) # Example: 8GB. Adjust as needed.
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=v_memory_limit)])

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU found, running on CPU.")

# --- End of NEW TF2 GPU Configuration ---


#numpy: For handling numerical data and array manipulations.
#pandas: For data manipulation and reading datasets (CSV files).
#MinMaxScaler: For normalizing the dataset.
#tensorflow.keras.models and tensorflow.keras.layers: For building and training the GRU model.
#Adam: An optimization algorithm used during training.
df = pd.read_csv('surge_training_datasets.csv', parse_dates=['timestamp'], index_col='timestamp')
#
data = df.sort_values(by='timestamp')
# data =  df.tail(52513) # This line seems redundant if you use the whole 'df' anyway
data = df # Use the full sorted dataframe

#Preparing Data for GRU
#
# Define functions
# Function to reshape the input arrays
#


def gru_data_transform(x_data, y_data, look_back, forecast_horizon):
    """ Changes data to the format for GRU training
for sliding window approach """
    # Prepare the list for the transformed data
    #X, y = list(), list()
    X, y = [], []
    # Loop of the entire data set
    for i in range(x_data.shape[0] - look_back - forecast_horizon + 1):
        # compute a new (sliding window) index
        end_ix = i + look_back
        # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        X.append(x_data[i:end_ix])
        # Get only the last element of the sequency for y
        #seq_y = y_data[end_ix]#ori end-----fking somw wrong
        y.append(y_data[(i + look_back):(i + look_back + forecast_horizon)])#first correct wtf
    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array

# Create sequences for training (sliding window)
def create_dataset(data, look_back, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:(i + look_back)])
        y.append(data[(i + look_back):(i + look_back + forecast_horizon)])
    return np.array(X), np.array(y)

#
# Define loss function
#
#
#
def dilate_loss(y_true, y_pred):
        # Calculate shape distortion loss (e.g., using Dynamic Time Warping or a similar method)
        shape_loss = tf.reduce_mean(tf.square(y_true - y_pred))  # Example: Mean Squared Error

        # Calculate temporal localisation loss (e.g., using a time-based error metric)
        time_loss = tf.reduce_mean(tf.abs(y_true - y_pred))  # Example: Mean Absolute Error

        # Combine the losses
        total_loss = shape_loss + time_loss  # Adjust weights as needed

        return total_loss
###################
#
def custom_loss_with_dilation(y_true, y_pred, dilation_rate=2):
    """
    Custom loss function that incorporates dilation.

    Args:
        y_true: True values (ground truth).
        y_pred: Predicted values.
        dilation_rate: Dilation rate for the loss calculation.

    Returns:
        The calculated loss value.
    """

    # 1. Calculate the difference between true and predicted values
    difference = tf.subtract(y_true, y_pred)
    # 1. Calculate the difference between true and predicted values
    difference = tf.subtract(y_true, y_pred)

    # 2. Apply dilation to the difference (using tf.nn.atrous_conv2d or similar)
    #   - This part depends on the specific type of dilation you need.
    #   - For example, if you want to apply dilation to the difference,
    #     you can use tf.nn.atrous_conv2d or tf.nn.convolution with dilation_rate.
    #   - Example (using tf.nn.atrous_conv2d):
    #   - Note: Replace with your specific dilation implementation
    #   -  dilation_rate = 2
    #  dilated_difference = tf.nn.atrous_conv2d(
    #   value=tf.expand_dims(tf.expand_dims(difference, axis=0), axis=0),
    #   filters=tf.expand_dims(tf.expand_dims(tf.ones_like(difference), axis=0), axis=0),
    #   rate=dilation_rate,
    #   padding="SAME"
    #  )

    # 3. Calculate the loss (e.g., mean squared error, mean absolute error, etc.)
    #   - Example (using mean squared error):
    loss = tf.reduce_mean(tf.square(difference)) # Or tf.reduce_mean(tf.abs(difference)) for MAE

    return loss

#
#Splitting: train set 80% and test data 20%
#
# y array to 2d
# label( true obs) label
y_data = df.water_level.values
#print(y_data)
# Reshape the inpu X array to 3d and y array to 2d

y_data =y_data.reshape(-1, 1)

train_ind = int(0.8 * df.values.shape[0])
x_train = df.values[:train_ind]
x_test = df.values[train_ind:]
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

#
#
# Define window size (input sequence length) and prediction length
look_back = 8760  # Number of past time steps to look at

forecast_horizon = 48 # Number of future time steps to predict

num_features = 6

# training set
# Reshape the original 2D data into 3D “sliding window” shape

(x_train_transformed,
 y_train_transformed) = gru_data_transform(x_train_sc, y_train_sc, look_back,forecast_horizon)
assert x_train_transformed.shape[0] == y_train_transformed.shape[0]
# test set
(x_test_transformed,
 y_test_transformed) = gru_data_transform(x_test_sc, y_test_sc, look_back,forecast_horizon)
assert x_test_transformed.shape[0] == y_test_transformed.shape[0]
#
# Define the model
#
early_stopping = EarlyStopping(
    monitor='val_loss',  # Or another metric like 'val_accuracy'
    min_delta=0,  # Minimum change in monitored quantity to qualify as an improvement
    patience=5,   # Number of epochs to wait before stopping if no improvement
    verbose=1,    # Print messages when training stops
    mode='min'    # 'min' for minimization, 'max' for maximization
    # restore_best_weights=True  # Restore the model weights from the epoch with the best monitored value (optional)
)

#model.add(GRU(50, activation='tanh', unit_forget_bias=True, bias_initializer='zeros', \
model = Sequential()
model.add(GRU(50, activation='tanh', input_shape=(look_back, num_features), dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(GRU(50))
model.add(Dense(forecast_horizon)) # Output layer for multi-step prediction

#adam = optimizers.Adam(lr=0.001)
#model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.compile(optimizer=Adam(learning_rate=0.0001), loss=dilate_loss)
#
# Train the model with validation data
#

# --- DATA TYPE CONVERSION (FIX 1) ---
# Convert data to float32, which is preferred by GPUs
print(f"Original x_train dtype: {x_train_transformed.dtype}")
x_train_transformed = np.asarray(x_train_transformed).astype('float32')
y_train_transformed = np.asarray(y_train_transformed).astype('float32')
x_test_transformed = np.asarray(x_test_transformed).astype('float32')
y_test_transformed = np.asarray(y_test_transformed).astype('float32')
print(f"Converted x_train dtype to: {x_train_transformed.dtype}")
# --- END OF FIX 1 ---


history = model.fit(x_train_transformed, y_train_transformed, epochs=100, \
          batch_size=32, validation_data=(x_test_transformed,y_test_transformed), callbacks=[early_stopping])
#
# 5. Make Predictions (REWRITTEN)
#
print("
--- Making and Processing Predictions ---")

# Generate prediction using the *last* window from the test set
# x_test_transformed[-1:] has shape (1, look_back, num_features)
# Ensure prediction input is also float32
prediction_input = np.asarray(x_test_transformed[-1:]).astype('float32')
prediction_scaled = model.predict(prediction_input)
# prediction_scaled has shape (1, forecast_horizon)

# Inverse transform the prediction
# Reshape to (forecast_horizon, 1) for scaler
y_prediction_scaled = prediction_scaled.flatten().reshape(-1, 1)
y_prediction = scaler_y.inverse_transform(y_prediction_scaled)

# Rescale to feet (as done for observations)
y_prediction_unscaled = y_prediction.flatten() * 3.28084

# --- Prepare data for plotting ---

# 1. Create the base observations dataframe (scaled to feet)
df_observations = data[['water_level']].reset_index()
# df_observations.rename(columns={'index': 'timestamp'}, inplace=True) # Not needed if index_col='timestamp' was used
df_observations['timestamp'] = pd.to_datetime(df_observations['timestamp'], format='%Y-%m-%d %H %M %S')
df_observations['water_level'] = df_observations['water_level'] * 3.28084

# 2. Get the last 72 hours of *historical* data for the plot
# Find the last timestamp in the *entire* dataset
last_historical_timestamp = df_observations['timestamp'].iloc[-1]
history_start_time = last_historical_timestamp - pd.Timedelta(hours=71) # 72 points inclusive

# Select the last 72 hours
plot_history_data = df_observations[
    df_observations['timestamp'] >= history_start_time
].copy() # Use .copy() to avoid SettingWithCopyWarning

print(f"Plotting history from {plot_history_data['timestamp'].iloc[0]} to {plot_history_data['timestamp'].iloc[-1]}")


# 3. Create the prediction dataframe
# Predictions start 1 hour *after* the last historical timestamp
prediction_start_time = last_historical_timestamp + pd.Timedelta(hours=1)
prediction_timestamps = pd.date_range(
    start=prediction_start_time,
    periods=forecast_horizon,
    freq='h'
)

df_prediction = pd.DataFrame({
    'timestamp': prediction_timestamps,
    'Prediction': y_prediction_unscaled
})

print(f"Plotting prediction from {df_prediction['timestamp'].iloc[0]} to {df_prediction['timestamp'].iloc[-1]}")

#
# 6. Visualize Results (REWRITTEN)
#
print("--- Plotting Results ---")

fig, ax = plt.subplots(figsize=(15, 7)) # Made figure wider

# Plot the historical data (last 72 hours)
ax.plot(
    plot_history_data['timestamp'],
    plot_history_data['water_level'],
    label='Observed water level (last 72h)',
    color='blue'
)

# Plot the predicted data (next 'forecast_horizon' hours)
ax.plot(
    df_prediction['timestamp'],
    df_prediction['Prediction'],
    label=f'Forecasted water level ({forecast_horizon}h)',
    color='red',
    linestyle='--'
)

# --- Formatting ---
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gcf().autofmt_xdate() # Auto-format dates
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Water level in feet")
plt.title("Water Level: Observation and Forecast")
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout() # Adjust plot to prevent label overlap

# Set the output file name
patience_epochs = 5 # This variable isn't used, but was in original
filename = f'gru_obs_prediction_ft_myers_{forecast_horizon}h_{look_back}h_lookback_plot.png'
plt.savefig(filename) # Set the output file name
print(f"Saved plot to {filename}")
# plt.show() # Uncomment to display plot interactively


# --- Saving trained model ---
print("
--- Saving Model ---")
# serialize model to JSON
#model_json = model.to_json()
fileout = model.to_json()
model_json_saved = f'gru_multi_steps_{look_back}hours_model.json'
with open(model_json_saved, "w") as json_file:
    json_file.write(fileout)

# serialize and weights to HDF5
weights_file = f'gru_multi_steps_{look_back}hours_model.h5'
model.save_weights(weights_file)
print("Saved model to disk")


# --- Load model (example) ---
print("
--- Loading Model Example ---")
# load json and create model
json_file = open(model_json_saved, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_file)
print("Loaded model from disk")

# Re-compile the loaded model (necessary for it to be used)
loaded_model.compile(loss=dilate_loss, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
print("Model re-compiled.")

# Example prediction with loaded model
# prediction_input = np.asarray(x_test_transformed[-1:]).astype('float32')
# predictions_from_loaded = loaded_model.predict(prediction_input)
# predictions_from_loaded = scaler_y.inverse_transform(predictions_from_loaded.flatten().reshape(-1, 1))
# print("Prediction from loaded model (first 5 values):
", predictions_from_loaded[:5])
