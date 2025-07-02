#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python

#https://www.geeksforgeeks.org/gated-recurrent-unit-networks/

#Import libraries
#
import os
import numpy as np
import pandas as pd
import csv
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam
#import mlnext # https://pypi.org/project/mlnext-framework/
from tensorflow.keras.layers import LSTM, Dense,Dropout
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from datetime import datetime
import datetime as dt


##################################
#Configuration of GPU Options
# TensorFlow wizardry
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.8

#sess = tf.Session()
#sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config));


#numpy: For handling numerical data and array manipulations.
#pandas: For data manipulation and reading datasets (CSV files).
#MinMaxScaler: For normalizing the dataset.
#tensorflow.keras.models and tensorflow.keras.layers: For building and training the LSTM model.
#Adam: An optimization algorithm used during training.
df = pd.read_csv('surge_training_datasets.csv', parse_dates=['timestamp'], index_col='timestamp')
#
data = df.sort_values(by='timestamp')
data =  df.tail(52513)

#Preparing Data for LSTM
#
# Define functions
# Function to reshape the input arrays
#



def lstm_data_transform(x_data, y_data, look_back, forecast_horizon):
    """ Changes data to the format for LSTM training
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

    # 2. Apply dilation to the difference (using tf.nn.atrous_conv2d or similar)
    #   - This part depends on the specific type of dilation you need.
    #   - For example, if you want to apply dilation to the difference,
    #     you can use tf.nn.atrous_conv2d or tf.nn.convolution with dilation_rate.
    #   - Example (using tf.nn.atrous_conv2d):
    #   - Note: Replace with your specific dilation implementation
    #   -  dilation_rate = 2
    #  dilated_difference = tf.nn.atrous_conv2d(
    #    value=tf.expand_dims(tf.expand_dims(difference, axis=0), axis=0),
    #    filters=tf.expand_dims(tf.expand_dims(tf.ones_like(difference), axis=0), axis=0),
    #    rate=dilation_rate,
    #    padding="SAME"
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
look_back = 720  # Number of past time steps to look at

forecast_horizon = 48 # Number of future time steps to predict

num_features = 6

# training set
# Reshape the original 2D data into 3D “sliding window” shape

(x_train_transformed,
 y_train_transformed) = lstm_data_transform(x_train_sc, y_train_sc, look_back,forecast_horizon)
assert x_train_transformed.shape[0] == y_train_transformed.shape[0]
# test set
(x_test_transformed,
 y_test_transformed) = lstm_data_transform(x_test_sc, y_test_sc, look_back,forecast_horizon)
assert x_test_transformed.shape[0] == y_test_transformed.shape[0]
#
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

model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(look_back, 6), dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(forecast_horizon)) # Output layer for multi-step prediction

#adam = optimizers.Adam(lr=0.001)
#model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.compile(optimizer=Adam(learning_rate=0.0001), loss=dilate_loss)
#
# Train the model with validation data
#
#x_train_transformed = np.asarray(x_train_transformed).astype('float32')
#y_train_transformed = np.asarray(y_train_transformed).astype('float32')
#x_test_transformed = np.asarray(x_test_transformed).astype('float32')
#y_test_transformed = np.asarray(y_test_transformed).astype('float32')
history = model.fit(x_train_transformed, y_train_transformed, epochs=100, \
          batch_size=32, validation_data=(x_test_transformed,y_test_transformed), callbacks=[early_stopping])
#
# 5. Make Predictions
# Generate some dummy input for prediction (last `look_back` data points)

# Saving history
np.save('loss_dropout_history.npy',history.history)
#
# Predict multiple steps ahead
predictions = model.predict(x_test_transformed[-look_back:])

#Create pandas dataframe for the predictions
df_observations = data[['water_level']].reset_index()
df_observations.rename(columns={'index': 'timestamp'}, inplace=True)
df_observations['timestamp'] = pd.to_datetime(df_observations['timestamp'], format='%Y-%m-%d %H %M %S')
df_observations['water_level'] = df_observations['water_level']*3.28084
# number of hours to plot for observation including hours corresponding to future projection (forecast_horizon)
num_hours_to_plot = forecast_horizon + 72
last_n_past_obs_values = df_observations.tail(num_hours_to_plot)
#last_n_past_obs_values = df_observations.tail(look_back)
periods=len(last_n_past_obs_values)
last_n_past_obs_values['timestamp'] = pd.date_range(start=last_n_past_obs_values['timestamp'].iloc[0] + pd.Timedelta(hours=1),freq='h', periods=len(last_n_past_obs_values))

thing_to_print = 'The last past ' + str(look_back) + ' observations'
print(thing_to_print, last_n_past_obs_values)
#
# prepare for future forecasts (forecast_horizon)
df_prediction = pd.DataFrame(columns=['timestamp', 'Prediction'])
all_predictions_rows = data.tail(look_back)
last_prediction_rows = all_predictions_rows.tail(forecast_horizon)
predictions = predictions[-forecast_horizon:]
print('predictions ', len(predictions))

#predictions = predictions[-look_back:]
##
print('last_prediction_rows: ',len(last_prediction_rows))
periods=len(last_prediction_rows)
range_data_datetime = pd.to_datetime(last_prediction_rows.index, format = '%Y-%m-%d %H %M %S')
print('range_data_datetime: ',range_data_datetime)
df_prediction['timestamp'] = pd.date_range(start= range_data_datetime[0] + pd.Timedelta(hours=1),freq='h', periods=len(last_prediction_rows))

last_prediction_rows['date'] = range_data_datetime

#print('range_data_datetime: ', range_data_datetime)
y_prediction =  np.array(predictions).reshape(-1, 1)
y_prediction = y_prediction[-forecast_horizon:]
y_prediction = scaler_y.inverse_transform(y_prediction)
# Inverse transform the predictions
#df_prediction['Prediction'] = y_prediction.flatten()
df_prediction['Prediction'] = y_prediction
#df_prediction['Prediction'] = df_prediction['Prediction'].tail(look_back)
print('df_prediction lenght: ', len(df_prediction['Prediction']))
df_prediction['Prediction']   = df_prediction['Prediction']*3.28084
print(df_prediction)


#range_from_last = last_look_back_rows.iloc[312:]
#print('range_from_last datetime ',range_from_last_datetime)
#df_prediction['timestamp'] = pd.date_range(start= range_data_datetime[-1] + pd.Timedelta(hours=1),freq='h', periods=len(last_prediction_rows))
last_prediction_rows['date'] = range_data_datetime
print('Prediction lentgh: ', len(predictions), len(x_train_transformed))
print(df_prediction)





# 6. Visualize Results

fig, ax = plt.subplots(figsize=(12, 6))
#ax.plot(df_future['date'], df_future['Forecast'])
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.plot(last_n_past_obs_values['timestamp'],last_n_past_obs_values['water_level'],df_prediction['timestamp'], df_prediction['Prediction'])
plt.gcf().autofmt_xdate()

plt.xticks(rotation=45)
plt.xlabel("Time (MM DD HH)")
plt.ylabel("Water level in feet")
plt.title("Observations and forecasts of water level trends)")
look_back_period = '(lookback period = ' + str(look_back) + 'hours)'
future_projection = str(forecast_horizon) + ' hours prediction of water levels '
plt.legend(['Observed water level)',future_projection + look_back_period ], loc='upper right')
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
#plt.show()
patience_epochs = 5
filename = 'lstm_obs_prediction_ft_myers_' + str(patience_epochs) + '_' + str(look_back) + 'hours.png'
 # Set the name of the variable to plot
plt.savefig(filename) # Set the output file name


#Saving trained model
# serialize model to JSON
#model_json = model.to_json()
fileout = 'lstm_multi_steps' + str(look_back) + 'hours_model_json'
fileout = model.to_json()
model_json_saved = 'lstm_multi_steps_' + str(look_back) + 'hours_model.json'
with open(model_json_saved, "w") as json_file:
    json_file.write(fileout)
# serialize and weights to HDF5
weights_file = 'lstm_multi_steps_' + str(look_back) + 'hours_model.h5'
model.save_weights(weights_file)
print("Saved model to disk")
#
#
# load json and create model
json_file = open(model_json_saved, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_file)
print("Loaded model from disk")

loaded_model.compile(loss=dilate_loss, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
predictions =loaded_model.predict(x_train_transformed)
predictions = scaler_y.inverse_transform(predictions)
print('load the model and print prediction array')
print(predictions)

