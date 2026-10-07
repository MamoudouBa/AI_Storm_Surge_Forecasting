# Developed with the assistance of the Gemini AI code assistant.
#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python


import numpy as np
import pandas as pd
import csv
import os
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.models import Sequential, model_from_json
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Custom loss functions
#
def lstm_gru_data_transform(x_data, y_data, num_steps = 1):
    """ Changes data to the format for GRU training (reshape to 3d)
for sliding window approach """
    # Prepare the list for the transformed data
    X, y = list(), list()
    # Loop of the entire data set
    for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps
        # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]
        # Get only the last element of the sequency for y
        #seq_y = y_data[end_ix]#ori end-----fking somw wrong
        seq_y = y_data[i]#first correct wtf
        # Append the list with sequencies
        X.append(seq_X)
        y.append(seq_y)
    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array


#
#
#
#create_dataset(): Prepares the dataset for time-series forecasting.
# It creates sliding windows of time_step length to predict the next time step.
# It is not use in the current version
def create_dataset(data,label, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        if i == 0: print(data[i])
        X.append(data[i:(i + time_step), 0])
        y.append(label[i + time_step, 0])
    return np.array(X), np.array(y)
#

# Define loss function
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
###################
#
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

# Opening input data
data = pd.read_csv('surge_training_datasets.csv', index_col='timestamp')
data = data.sort_values(by='timestamp')
data =  data.tail(52513)

# Retrieve the date from the index
# Retrieve the date from the index
print('******************** Data shape :', data.shape),
print(data.describe())
# scale the data
# Preprocessing Stages
#
#Splitting: train set 80% and test data 20%
# Label(true) data
#
#Splitting: train set 80% and test data 20%
# Label(true) data
y_data = data.water_level.values
#print(y_data)

# Reshape the inpu X array to 3d and y array to 2d

y_data =y_data.reshape(-1, 1)
#train_ind = int(0.8 * data.values.shape[0])
#x_train = data.values[:train_ind]
x_train = data.values
#x_test = data.values[train_ind:]
#y_train = y_data[:train_ind]
y_train = y_data
#y_test = y_data[train_ind:]
#print(data.values)
# scalers
scaler_x = StandardScaler()
scaler_y = StandardScaler()
# scaling
x_train_sc = scaler_x.fit_transform(x_train)
#x_test_sc = scaler_x.transform(x_test)
y_train_sc = scaler_y.fit_transform(y_train)
#y_test_sc = scaler_y.transform(y_test)
#
#
# Reshape the original 2D data into 3D ▒~@~\sliding window▒~@~] shape
num_steps = 3
num_features = 6
#x_shaped = np.reshape(X, newshape=(-1, num_steps, num_features))
# training set

# training set

(x_train_transformed,
 y_train_transformed) = lstm_gru_data_transform(x_train_sc, y_train_sc, num_steps=num_steps)
assert x_train_transformed.shape[0] == y_train_transformed.shape[0]
# test set
#
#
# load json and create model
json_file = open('/contrib/Mamoudou.Ba/gru_steps_3hours_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/contrib/Mamoudou.Ba/gru_steps_3hours_model.h5")
print("Loaded model from disk")
print(loaded_model.summary())
#loaded_model.compile(loss=dilate_loss, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
loaded_model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
predictions =loaded_model.predict(x_train_transformed)
#
# Prepare for plotting the last 48 observations

#Create pandas dataframe for the predictions
df_observations = data[['water_level']].reset_index()
df_observations.rename(columns={'index': 'timestamp'}, inplace=True)
df_observations['timestamp'] = pd.to_datetime(df_observations['timestamp'], format='%Y-%m-%d %H %M %S')
df_observations['water_level'] = df_observations['water_level']*3.28084
N = 720
last_n_past_obs_values = df_observations.tail(N)
periods=len(last_n_past_obs_values)
last_n_past_obs_values['timestamp'] = pd.date_range(start=last_n_past_obs_values['timestamp'].iloc[0] + pd.Timedelta(hours=1),freq='h', periods=len(last_n_past_obs_values))
#df_observations['date'] = df_observations['timestamp'].dt.date
#df_observations['hour'] = df_observations['timestamp'].dt.hour
# Prepare for plotting the last 48 observations
print('The last past 720 observations', last_n_past_obs_values)

# Prepare for plotting the last 48 predictions based on the trained model
#Create pandas dataframe for the predictions
#
#


# prepare for future forecasts
#
#predictions = predictions[-720:]
df_prediction = pd.DataFrame(columns=['timestamp', 'Forecast'])
all_predictions_rows = data.tail(52513)
last_720_prediction_rows = all_predictions_rows.tail(720)
predictions = predictions[-720:]

#range_from_last = last_720_rows.iloc[312:]
periods=len(last_720_prediction_rows)
range_data_datetime = pd.to_datetime(last_720_prediction_rows.index, format = '%Y-%m-%d %H %M %S')
#print('range_from_last datetime ',range_from_last_datetime)
df_prediction['timestamp'] = pd.date_range(start= range_data_datetime[0] + pd.Timedelta(hours=1),freq='h', periods=len(last_720_prediction_rows))
last_720_prediction_rows['date'] = range_data_datetime
print('Prediction lentgh: ', len(predictions), len(x_train_transformed))
#Reshape and inverse the predictions to the actual units (feets)
y_prediction = np.array(predictions).reshape(-1, 1)
y_prediction = scaler_y.inverse_transform(y_prediction)
# Prepare for plotting the last 48 observations
df_prediction['Prediction'] = y_prediction.flatten()
#df_future['date'] = range_from_last_datetime.date
df_prediction['date'] = range_data_datetime
#df_future['hour'] = range_from_last_datetime.hour
df_prediction['Prediction']   = df_prediction['Prediction']*3.28084
print(df_prediction)
#
#
#
early_stopping = EarlyStopping(
    monitor='val_loss',  # Or another metric like 'val_accuracy'
    min_delta=0,  # Minimum change in monitored quantity to qualify as an improvement
    patience=2,   # Number of epochs to wait before stopping if no improvement
    verbose=1,    # Print messages when training stops
    mode='min'    # 'min' for minimization, 'max' for maximization
    # restore_best_weights=True  # Restore the model weights from the epoch with the best monitored value (optional)
)
#
early_stopping = EarlyStopping(
    monitor='val_loss',  # Or another metric like 'val_accuracy'
    min_delta=0,  # Minimum change in monitored quantity to qualify as an improvement
    patience=2,   # Number of epochs to wait before stopping if no improvement
    verbose=1,    # Print messages when training stops
    mode='min'    # 'min' for minimization, 'max' for maximization
    # restore_best_weights=True  # Restore the model weights from the epoch with the best monitored value (optional)
)

#Slacle the data
# scaling
#predictions = scaler_y.fit_transform(predictions)
#
# generate the training sequences
n_forecast = 1
n_lookback = 120
#
#Input data preparation
X = []
Y = []

for i in range(n_lookback, len(predictions) - n_forecast + 1):
    X.append(predictions[i - n_lookback: i])
    Y.append(predictions[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)
print('X and Y shapes: ', X.shape, Y.shape)
# train the model
tf.random.set_seed(0)
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(GRU(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
#
#
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
#model.compile(optimizer=Adam(learning_rate=0.001), loss= dilate_loss)
#
#Training the Model
history = model.fit(X, Y, epochs=100, \
          batch_size=128, validation_split=0.2, verbose=1, callbacks=[early_stopping])


#
#print(predictions.shape)
#print(predictions)
# generate the training sequences for future forecasts
# generate the multi-step forecasts
n_future = 48
y_future = []


print('lentgh o X and Y: ', len(X), len(Y))
# Use the the last 48th values as imput for future forecasts

#row_requested = data.iloc[52585:52586]
#print('Printing X and Y values')
#print(' x and y shapes: ', row_requested)
#print(row_requested)
#X = row_requested.values
#Y = row_requested['water_level'].values
#Reshape to two dimensions
x_pred = X[-552:, :, :]  # last observed input sequence
y_pred = Y[-552:, :, :]         # last observed target value
x_pred = x_pred[0]
y_pred = y_pred[0]
x_pred = x_pred.reshape(1,n_lookback,1)
y_pred = y_pred.reshape(1,1,1)
#y_pred = y_pred.reshape(1, 1, 1)
#print(x_pred)
print('************** y-pred ****************')
#print(y_pred)
#print('last_48_past_obs_values: ', last_n_past_obs_values.tail(48))

print('x_pred, y_pred shapes:  ', x_pred.shape, y_pred.shape)

#GRU and GRU accesp three dimension arrays, so reshaping now the 2d into 3d (sample, number of feature, number of values)
for i in range(n_future):

    # feed the last forecast back to the model as an input
    x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

    # generate the next forecast
    y_pred = model.predict(x_pred)
    print('Inside forecast loop: ',x_pred.shape, y_pred)
    # save the forecast
    y_future.append(y_pred.flatten()[0])

    # feed the last forecast back to the model as an input
# transform the forecasts back to the original scale
y_future = np.array(y_future).reshape(-1, 1)
y_future = scaler_y.inverse_transform(y_future)
df_past = data[['water_level']].reset_index()
df_past.rename(columns={'index': 'timestamp'}, inplace=True)
df_past['timestamp'] = pd.to_datetime(df_past['timestamp'], format='%Y-%m-%d %H %M %S')
#
# prepare for future forecasts
#
df_future = pd.DataFrame(columns=['timestamp', 'Forecast'])
last_720_rows = data.tail(720)
range_from_last = last_720_rows.iloc[672:]
periods=len(range_from_last)
range_from_last_datetime = pd.to_datetime(range_from_last.index, format = '%Y-%m-%d %H %M %S')
#print('range_from_last datetime ',range_from_last_datetime)
df_future['timestamp'] = pd.date_range(start= range_from_last_datetime[0] + pd.Timedelta(hours=1),freq='h', periods=len(range_from_last))
range_from_last['date'] = range_from_last_datetime
# Prepare for plotting the last 48 observations
df_future['Forecast'] = y_future.flatten()
#df_future['date'] = range_from_last_datetime.date
df_future['date'] = range_from_last_datetime
#df_future['hour'] = range_from_last_datetime.hour
df_future['Forecast']   = df_future['Forecast']*3.28084
print(df_future)

#df_future.index = pd.to_datetime(df_future.index)
#string_index = df_future.index.strftime('%Y-%m-%d %H:%M:%S')

#data.index = pd.to_datetime(data.index)
#string_index = data.index.strftime('%Y-%m-%d %H:%M:%S')
#fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
fig, ax = plt.subplots(figsize=(12, 6))
#ax.plot(df_future['date'], df_future['Forecast'])
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.plot(last_n_past_obs_values['timestamp'],last_n_past_obs_values['water_level'],df_prediction['timestamp'], df_prediction['Prediction'])
#ax.plot(last_n_past_obs_values['timestamp'],last_n_past_obs_values['water_level'],df_prediction['timestamp'], df_prediction['Prediction'], df_future['timestamp'], df_future['Forecast'])
plt.gcf().autofmt_xdate()

plt.xticks(rotation=45)
plt.xlabel("Time (MM DD HH)")
plt.ylabel("Water level in feet")
plt.title("Observations and forecasts of water level trends)")
plt.legend(['Observed water level)','Predicted water levels (lookback period = 3 hours)'], loc='lower center')
#plt.legend(['Observed water level)','Past predicted water levels', 'Future predicted water levels'], loc='lower center')

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
#plt.show()
filename = 'gru_obs_prediction_ft_myers_num_steps_3hours.png'
 # Set the name of the variable to plot
plt.savefig(filename) # Set the output file name

#print(df_past)
