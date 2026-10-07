# Developed with the assistance of the Gemini AI code assistant.
#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
#Imports

#
#https://www.geeksforgeeks.org/gated-recurrent-unit-networks/

#Import libraries
#
import numpy as np
import pandas as pd
import csv
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#import mlnext # https://pypi.org/project/mlnext-framework/
# Opening input data
df = pd.read_csv('surge_training_datasets', index_col='timestamp')

#Preprocessing the Data
#
# Function to reshape the input arrays
#
"""

https://medium.com/data-science/how-to-reshape-data-and-do-regression-for-time-series-using-lstm-133dad96cd00#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjgyMWYzYmM2NmYwNzUxZjc4NDA2MDY3OTliMWFkZjllOWZiNjBkZmIiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMTYxNTcxODc3ODE2MTU1ODcyMzIiLCJoZCI6Im5vYWEuZ292IiwiZW1haWwiOiJtYW1vdWRvdS5iYUBub2FhLmdvdiIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYmYiOjE3NDM0MjUwOTQsIm5hbWUiOiJNYW1vdWRvdSBCYSAtIE5PQUEgRmVkZXJhbCIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NJY2FsV1JldUZ5TXNXVmp3QkhVRlJYT1RqR0RBeTdzVVJfZ2VBNmxVU1RrOVM3dFEwej1zOTYtYyIsImdpdmVuX25hbWUiOiJNYW1vdWRvdSIsImZhbWlseV9uYW1lIjoiQmEgLSBOT0FBIEZlZGVyYWwiLCJpYXQiOjE3NDM0MjUzOTQsImV4cCI6MTc0MzQyODk5NCwianRpIjoiYzA4OTE0OTgyOWU3ZjQ4ODQ5NzA1NDkzZGUxNzRjNzFjYWI5ZDdhMSJ9.NNfCSi2tfyn5red-TXPbzbFZjwrGlv2XdLNUdNEIE_TOSd6ZDvV6uzb9SteB-GRxrGqrV36boSTmWuKp9tNepDb2Yp16jWvqR6NmPQ3CRnLjkK4DGFthSvNQPf_7dEgsPqMhHJzgyRzWyLqGccaaafPhBLLjeq766GquOtnpao6Q8To3CxPuz_eqyKJhz1zpIcIrhoiRn6a2PAJ17r3fOL3auAsg6QNM15N0XGaokj54FOS4HY_nW53x_Z5PXw6gAe8V4CuETo5TCLNNEcOWCzjD_5PHbppGhUUr3CSEWpSivMQbDBa7ICORXO5VGYnr3jy83dXBJNEPBx5TwlXnvw"""
#
# Custom loss functions
#
#
def lstm_gru_data_transform(x_data, y_data, num_steps = 1):
    """ Changes data to the format for GRU training
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
#create_dataset(): Prepares the dataset for time-series forecasting.
# It creates sliding windows of time_step length to predict the next time step.
# It is not use in the current version
#def create_dataset(data,label, time_step=1):
#    X, y = [], []
#    for i in range(len(data) - time_step - 1):
#        if i == 0: print(data[i])
#        X.append(data[i:(i + time_step), 0])
#        y.append(label[i + time_step, 0])
#    return np.array(X), np.array(y)


def create_dataset(data,label, n_lookback):
    X, Y = list(), list()
    for i in range(n_lookback, len(data) - n_forecast + 1):
        X.append(data[i - n_lookback: i])
        Y.append(label[i: i + n_forecast])
    X = np.array(X)
    Y = np.array(Y)

#
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

#
#Splitting: train set 80% and test data 20%
# Label(true) data
y_data = df.water_level.values
#print(y_data)

# Reshape the inpu X array to 3d and y array to 2d

y_data =y_data.reshape(-1, 1)
train_ind = int(0.8 * df.values.shape[0])
x_train = df.values[:train_ind]
x_val = df.values[train_ind:]
y_train = y_data[:train_ind]
y_val = y_data[train_ind:]
#print(df.values)
# scalers
scaler_x = StandardScaler()
scaler_y = StandardScaler()
# scaling
x_train_sc = scaler_x.fit_transform(x_train)
x_val_sc = scaler_x.fit_transform(x_val)
y_train_sc = scaler_y.fit_transform(y_train)
y_val_sc = scaler_y.fit_transform(y_val)
#
# generate the training sequences

print('****** xtrain shape : ', x_train_sc.shape, x_val_sc.shape)
n_forecast = 1
n_lookback = 6
X = []
Y = []
for i in range(n_lookback, len(y_train_sc) - n_forecast + 1):
    X.append(x_train_sc[i - n_lookback: i])
    Y.append(y_train_sc[i: i + n_forecast])
X = np.array(X)
Y = np.array(Y)

X_val = []
Y_val = []


for i in range(n_lookback, len(y_val_sc) - n_forecast + 1):
    X_val.append(x_val_sc[i - n_lookback: i])
    Y_val.append(y_val_sc[i: i + n_forecast])
X_val = np.array(X)
Y_val = np.array(Y)


#
# Reshape the original 2D data into 3D “sliding window” shape
#num_steps = 90
#num_features = 6
#x_shaped = np.reshape(X, newshape=(-1, num_steps, num_features))
# training set

# training set
#(x_train_transformed,
# y_train_transformed) = lstm_gru_data_transform(X, Y, num_steps=num_steps)
#assert x_train_transformed.shape[0] == y_train_transformed.shape[0]
# test set
#(x_val_transformed,
#y_val_transformed) = lstm_gru_data_transform(x_val_sc, y_val_sc, num_steps=num_steps)
#assert x_train_transformed.shape[0] == y_train_transformed.shape[0]
#


#
#X.reshape(): Reshapes the input data to fit the expected shape for the GRU which is 3D: [samples, time steps, features].
#5. Building the GRU Model
# train the model
#model.add(GRU(50, activation='tanh', input_shape=(x_train_transformed.shape[1], x_train_transformed.shape[2]), dropout=0.2, recurrent_dropout=0.2))
#tf.random.set_seed(0)
#model = Sequential()
#model.add(GRU(50, activation='tanh', return_sequences=True,input_shape=(X.shape[1], X.shape[2]), dropout=0.2, recurrent_dropout=0.2))
#model.add(GRU(units=50, return_sequences=False))
#model.add(Dense(25))
#model.add(Dense(1))
# train the model


#tf.random.set_seed(0)

model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(GRU(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

print('***** (X.shape[1], (X.shape[2]',X.shape[1], X.shape[2])
#Compile the model
#model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.compile(optimizer=Adam(learning_rate=0.0001), loss=dilate_loss)

#
#Training the Model
print(x_train_sc,x_val_sc.shape,y_val_sc.shape)


history = model.fit(X, Y, epochs=2, \
          batch_size=32, validation_data=(X_val, Y_val))
#
#Saving trained model


# generate the multi-step forecasts
n_future = 12
y_future = []
x_pred = X[-1:, :, :] #last observed input sequence
y_pred = Y[-1] #last observed target value
print('Y_pred:  ', y_pred)

ihour = 0
print(y_pred)
y_pred =y_pred.reshape(-1, 1)
print(x_pred.shape, y_pred.shape)

for i in range(n_future):
# feed the last forecast back to the model as an input
   #x_pred = np.append(x_pred[:, 1:, :],x_pred[:, 1:, :], axis=1)
   x_pred = np.append(x_pred[:, 1:, :],y_pred.shape[1, 1, 1], axis=1)
# generate the next forecast
   #y_pred.reshape(1, 1, 1)
   print('xpred shape: ', x_pred.shape)
   y_pred = model.predict(x_pred)
   #print(ihour, x_pred.shape,Y[-1], y_pred)
   ihour += 1
# save the forecast
   y_future.append(y_pred.flatten()[0])
# transform the forecasts back to the original scale
y_future = np.array(y_future).reshape(-1, 1)
y_future = scaler_y.inverse_transform(y_future)
print('The 12 forecasts: ')
print(y_future)
# organize the results in a data frame
# Convert datetime string to dateime object
#print(df.index[-1])
# organize the results in a data frame
df_past = df[['water_level']].reset_index()
df_past.rename(columns={'index': 'timestamp'}, inplace=True)
df_past['timestamp'] = pd.to_datetime(df_past['timestamp'], format='%Y-%m-%d %H %M %S')
print(df_past['timestamp'])
print(df_past['timestamp'].iloc[-1])
#df_date = datetime.strptime(df_past['timestamp'], "%Y-%m-%d %H %M %S")

df_future = pd.DataFrame(columns=['timestamp', 'Forecast'])
print("strat time : ", df_past['timestamp'].iloc[-1])
periods=n_future
#df_future['timestamp'] = pd.date_range(start=df_past['timestamp'].iloc[-1] + pd.Timedelta(hours=1), periods=n_future)
df_future['timestamp'] = pd.date_range(start=df_past['timestamp'].iloc[-1] + pd.Timedelta(hours=1),freq='h', periods=n_future)
df_future['Forecast'] = y_future.flatten()
df_future['date'] = df_future['timestamp'].dt.date
df_future['hour'] = df_future['timestamp'].dt.hour
print(df_future['timestamp'])
print(df_future)

df_future.to_csv('/contrib/Mamoudou.Ba/ft_myers_forecast.csv', index=False, mode='a', float_format='%g')
# plot the data
#

#df_past['date'] = df_past['timestamp'].dt.date
#df_past['hour'] = df_past['timestamp'].dt.hour
#last_12_rows = df_past.tail(96)
#fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
#print("obs dates:  ",last_12_rows['date'], last_12_rows['water_level'])
#x_axis = np.linspace(0.0, len(last_12_rows) , num = len(last_12_rows))
#ax.plot(x_axis, last_12_rows['water_level'])
#ax.plot(last_12_rows['date'], last_12_rows['water_level'], df_future['date'], df_future['Forecast'])
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#ax.xaxis.set_major_locator(mdates.HourLocator(interval=4)) # Show every hour
#date_format = mdates.DateFormatter('%Y-%m-%d')
#plt.gca().xaxis.set_major_formatter(date_format)
#plt.gcf().autofmt_xdate()
#x_axis = np.linspace(0.0, len(df_future) , num = len(df_future))
#ax.plot(x_axis, df_future['Forecast'])
#ax.plot(df_future['date'], df_future['Forecast'])
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#ax.xaxis.set_major_locator(mdates.HourLocator(interval=4)) # Show every hour

#date_format = mdates.DateFormatter('%Y-%m-%d')
#plt.gca().xaxis.set_major_formatter(date_format)
#plt.gcf().autofmt_xdate()
#plt.ylabel('Water level')
#plt.title('Plot of observed and 12 hour predictions')
#plt.show()
