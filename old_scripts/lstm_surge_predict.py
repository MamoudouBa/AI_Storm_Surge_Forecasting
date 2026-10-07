# Developed with the assistance of the Gemini AI code assistant.
#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python

#https://www.geeksforgeeks.org/gated-recurrent-unit-networks/

#Import libraries
#
import numpy as np
import pandas as pd
import csv
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
#import mlnext # https://pypi.org/project/mlnext-framework/
from tensorflow.keras.layers import LSTM, Dense


df = pd.read_csv('/contrib/Mamoudou.Ba/st_petersburg_surge_test.csv', parse_dates=['timestamp'], index_col='timestamp')

#Preprocessing the Data
#print(df)
#
#print("water level ",df.water_level.values[0])
print("df.values shape: ",df.values.shape)
#

"""

https://medium.com/data-science/how-to-reshape-data-and-do-regression-for-time-series-using-lstm-133dad96cd00#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjgyMWYzYmM2NmYwNzUxZjc4NDA2MDY3OTliMWFkZjllOWZiNjBkZmIiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMTYxNTcxODc3ODE2MTU1ODcyMzIiLCJoZCI6Im5vYWEuZ292IiwiZW1haWwiOiJtYW1vdWRvdS5iYUBub2FhLmdvdiIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYmYiOjE3NDM0MjUwOTQsIm5hbWUiOiJNYW1vdWRvdSBCYSAtIE5PQUEgRmVkZXJhbCIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NJY2FsV1JldUZ5TXNXVmp3QkhVRlJYT1RqR0RBeTdzVVJfZ2VBNmxVU1RrOVM3dFEwej1zOTYtYyIsImdpdmVuX25hbWUiOiJNYW1vdWRvdSIsImZhbWlseV9uYW1lIjoiQmEgLSBOT0FBIEZlZGVyYWwiLCJpYXQiOjE3NDM0MjUzOTQsImV4cCI6MTc0MzQyODk5NCwianRpIjoiYzA4OTE0OTgyOWU3ZjQ4ODQ5NzA1NDkzZGUxNzRjNzFjYWI5ZDdhMSJ9.NNfCSi2tfyn5red-TXPbzbFZjwrGlv2XdLNUdNEIE_TOSd6ZDvV6uzb9SteB-GRxrGqrV36boSTmWuKp9tNepDb2Yp16jWvqR6NmPQ3CRnLjkK4DGFthSvNQPf_7dEgsPqMhHJzgyRzWyLqGccaaafPhBLLjeq766GquOtnpao6Q8To3CxPuz_eqyKJhz1zpIcIrhoiRn6a2PAJ17r3fOL3auAsg6QNM15N0XGaokj54FOS4HY_nW53x_Z5PXw6gAe8V4CuETo5TCLNNEcOWCzjD_5PHbppGhUUr3CSEWpSivMQbDBa7ICORXO5VGYnr3jy83dXBJNEPBx5TwlXnvw"""
#
def lstm_gru_data_transform(x_data, y_data, num_steps=102):
    """ Changes data to the format for LSTM training
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
# Reshape the original 2D data into 3D “sliding window” shape
num_steps = 5
num_features = 6
#x_shaped = np.reshape(X, newshape=(-1, num_steps, num_features))
# training set

# training set

(x_train_transformed,
 y_train_transformed) = lstm_gru_data_transform(x_train_sc, y_train_sc, num_steps=num_steps)
assert x_train_transformed.shape[0] == y_train_transformed.shape[0]
# test set
(x_test_transformed,
 y_test_transformed) = lstm_gru_data_transform(x_test_sc, y_test_sc, num_steps=num_steps)
assert x_test_transformed.shape[0] == y_test_transformed.shape[0]
#
# Training phae
#
model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(num_steps, 6), return_sequences=False))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=1, activation='linear'))
#adam = optimizers.Adam(lr=0.001)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#
# Train the model with validation data
#
history = model.fit(x_train_transformed, y_train_transformed, epochs=10, \
          batch_size=32, validation_data=(x_test_transformed,y_test_transformed))
#
#Saving trained model
#
model.save_weights('/contrib/Mamoudou.Ba/lstm_mse.h5')
model.save('/contrib/Mamoudou.Ba/lstm_predict.h5')
lstm_training_loss = history.history['loss']
lstm_validation_loss = history.history['val_loss']

# Combine epochs, training loss, and validation loss into rows
rows = zip(range(1, len(lstm_training_loss) + 1), lstm_training_loss, lstm_validation_loss)

# Write to CSV file
with open('/contrib/Mamoudou.Ba/_lstm_loss_history.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])  # Write header
    writer.writerows(rows)
