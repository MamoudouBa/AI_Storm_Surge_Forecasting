# Developed with the assistance of the Gemini AI code assistant.
#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python


import numpy as np
import pandas as pd
import csv
import os
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential, model_from_json
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor


# GridSearchCV tuning
#
# Define Keras model function
'''
def create_lstm_model(optimizer='adam', activation='relu'):
    # Define model architecture
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer=Adam(learning_rate=0.001), metrics=['mae'])
    return model
'''
# Create LSTM function for sklearn grid search
def create_lstm_model(neurons=12, loss='mean_squared_error', activation='relu', optimizer='Adam', batch_size=12):
    model = Sequential()
    model.add(LSTM(units=neurons, activation=activation, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(units=1))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model
    #model.add(Dense(units=n_steps_out))

# Define hyperparameter grid
param_grid = {
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'activation': ['relu', 'sigmoid', 'tanh']
}
#
#

# Custom loss functions
#
def lstm_data_transform(x_data, y_data, num_steps = 1):
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
num_steps = 24
num_features = 24
#x_shaped = np.reshape(X, newshape=(-1, num_steps, num_features))
# training set

# training set

(x_train_transformed,
 y_train_transformed) = lstm_data_transform(x_train_sc, y_train_sc, num_steps=num_steps)
assert x_train_transformed.shape[0] == y_train_transformed.shape[0]
# test set
#
#
# load json and create model
json_file = open('/contrib/Mamoudou.Ba/lstm_steps_24hours_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/contrib/Mamoudou.Ba/lstm_steps_24hours_model.h5")
print("Loaded model from disk")
print(loaded_model.summary())
loaded_model.compile(loss=dilate_loss, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
predictions =loaded_model.predict(x_train_transformed)
#predictions = scaler_y.inverse_transform(predictions)
#print(' predictions inverse  ', scaler_y.inverse_transform(predictions))
#
# Prepare for plotting the last 48 observations

#Create pandas dataframe for the predictions
df_observations = data[['water_level']].reset_index()
df_observations.rename(columns={'index': 'timestamp'}, inplace=True)
df_observations['timestamp'] = pd.to_datetime(df_observations['timestamp'], format='%Y-%m-%d %H %M %S')
df_observations['water_level'] = df_observations['water_level']*3.28084
N = 96
last_n_past_obs_values = df_observations.tail(N)
periods=len(last_n_past_obs_values)
last_n_past_obs_values['timestamp'] = pd.date_range(start=last_n_past_obs_values['timestamp'].iloc[0] + pd.Timedelta(hours=1),freq='h', periods=len(last_n_past_obs_values))
#df_observations['date'] = df_observations['timestamp'].dt.date
#df_observations['hour'] = df_observations['timestamp'].dt.hour
# Prepare for plotting the last 48 observations
print('The last past 96 observations', last_n_past_obs_values)

# Prepare for plotting the last 48 predictions based on the trained model
#Create pandas dataframe for the predictions
#
#
predictions =loaded_model.predict(x_train_transformed)
#predictions = scaler_y.inverse_transform(predictions)
#print(' predictions inverse  ', scaler_y.inverse_transform(predictions))
#
#Create pandas dataframe for the predictions
y_past = np.array(predictions).reshape(-1, 1)
y_past = scaler_y.inverse_transform(y_past)
df_past = data[['water_level']].reset_index()
df_past.rename(columns={'index': 'timestamp'}, inplace=True)
df_past['timestamp'] = pd.to_datetime(df_past['timestamp'], format='%Y-%m-%d %H %M %S')
df_past['water_level'] = df_past['water_level']*3.28084
N = 96
last_n_past_values = df_past.tail(N)
periods=len(last_n_past_values)

y_past = pd.DataFrame(columns=['timestamp', 'water_level'])
print(' df_past :', df_past)
periods=len(last_n_past_values)
#df_future['timestamp'] = pd.date_range(start=df_past['timestamp'].iloc[-1] + pd.Timedelta(hours=1), periods=n_future)
y_past['timestamp'] = pd.date_range(start=last_n_past_values['timestamp'].iloc[0] + pd.Timedelta(hours=1),freq='h', periods=len(last_n_past_values))
#print('y_past timestamp', df_past['timestamp'])
y_past = last_n_past_values
print('***+++++**** y_past ', y_past)
#y_past['date'] = y_past['timestamp'].dt.date
#y_past['hour'] = y_past['timestamp'].dt.hour

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
#
#
# generate the training sequences
n_forecast = 1
n_lookback = 24
scaler_p = StandardScaler()
# scaling
predictions = scaler_p.fit_transform(predictions)

X = []
Y = []
for i in range(n_lookback, len(predictions) - n_forecast + 1):
    X.append(predictions[i - n_lookback: i])
    Y.append(predictions[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)
# Running GridSearcgCV
# Wrap the custom wrapper function for use with scikit-learn
lstm_regressor = KerasRegressor(build_fn=create_lstm_model, epochs=50, batch_size=6, verbose=1)

# Define the parameter grid
param_grid = {
    'optimizer': ['SGD', 'RMSprop', 'Adam'],
    'loss': ['mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error', 'huber'],
    'batch_size': [64, 128],
    'epochs': [50, 80],
    'neurons': [50, 128],
    'activation': ['relu', 'tanh', 'sigmoid']}
    #'activation': ['relu', 'tanh', 'sigmoid', 'linear', 'swish']}


# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=lstm_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)

# Fit the grid search to the data
grid_search_result_second_tuning = grid_search.fit(X, Y)

print('grid_search_result_second_tuning : ', grid_search_result_second_tuning)
#

'''
# train the model
tf.random.set_seed(0)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
#
#
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#model.compile(optimizer=Adam(learning_rate=0.01), loss= dilate_loss)
#
#Training the Model
history = model.fit(X, Y, epochs=10, \
          batch_size=16, validation_split=0.2, verbose=1, callbacks=[early_stopping])

model = Sequential()
#First layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
#Second layer
model.add(keras.layers.LSTM(64, return_sequences=False))
#Third layer

model.add(keras.layers.Dense(16, activation="relu"))
#4th layer
model.add(keras.layers.Dropout(0.2))
#Final Output layer
model.add(keras.layers.Dense(1))
model.summary()

model.compile(optimizer="adam",
             loss ="mae",
             metrics=[keras.metrics.RootMeanSquaredError()])

history = model.fit(X, Y, epochs =10, batch_size=16, validation_split=0.2,verbose=1,callbacks=[early_stopping])

#
#print(predictions.shape)
#print(predictions)
# generate the training sequences for future forecasts
# generate the multi-step forecasts
n_future = 48
y_future = []


# Use the the last 48th values as imput for future forecasts

X = data.iloc[47:48]
Y = y_past.iloc[47:48] 
X = X.values
Y = Y['water_level'].values
#Reshape to two dimensions
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)


#print('++++++++++++ X, and Y shapes ',  X.shape, Y.shape)
#print('X ', X)
#print(' Y ', Y)
#print(' df_past Y_past shapes: ', data.shape,y_past.shape)
print('*********** Printing Y[48] ', X, Y, y_past.iloc[47], data.iloc[47])
#LSTM and LSTM accesp three dimension arrays, so reshaping now the 2d into 3d (sample, number of feature, number of values)
X = X.reshape(1,  6, 1)
Y = Y.reshape(1, 1, 1)
#x_pred = X[-1:, :, :]  # last observed input sequence
#y_pred = Y[-1:, :, :]  # last observed input sequence
x_pred = X  #the first element of the last 48 input sequences
y_pred = Y        # The first of 48 last observed input sequence

print('x_pred y_pred ', x_pred.shape, y_pred.shape)
print('*******  ', y_pred)


for i in range(n_future):


    # feed the last forecast back to the model as an input
    x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

    #print(x_pred.shape)
    # generate the next forecast
    y_pred = model.predict(x_pred)
    # save the forecast
    y_future.append(y_pred.flatten()[0])


    # feed the last forecast back to the model as an input
#    x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)
#    y_future.append(y_pred.flatten()[0])
# transform the forecasts back to the original scale
y_future = np.array(y_future).reshape(-1, 1)
y_future = scaler_y.inverse_transform(y_future)
#df_past = last_n_past_obs_values[['water_level']].reset_index()
df_past = data[['water_level']].reset_index()
df_past.rename(columns={'index': 'timestamp'}, inplace=True)
df_past['timestamp'] = pd.to_datetime(df_past['timestamp'], format='%Y-%m-%d %H %M %S')
#
# prepare for future forecasts
#
df_future = pd.DataFrame(columns=['timestamp', 'Forecast'])
last_96_rows = data.tail(96)
range_from_last = last_96_rows.iloc[48:]
#print('range_from_last ', range_from_last)
periods=len(range_from_last)
print(range_from_last.index)
range_from_last_datetime = pd.to_datetime(range_from_last.index, format = '%Y-%m-%d %H %M %S')
#print('range_from_last datetime ',range_from_last_datetime)
df_future['timestamp'] = pd.date_range(start= range_from_last_datetime[0] + pd.Timedelta(hours=1),freq='h', periods=len(range_from_last))
range_from_last['date'] = range_from_last_datetime
print('range_from_last ', range_from_last['date'])
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
ax.plot(last_n_past_obs_values['timestamp'],last_n_past_obs_values['water_level'],y_past['timestamp'], y_past['water_level'], df_future['timestamp'], df_future['Forecast'])
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

plt.xticks(rotation=45)
plt.xlabel("Time (MM DD HH)")
plt.ylabel("Water level in feet")
plt.title("Plot over 48 Hours")
plt.legend(['Observed water level)','Past predicted water levels', 'Future predicted water levels'], loc='upper left')

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
#plt.show()
filename = 'lstm_aws_prediction_ft_myers.png'
 # Set the name of the variable to plot
plt.savefig(filename) # Set the output file name

#print(df_past)

'''
