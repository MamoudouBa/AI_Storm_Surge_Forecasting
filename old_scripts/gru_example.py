# Developed with the assistance of the Gemini AI code assistant.
#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python

#https://www.geeksforgeeks.org/gated-recurrent-unit-networks/

#Import libraries
#
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
import mlnext # https://pypi.org/project/mlnext-framework/

#numpy: For handling numerical data and array manipulations.
#pandas: For data manipulation and reading datasets (CSV files).
#MinMaxScaler: For normalizing the dataset.
#tensorflow.keras.models and tensorflow.keras.layers: For building and training the GRU model.
#Adam: An optimization algorithm used during training.
df = pd.read_csv('surge_test.csv', parse_dates=['Date'], index_col='Date')
#df["Temperature_bis"] = df["Temperature"] - 0.5
#print(df.head())
#pd.read_csv(): Reads a CSV file into a pandas DataFrame. Here, we are assuming that the dataset has a
#               'Date' column which is set as the index of the DataFrame.
#date_parser=True: Ensures that pandas parses the 'Date' column as datetime.

#Preprocessing the Data
print(df)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)
#print("scaled data shape is :", scaled_data.shape)
#print(scaled_data[0])
#MinMaxScaler(): This scales the data to a range of 0 to 1.
#This is important because neural networks perform better when input features are scaled properly.
#
#Preparing Data for GRU
#print("weater level ",df.water_level.values[0])
print("df.values shape: ",df.values.shape)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)
print("scaled_data shape: ", scaled_data.shape)
y_data = df.water_level.values
#print(y_data)
# Reshape the inpu X array to 3d and y array to 2d

y_data =y_data.reshape(-1, 1)
#print(y_data.shape,df.shape)
water_level_scaled = scaler.fit_transform(y_data)
#print(scaled_data[0])
#MinMaxScaler(): This scales the data to a range of 0 to 1.
#This is important because neural networks perform better when input features are scaled properly.
#
#Preparing Data for GRU
#
# Function to reshape the input arrays
#
def lstm_gru_data_transform(x_data, y_data, num_steps=100):
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
time_step = 100
print("****************** scaled_data shape is:", scaled_data.shape)
X, y = create_dataset(scaled_data,water_level_scaled, time_step)
print("X shape is ",X.shape)
print("scaled data: ", scaled_data, X, y)
# Now reshape the arrays
#lstm_gru_data_transform(X, y, num_steps=100)
data = X
print("Printing X: ",X)
X = mlnext.temporalize(data=data, timesteps=100, stride=1,verbose=True)
#Now reduce the leth of y sample to that of X sample
y = y[:len(X)]
#print("Printinh shape  ",X.shape, y.shape)
#print("printing shape   ",X.shape[0],X.shape[1], X.shape[2])
#print(X)
#
#create_dataset(): Prepares the dataset for time-series forecasting. It creates sliding windows of time_step length to predict the next time step.
#X.reshape(): Reshapes the input data to fit the expected shape for the GRU which is 3D: [samples, time steps, features].
#5. Building the GRU Model

model = Sequential()
#print(X.shape)
model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[1])))
#model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
print("++++++++++++++++ ",X.shape[0], X.shape[1])
model.add(GRU(units=50))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss=dilate_loss)
#model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss_with_dilation)
#model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

#GRU(units=50): Adds a GRU layer with 50 units (neurons).
#return_sequences=True: Ensures that the GRU layer returns the entire sequence (required for stacking multiple GRU layers).
#Dense(units=1): The output layer which predicts a single value for the next time step.
#Adam(): An adaptive optimizer commonly used in deep learning.
#
#Training the Model

print("X shape is: ******************************* ",X.shape,y.shape)
# reduce X size to match y size
#X = X[:len(y)]
model.fit(X, y, epochs=5, batch_size=32)
#model.fit(X, y, epochs=15, batch_size=16)
#model.fit(): Trains the model on the prepared dataset. The epochs=10 specifies the number of iterations over the entire dataset,
#and batch_size=32 defines the number of samples per batch.
#Making Predictions

data = scaled_data
#print("Printing X: ",X)
#scaled_data = mlnext.temporalize(data=data, timesteps=100, verbose=True)

input_sequence = scaled_data
print("********************************************************")
print("values of scaled_data[1]: is", scaled_data.shape, X.shape)
#input_sequence = scaled_data[-time_step:].reshape(1, time_step, 6)
predicted_values = model.predict(X)
#Making Predictions
print(input_sequence.shape,X.shape)
#print(f"The predicted temperature for the next day is: {predicted_values[0][0]:.2f}°C")
input_sequence = scaled_data
#input_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
predicted_values = model.predict(X)
#print(predicted_values.shape)
#print(predicted_values)
#
#Inverse Transforming the Predictions
#Inverse Transforming the Predictions refers to the process of converting the scaled (normalized) predictions back to their original scale.

predicted_values = scaler.inverse_transform(predicted_values)
#print(f"The predicted temperature for the next day is: {predicted_temperature[0][0]:.2f}°C")
print(f"The predicted water level for the next day is: {predicted_values[0][0]:.2f} meters")
print("day 2 ",predicted_values[1][0],"day 3 ",predicted_values[2][0], "day 4 ",predicted_values[3][0])
print("day 10 ",predicted_values[10][0])
