#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#
# A modified template created by Gemini.
# --- 1. Data Simulation ---
# In a real scenario, you would load your data here, e.g., pd.read_csv('your_data.csv')
data = pd.read_csv('surge_training_datasets.csv', parse_dates=['timestamp'], index_col='timestamp')
#
df = data.sort_values(by='timestamp')

data_points = len(data)
print("Data Head:\n", df.head())
#
#
#
# Define loss function
#
#
#This function is not from Gemini
def dilate_loss(y_true, y_pred):
        # Calculate shape distortion loss (e.g., using Dynamic Time Warping or a similar method)
        shape_loss = tf.reduce_mean(tf.square(y_true - y_pred))  # Example: Mean Squared Error

        # Calculate temporal localisation loss (e.g., using a time-based error metric)
        time_loss = tf.reduce_mean(tf.abs(y_true - y_pred))  # Example: Mean Absolute Error

        # Combine the losses
        total_loss = shape_loss + time_loss  # Adjust weights as needed

        return total_loss
###################

#Gemini template
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

# --- 3. Model Building ---
def build_lstm_model(input_shape, n_outputs):
    """Builds and compiles the LSTM model."""
    print("Building LSTM model...")
    model = Sequential()
    model.add(LSTM(units=70, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dense(units=n_outputs)) # One output neuron for each forecast hour

    #model.compile(optimizer='adam', loss='mean_squared_error')
    model.compile(optimizer='adam', loss=  dilate_loss)
    model.summary()
    return model

# --- Main Execution ---
if __name__ == '__main__':
    # Configuration
    N_PAST_HOURS = 24  # Use the last 24 hours of data to predict
    N_FUTURE_HOURS = 24   # Predict the next 5 hours
    TARGET_COLUMN = 'water_level' # The column we want to predict

    # 1. Get Data

    # 2. Preprocess Data
    X, y, scaler_features, scaler_target = prepare_data(df, N_PAST_HOURS, N_FUTURE_HOURS, TARGET_COLUMN)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('*** y_train shape ** ', y_train.shape)
    # 3. Build and Train Model
    input_shape = (X_train.shape[1], X_train.shape[2]) # (n_past_hours, n_features)
    model = build_lstm_model(input_shape, N_FUTURE_HOURS)

    print("\nTraining model...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    print("Training complete.")

    # --- 4. Make and Interpret a Prediction ---
    print("\n--- Making a sample prediction ---")
    # Take the first sample from the test set
    sample_input = X_test[0].reshape(1, N_PAST_HOURS, X_test.shape[2])
    last_72_obs = df.tail(72)
    sample_input = last_72_obs[0].reshape

    # Predict
    predicted_scaled = model.predict(sample_input)

    # Inverse transform the prediction to get the actual water level values
    predicted_water_levels = scaler_target.inverse_transform(predicted_scaled)

    # Get the actual future values for comparison
    actual_scaled = y_test[0].reshape(1, N_FUTURE_HOURS)
    actual_water_levels = scaler_target.inverse_transform(actual_scaled)

    print(f"Input data shape for prediction: {sample_input.shape}")
    print("\nForecast for the next 5 hours:")
    for i in range(N_FUTURE_HOURS):
        print(f"  - Hour {i+1}: Predicted={predicted_water_levels[0][i]:.2f}m, Actual={actual_water_levels[0][i]:.2f}m")
