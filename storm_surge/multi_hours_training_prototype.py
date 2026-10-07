#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#This scripted is a modified a Gemini prototyped lstm to train 5 hours lumped together
## 1. Data Generation (Replace with your own data)
# ----------------------------------------------------------------
# This section generates synthetic data for demonstration.
# To use your own data, load it here, e.g.,
# df = pd.read_csv('your_storm_data.csv', index_col='timestamp', parse_dates=True)

data = pd.read_csv('surge_training_datasets.csv', parse_dates=['timestamp'], index_col='timestamp')
#
df = data.sort_values(by='timestamp')

data_points = len(data)
print("Data Head:\n", df.head())


## 2. Data Preprocessing
# ----------------------------------------------------------------
# Define model parameters
n_past_steps = 24       # Lookback window: Use the last 24 hours of data
n_future_steps = 5      # Forecast horizon: The window for our target value
n_features = df.shape[1]

# Scale all features to a [0, 1] range
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# We need a separate scaler for the target variable to reverse the scaling on our final prediction
target_col_index = df.columns.get_loc('water_level')
target_scaler = MinMaxScaler()
target_scaler.fit(df[['water_level']])

# Create sequences of data for the LSTM
X, y = [], []
for i in range(n_past_steps, len(df_scaled) - n_future_steps + 1):
    X.append(df_scaled[i - n_past_steps:i, 0:n_features])
    
    # --- KEY CHANGE HERE ---
    # Instead of taking all 5 future steps as the target, we find the MAXIMUM value
    # in the next 5 hours and use that single value as our target (y).
    future_window = df_scaled[i:i + n_future_steps, target_col_index]
    y.append(np.max(future_window))

X, y = np.array(X), np.array(y)

print(f"\nShape of input sequences (X): {X.shape}")   # (samples, timesteps, features)
print(f"Shape of target values (y): {y.shape}")     # (samples,) -> Note: this is now a 1D array

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print('*** y_train shape is: ', y_train.shape)

## 3. Build and Train the LSTM Model
# ----------------------------------------------------------------
print("\n🧠 Building the LSTM model...")
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_past_steps, n_features), return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=False))
model.add(Dropout(0.2))

# --- KEY CHANGE HERE ---
# The output layer now has only 1 neuron because we are predicting a single value
# (the max water level) instead of a sequence of 5 values.
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.summary()

print("\n⏳ Training the model...")
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1, verbose=1)
print("✅ Model training complete.")

## 4. Make and Display a Prediction
# ----------------------------------------------------------------
# Select a sample from the test set to forecast
sample_input = X_test[50:51]
actual_max_scaled = y_test[50]

# Generate the forecast
predicted_max_scaled = model.predict(sample_input)[0][0]

# To inverse transform, scalers expect a 2D array, so we reshape our single values
predicted_water_level = target_scaler.inverse_transform(np.array([[predicted_max_scaled]]))
actual_water_level = target_scaler.inverse_transform(np.array([[actual_max_scaled]]))

# Display the results
print("\n--- 🌊 Maximum Storm Surge Forecast (Next 5 Hours) ---")
print(f"\nPredicted Max Water Level: {predicted_water_level[0][0]:.2f} meters")
print(f"  Actual Max Water Level: {actual_water_level[0][0]:.2f} meters")

