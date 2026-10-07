#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, ConvGRU2D, BatchNormalization, Conv2D
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- 1. Configuration Parameters ---
# Grid dimensions
GRID_HEIGHT = 50
GRID_WIDTH = 50

# Number of input features
# (surface temp, pressure, wind speed, gust, direction, water level)
N_FEATURES = 6

# Number of past time steps to use for prediction
INPUT_SEQUENCE_LENGTH = 12 

# How many time steps into the future to predict
FORECAST_HORIZON = 3 

# --- 2. Data Simulation ---
# This function generates realistic-looking dummy data.
# In a real-world scenario, you would replace this with your data loader.
def generate_dummy_data(time_steps, grid_h, grid_w, n_features):
    """Generates a dataset simulating a moving storm front."""
    print("Generating simulated storm data...")
    data = np.random.rand(time_steps, grid_h, grid_w, n_features).astype(np.float32) * 0.1
    
    # Simulate a storm center moving diagonally
    for t in range(time_steps):
        center_y = int(grid_h * (t / time_steps))
        center_x = int(grid_w * (t / time_steps))
        
        # Create a radial gradient around the storm center
        for y in range(grid_h):
            for x in range(grid_w):
                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                # Create a pressure drop and increase in other variables near the center
                if dist < 15:
                    factor = (15 - dist) / 15
                    data[t, y, x, 1] -= factor * 0.5 # Pressure decreases
                    data[t, y, x, 2] += factor * 0.7 # Wind speed increases
                    data[t, y, x, 5] += factor * 0.9 # Water Level increases
    
    # Add some temporal sine wave to simulate tides for water level
    t_axis = np.arange(time_steps)
    tide_effect = np.sin(t_axis * np.pi / 6) * 0.1 # Simple tide simulation
    data[:, :, :, 5] += tide_effect[:, np.newaxis, np.newaxis]
    
    print("Data generation complete.")
    return data

# --- 3. Data Preprocessing ---
def create_sequences(data, input_len, forecast_horizon):
    """Creates input sequences and corresponding target grids."""
    X, y = [], []
    for i in range(len(data) - input_len - forecast_horizon + 1):
        X.append(data[i : i + input_len])
        # Target is the water level grid (last feature) at the forecast horizon
        y.append(data[i + input_len + forecast_horizon - 1, :, :, -1])
    return np.array(X), np.array(y).reshape(-1, GRID_HEIGHT, GRID_WIDTH, 1)

# Generate the data
raw_data = generate_dummy_data(time_steps=200, grid_h=GRID_HEIGHT, grid_w=GRID_WIDTH, n_features=N_FEATURES)

# Normalize each feature across the entire dataset
data_reshaped = raw_data.reshape(-1, N_FEATURES)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_reshaped).reshape(raw_data.shape)

# Create a separate scaler for the target variable to inverse transform predictions
water_level_scaler = MinMaxScaler()
water_level_scaler.fit(raw_data.reshape(-1, N_FEATURES)[:, -1].reshape(-1, 1))

# Create sequences
X, y = create_sequences(data_scaled, INPUT_SEQUENCE_LENGTH, FORECAST_HORIZON)

# Split data into training and testing sets
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# --- 4. Build the ConvGRU Model ---
model = Sequential([
    Input(shape=(INPUT_SEQUENCE_LENGTH, GRID_HEIGHT, GRID_WIDTH, N_FEATURES)),
    
    # First ConvGRU layer
    ConvGRU2D(
        filters=64, 
        kernel_size=(3, 3), 
        padding='same', 
        return_sequences=True, # Return sequence to stack another layer
        activation='relu'
    ),
    BatchNormalization(),

    # Second ConvGRU layer
    ConvGRU2D(
        filters=64, 
        kernel_size=(3, 3), 
        padding='same', 
        return_sequences=False, # Return only the last output
        activation='relu'
    ),
    BatchNormalization(),

    # Output layer: A Conv2D layer to produce the final 2D grid
    Conv2D(
        filters=1, # One filter for the single output channel (water level)
        kernel_size=(1, 1), 
        activation='sigmoid' # Sigmoid because our data is scaled between 0 and 1
    )
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 5. Train the Model ---
print("\nTraining the model...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=20, 
    batch_size=8, 
    validation_split=0.2
)

# --- 6. Make and Visualize a Forecast ---
print("\nMaking a prediction on a test sample...")

# Select a random sample from the test set
sample_idx = np.random.randint(0, len(X_test))
test_sample = X_test[sample_idx]
ground_truth = y_test[sample_idx]

# Add a batch dimension for prediction
test_sample_batch = np.expand_dims(test_sample, axis=0)
prediction_scaled = model.predict(test_sample_batch)[0]

# Inverse transform the prediction and ground truth to their original scale
prediction = water_level_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).reshape(GRID_HEIGHT, GRID_WIDTH)
ground_truth_unscaled = water_level_scaler.inverse_transform(ground_truth.reshape(-1, 1)).reshape(GRID_HEIGHT, GRID_WIDTH)

# Visualize the results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Storm Surge Forecast (T+{FORECAST_HORIZON} hours) using ConvGRU', fontsize=16)

# Plot Ground Truth
im1 = axes[0].imshow(ground_truth_unscaled, cmap='viridis')
axes[0].set_title('Ground Truth Water Level')
axes[0].set_xlabel('Grid Width')
axes[0].set_ylabel('Grid Height')
fig.colorbar(im1, ax=axes[0], label='Water Level (meters)')

# Plot Prediction
im2 = axes[1].imshow(prediction, cmap='viridis')
axes[1].set_title('Predicted Water Level')
axes[1].set_xlabel('Grid Width')
plt.colorbar(im2, ax=axes[1], label='Water Level (meters)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.show()
filename = 'Convgru_storm_surge_forecast.png'
 # Set the name of the variable to plot
plt.savefig(filename) # Set the output file name

