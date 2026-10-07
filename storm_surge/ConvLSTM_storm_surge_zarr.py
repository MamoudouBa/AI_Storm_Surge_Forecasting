#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python

import zarr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from sklearn.preprocessing import MinMaxScaler

# --- 1. DUMMY DATA GENERATION ---
# In a real scenario, you would already have this Zarr dataset.
# This function creates a sample dataset for demonstration purposes.
def create_dummy_zarr_data(path, time_steps, height, width, channels):
    """
    Creates a Zarr array with random data simulating weather patterns.
    Shape: (time, height, width, channels)
    Channels correspond to: [temp, pressure, wind_speed, gust, dir, water_level]
    """
    print("Creating dummy Zarr dataset...")
    shape = (time_steps, height, width, channels)
    # The `chunks` argument is important for performance with Zarr
    chunks = (10, height, width, 1) 
    z = zarr.open(path, mode='w', shape=shape, chunks=chunks, dtype='float32')
    
    # Fill with random data
    z[:] = np.random.rand(*shape).astype('float32')
    print(f"Dummy data created at '{path}' with shape {z.shape}")
    return z

# --- 2. DATA LOADING & PREPROCESSING ---
def prepare_data(zarr_path, sequence_length, target_channel_index):
    """
    Loads data from Zarr, normalizes it, and creates input/output sequences.
    """
    print("Loading and preprocessing data...")
    # Open the Zarr array
    data = zarr.open(zarr_path, mode='r')
    
    # --- Normalization ---
    # Reshape for scaler: from (time, h, w, c) to (time*h*w, c)
    original_shape = data.shape
    reshaped_data = data[:].reshape(-1, original_shape[-1])
    
    # Scale all features to a range of [0, 1]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(reshaped_data)
    
    # Reshape back to original grid format
    scaled_data = scaled_data.reshape(original_shape)
    
    # --- Create Sequences for ConvLSTM ---
    # We will use 'sequence_length' past steps to predict the next step.
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        # Input sequence (e.g., 10 time steps of all 6 features)
        X.append(scaled_data[i:(i + sequence_length)])
        # Target (e.g., the water level at the 11th time step)
        y.append(scaled_data[i + sequence_length, :, :, target_channel_index])

    X = np.array(X)
    y = np.array(y)
    
    # Add a channel dimension to the target 'y'
    y = np.expand_dims(y, axis=-1)
    
    print(f"Input shape (X): {X.shape}") # (samples, timesteps, h, w, channels)
    print(f"Target shape (y): {y.shape}") # (samples, h, w, channels)
    
    return X, y

# --- 3. MODEL BUILDING ---
def build_convlstm_model(input_shape):
    """
    Builds the ConvLSTM model architecture.
    """
    print("Building the ConvLSTM model...")
    model = Sequential()
    
    # Input shape: (timesteps, height, width, channels)
    # The model expects a 5D tensor.
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                       input_shape=input_shape,
                       padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                       padding='same', return_sequences=False))
    model.add(BatchNormalization())
    
    # The output of the last ConvLSTM layer is a 3D tensor (h, w, filters).
    # We use a final Conv2D layer to collapse the filters to our desired
    # single-channel output (the predicted water level map).
    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                     activation='linear', padding='same'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

# --- 4. MAIN EXECUTION ---
if __name__ == '__main__':
    # --- Configuration ---
    ZARR_FILE_PATH = 'storm_data.zarr'
    TIME_STEPS = 100    # Total time steps in the dataset
    GRID_HEIGHT = 20    # Spatial grid height
    GRID_WIDTH = 20     # Spatial grid width
    NUM_FEATURES = 6    # temp, pressure, wind_speed, gust, dir, water_level
    WATER_LEVEL_IDX = 5 # The index of our target variable
    SEQ_LENGTH = 10     # Number of past time steps to use as input
    
    # --- Workflow ---
    # 1. Create dummy data (replace this with your actual data)
    create_dummy_zarr_data(ZARR_FILE_PATH, TIME_STEPS, GRID_HEIGHT, GRID_WIDTH, NUM_FEATURES)
    
    # 2. Prepare the data for the model
    X_train, y_train = prepare_data(ZARR_FILE_PATH, SEQ_LENGTH, WATER_LEVEL_IDX)
    
    # 3. Build the model
    # The input shape is (SEQ_LENGTH, GRID_HEIGHT, GRID_WIDTH, NUM_FEATURES)
    model_input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])
    model = build_convlstm_model(model_input_shape)
    
    # 4. Train the model
    print("\nStarting model training...")
    model.fit(X_train, y_train, batch_size=8, epochs=10, validation_split=0.2)
    print("Model training complete.")
