#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K  # Import Keras backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.callbacks import EarlyStopping

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
# --- Define Custom DILATE-inspired Loss Function ---
#
def create_dilate_loss(alpha=0.5):
    """
    Factory function to create the DILATE-inspired loss.
    This loss combines MSE (point-wise error) with a shape-based loss
    (error of the first derivative).

    alpha: float, weight given to the MSE (point-wise) part of the loss.
           (1-alpha) will be given to the shape (derivative) part.
    """
    def dilate_loss(y_true, y_pred):
        """
        The actual loss function.
        """
        # 1. Point-wise Loss (MSE)
        # Measures the error at each individual time step.
        mse = K.mean(K.square(y_true - y_pred), axis=-1)

        # 2. Shape Loss (MSE of first derivative)
        # Measures if the *trend* or *shape* of the prediction is correct.

        # Pad with first value to keep shape [batch, n_steps]
        # This calculates y_true[t] - y_true[t-1]
        y_true_shifted = K.concatenate([K.expand_dims(y_true[:, 0], -1), y_true[:, :-1]], axis=-1)
        y_pred_shifted = K.concatenate([K.expand_dims(y_pred[:, 0], -1), y_pred[:, :-1]], axis=-1)

        # Calculate first derivative (difference)
        y_true_diff = y_true - y_true_shifted
        y_pred_diff = y_pred - y_pred_shifted

        # Calculate MSE of the differences
        shape_loss = K.mean(K.square(y_true_diff - y_pred_diff), axis=-1)

        # 3. Combine losses
        # alpha controls the trade-off.
        # alpha = 1.0 -> pure MSE
        # alpha = 0.0 -> pure shape loss
        # alpha = 0.5 -> balanced
        loss = alpha * mse + (1.0 - alpha) * shape_loss
        
        return loss
    
    return dilate_loss
#
#
# --- 2. Data Preprocessing ---
def prepare_data(df, n_past, n_future, target_col):
    """Prepares data for the GRU model."""
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
    """Builds and compiles the GRU model."""
    print("Building GRU model...")
    model = Sequential()
    model.add(GRU(units=70, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(GRU(units=50, activation='relu'))
    model.add(Dense(units=n_outputs)) # One output neuron for each forecast hour

    # --- Use the custom DILATE loss here ---
    dilate_loss_fn = create_dilate_loss(alpha=0.5)
    model.compile(optimizer='adam', loss=dilate_loss_fn)
   
    model.summary()
    return model

# --- Main Execution ---

if __name__ == '__main__':
    # Early stopping
    early_stopping = EarlyStopping(
    monitor='val_loss',  # Or another metric like 'val_accuracy'
    min_delta=0,  # Minimum change in monitored quantity to qualify as an improvement
    patience=5,   # Number of epochs to wait before stopping if no improvement
    verbose=1,    # Print messages when training stops
    mode='min'    # 'min' for minimization, 'max' for maximization
    # restore_best_weights=True  # Restore the model weights from the epoch with the best monitored value (optional)
)


    # Configuration
    N_PAST_HOURS = 24   # Use the last 24 hours of data to predict
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
    #history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1, \
    #                callbacks=[early_stopping])
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    print("Training complete.")

    # --- 4. Make and Interpret a Prediction ---
    print("\n--- Making a sample prediction ---")
    
    #Saving trained model
    print("Saving model to disk...")
    # serialize model to JSON
    fileout = model.to_json()
    model_json_saved = 'gru_multi_steps_' + str(N_PAST_HOURS) + 'hours_model.json'
    with open(model_json_saved, "w") as json_file:
            json_file.write(fileout)
    # serialize and weights to HDF5
    weights_file = 'gru_multi_steps_' + str(N_PAST_HOURS) + 'hours_model.h5'
    model.save_weights(weights_file)
    print("Saved model to disk")
#
#
# load json and create model
    print("\nLoading model from disk for prediction...")
    json_file = open(model_json_saved, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # --- Must pass custom loss to load_model ---
    dilate_loss_fn = create_dilate_loss(alpha=0.5)
    loaded_model = model_from_json(loaded_model_json, 
                                   custom_objects={'dilate_loss': dilate_loss_fn})
    
    # load weights into new model
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")
    
    # It's good practice to compile the loaded model,
    # especially if you were to use it for evaluation
    loaded_model.compile(optimizer='adam', loss=dilate_loss_fn)

    # Take the first sample from the test set
    sample_input = X_test[0].reshape(1, N_PAST_HOURS, X_test.shape[2])

    # --- Predict with the LOADED model ---
    predicted_scaled = loaded_model.predict(sample_input)

    # Inverse transform the prediction to get the actual water level values
    predicted_water_levels = scaler_target.inverse_transform(predicted_scaled)

    # Get the actual future values for comparison
    actual_scaled = y_test[0].reshape(1, N_FUTURE_HOURS)
    actual_water_levels = scaler_target.inverse_transform(actual_scaled)

    print(f"Input data shape for prediction: {sample_input.shape}")
    print(f"\nForecast for the next {N_FUTURE_HOURS} hours:")
    for i in range(N_FUTURE_HOURS):
        print(f"  - Hour {i+1}: Predicted={predicted_water_levels[0][i]:.2f}m, Actual={actual_water_levels[0][i]:.2f}m")

