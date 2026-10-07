#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.callbacks import EarlyStopping
import random # Import the random library for manual tuning
import sys # To exit if file not found

#
# --- 1. Load Data ---
#
try:
    data = pd.read_csv('surge_training_datasets.csv', parse_dates=['timestamp'], index_col='timestamp')
    df = data.sort_values(by='timestamp')
    data_points = len(data)
    print("Data Head:\n", df.head())
except FileNotFoundError:
    print("Error: 'surge_training_datasets.csv' not found.")
    print("Please make sure the file is in the same directory as the script.")
    sys.exit(1) # Exit the script with an error code

#
#
# --- Custom DILATE-inspired Loss Class ---
# This class-based loss is robust for saving and loading.
#
class DilateLoss(tf.keras.losses.Loss):
    """
    Custom DILATE-inspired loss as a serializable Keras class.
    Combines MSE (point-wise error) with a shape-based loss (first derivative).
    """
    def __init__(self, alpha=0.5, name="dilate_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Ensure tensors are float32
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # 1. Point-wise MSE Loss
        mse = K.mean(K.square(y_true - y_pred), axis=-1)
        
        # 2. Shape Loss (MSE of first derivative)
        # Shift tensors to get (t) and (t-1) values
        y_true_shifted = K.concatenate([K.expand_dims(y_true[:, 0], -1), y_true[:, :-1]], axis=-1)
        y_pred_shifted = K.concatenate([K.expand_dims(y_pred[:, 0], -1), y_pred[:, :-1]], axis=-1)
        
        # Calculate differences (derivatives)
        y_true_diff = y_true - y_true_shifted
        y_pred_diff = y_pred - y_pred_shifted
        
        shape_loss = K.mean(K.square(y_true_diff - y_pred_diff), axis=-1)
        
        # 3. Combine losses
        loss = self.alpha * mse + (1.0 - self.alpha) * shape_loss
        return loss

    def get_config(self):
        # This allows the loss to be saved and loaded correctly
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config
#
#
# --- 2. Data Preprocessing ---
def prepare_data(df, n_past, n_future, target_col):
    """Prepares data for the GRU model."""
    print("Preprocessing data...")
    features = df.columns
    target = df[target_col]
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

# --- 3. Model Building (Simplified Function) ---
def build_model(input_shape, n_outputs, hp):
    """
    Builds and compiles the GRU model from a hyperparameter dictionary (hp).
    """
    model = Sequential()
    
    num_layers = hp['num_layers']
    
    for i in range(num_layers):
        hp_units = hp[f'units_layer_{i+1}']
        is_last_layer = (i == num_layers - 1)
        return_sequences = not is_last_layer
        
        if i == 0:
            model.add(GRU(units=hp_units, activation='relu',
                          input_shape=input_shape,
                          return_sequences=return_sequences))
        else:
            model.add(GRU(units=hp_units, activation='relu',
                          return_sequences=return_sequences))

    model.add(Dense(units=n_outputs)) 
    
    learning_rate = hp['learning_rate']
    
    dilate_loss_fn = DilateLoss(alpha=0.5) 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=dilate_loss_fn)
    
    return model

# --- Main Execution ---

if __name__ == '__main__':
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='min',
        restore_best_weights=True 
    )

    # Configuration
    N_PAST_HOURS = 24
    N_FUTURE_HOURS = 24
    TARGET_COLUMN = 'water_level'

    # 1. Preprocess Data
    X, y, scaler_features, scaler_target = prepare_data(df, N_PAST_HOURS, N_FUTURE_HOURS, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = (X_train.shape[1], X_train.shape[2])

    # --- 3. Manual Hyperparameter Tuning Loop ---
    # This loop REPLACES KerasTuner to avoid the bug
    print("\n--- Starting Manual Hyperparameter Search (No KerasTuner) ---")
    
    # Define the full search space
    possible_learning_rates = [1e-2, 1e-3, 1e-4]
    possible_num_layers = [1, 2, 3] 
    possible_units = [32, 48, 64, 80, 96]
    
    num_trials = 10 # You can increase this number to try more combinations
    results = []
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        
        # 1. Randomly select hyperparameters
        hp = {}
        hp['learning_rate'] = random.choice(possible_learning_rates)
        hp['num_layers'] = random.choice(possible_num_layers)
        for i in range(hp['num_layers']):
            hp[f'units_layer_{i+1}'] = random.choice(possible_units)
            
        print(f"Testing parameters: {hp}")

        # 2. Build and train the model
        # We must clear the session to free GPU memory between trials
        K.clear_session()
        
        model_trial = build_model(input_shape, N_FUTURE_HOURS, hp)
        
        try:
            history = model_trial.fit(X_train, y_train, 
                                epochs=100, # High epoch count, EarlyStopping will handle it
                                batch_size=32, 
                                validation_split=0.1, 
                                callbacks=[early_stopping], 
                                verbose=0) # Set verbose=0 to keep the log clean
            
            # 3. Get the best validation loss and store results
            best_val_loss = min(history.history['val_loss'])
            print(f"Trial {trial + 1} Best val_loss: {best_val_loss}")
            results.append((best_val_loss, hp))

        except Exception as e:
            # This handles any other errors, like GPU OOM
            print(f"!!! Trial {trial + 1} failed with error: {e} !!!")
            print("Skipping this trial.")
            # We append a very high loss so this trial is ignored
            results.append((float('inf'), hp))

    print("\n--- Manual Search Complete ---")

    # Get the optimal hyperparameters
    results.sort(key=lambda x: x[0]) # Sort by val_loss (ascending)
    
    if len(results) == 0 or results[0][0] == float('inf'):
        print("All tuning trials failed. Cannot proceed.")
        print("This might be a persistent GPU OOM error.")
        sys.exit(1)
        
    best_val_loss, best_hps = results[0]

    print(f"""
    The hyperparameter search is complete.
    Best Validation Loss: {best_val_loss}
    The optimal parameters are: {best_hps}
    """)

    # --- 4. Build and Retrain the Best Model ---
    print("\nBuilding and retraining the best model...")
    K.clear_session()
    model = build_model(input_shape, N_FUTURE_HOURS, best_hps)
    
    history = model.fit(X_train, y_train, 
                        epochs=100, # Train for longer, ES will stop it
                        batch_size=32, 
                        validation_split=0.1, 
                        verbose=1, 
                        callbacks=[early_stopping]) 
    print("Final training complete.")

    # --- 5. Make and Interpret a Prediction ---
    print("\n--- Making a sample prediction ---")
    
    #Saving trained model
    print("Saving model to disk...")
    fileout = model.to_json()
    model_json_saved = 'gru_multi_steps_' + str(N_PAST_HOURS) + 'hours_model.json'
    with open(model_json_saved, "w") as json_file:
            json_file.write(fileout)
    weights_file = 'gru_multi_steps_' + str(N_PAST_HOURS) + 'hours_model.h5'
    model.save_weights(weights_file)
    print("Saved model to disk")

    # load json and create model
    print("\nLoading model from disk for prediction...")
    json_file = open(model_json_saved, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json, 
                                   custom_objects={'DilateLoss': DilateLoss})
    
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")
    
    loaded_model.compile(optimizer='adam', loss=DilateLoss(alpha=0.5)) 

    # Predict
    sample_input = X_test[0].reshape(1, N_PAST_HOURS, X_test.shape[2])
    predicted_scaled = loaded_model.predict(sample_input)
    predicted_water_levels = scaler_target.inverse_transform(predicted_scaled)
    actual_scaled = y_test[0].reshape(1, N_FUTURE_HOURS)
    actual_water_levels = scaler_target.inverse_transform(actual_scaled)

    print(f"\nForecast for the next {N_FUTURE_HOURS} hours:")
    for i in range(N_FUTURE_HOURS):
        print(f"  - Hour {i+1}: Predicted={predicted_water_levels[0][i]:.2f}m, Actual={actual_water_levels[0][i]:.2f}m")

