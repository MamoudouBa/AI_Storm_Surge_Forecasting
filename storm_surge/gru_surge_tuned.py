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

# --- Import KerasTuner ---
import keras_tuner as kt

#
# --- 1. Load Data ---
#
data = pd.read_csv('surge_training_datasets.csv', parse_dates=['timestamp'], index_col='timestamp')
df = data.sort_values(by='timestamp')
data_points = len(data)
print("Data Head:\n", df.head())

#
#
# --- Define Custom DILATE-inspired Loss Function ---
#
def create_dilate_loss(alpha=0.5):
    """
    Factory function to create the DILATE-inspired loss.
    This loss combines MSE (point-wise error) with a shape-based loss
    (error of the first derivative).
    """
    def dilate_loss(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred), axis=-1)
        y_true_shifted = K.concatenate([K.expand_dims(y_true[:, 0], -1), y_true[:, :-1]], axis=-1)
        y_pred_shifted = K.concatenate([K.expand_dims(y_pred[:, 0], -1), y_pred[:, :-1]], axis=-1)
        y_true_diff = y_true - y_true_shifted
        y_pred_diff = y_pred - y_pred_shifted
        shape_loss = K.mean(K.square(y_true_diff - y_pred_diff), axis=-1)
        loss = alpha * mse + (1.0 - alpha) * shape_loss
        return loss
    
    # We must name the function for Keras to save/load it properly
    dilate_loss.__name__ = 'dilate_loss'
    return dilate_loss
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

# --- 3. Model Building (Hypermodel) ---
def create_hypermodel(input_shape, n_outputs):
    """
    Factory function to create the hypermodel builder.
    This allows us to pass fixed parameters (input_shape, n_outputs)
    while letting KerasTuner control the hyperparameters (hp).
    """
    def build_model(hp):
        """
        Builds and compiles the GRU model with tunable hyperparameters.
        """
        model = Sequential()
        
        # --- 1. Tune the number of GRU layers ---
        num_layers = hp.Int('num_layers', min_value=1, max_value=3)
        
        for i in range(num_layers):
            # --- 2. Tune the number of units in each layer ---
            hp_units = hp.Int(f'units_layer_{i+1}', min_value=32, max_value=96, step=16)
            
            is_last_layer = (i == num_layers - 1)
            return_sequences = not is_last_layer
            
            if i == 0:
                # First layer needs the input_shape
                model.add(GRU(units=hp_units, activation='relu',
                              input_shape=input_shape,
                              return_sequences=return_sequences))
            else:
                # Subsequent layers
                model.add(GRU(units=hp_units, activation='relu',
                              return_sequences=return_sequences))

        # Output layer
        model.add(Dense(units=n_outputs)) 

        # --- 3. Tune the learning rate ---
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        # Compile the model
        dilate_loss_fn = create_dilate_loss(alpha=0.5)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=dilate_loss_fn)
        
        return model
    
    return build_model

# --- Main Execution ---

if __name__ == '__main__':
    # Early stopping
    # This callback is what "tunes" the number of epochs for each trial
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='min',
        restore_best_weights=True # Restore best weights at the end of training
    )

    # Configuration
    N_PAST_HOURS = 24
    N_FUTURE_HOURS = 24
    TARGET_COLUMN = 'water_level'

    # 1. Preprocess Data
    X, y, scaler_features, scaler_target = prepare_data(df, N_PAST_HOURS, N_FUTURE_HOURS, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define Model Builder
    input_shape = (X_train.shape[1], X_train.shape[2])
    model_builder = create_hypermodel(input_shape, N_FUTURE_HOURS)

    # --- 3. Set up the KerasTuner ---
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=10,  # Number of different hyperparameter combinations to try
        executions_per_trial=1, # How many times to train each combination
        directory='keras_tuner_dir',
        project_name='surge_gru_tuning'
    )

    tuner.results_summary()
    print("\n--- Starting Hyperparameter Search ---")
    
    # Run the search
    # --- MODIFICATION ---
    # Increased epochs from 20 to 100.
    # This is the "max" epochs. EarlyStopping will stop each trial
    # at its *actual* best epoch (e.g., at 15, 30, or 60)
    tuner.search(X_train, y_train, 
                 epochs=100, # Increased max epochs *per trial*
                 batch_size=32, 
                 validation_split=0.1, 
                 callbacks=[early_stopping], # Early stopping finds the best epoch
                 verbose=1)

    print("\n--- Search Complete ---")
    
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"""
    The hyperparameter search is complete. The optimal parameters are:
    - Learning Rate: {best_hps.get('learning_rate')}
    - Number of Layers: {best_hps.get('num_layers')}
    """)
    
    for i in range(best_hps.get('num_layers')):
        print(f"  - Units in Layer {i+1}: {best_hps.get(f'units_layer_{i+1}')}")

    # --- 4. Build and Retrain the Best Model ---
    print("\nBuilding and retraining the best model...")
    model = tuner.build_model(best_hps)
    
    # Retrain the best model with more epochs (EarlyStopping will find the best point)
    # We also use a high number here (like 100) and let EarlyStopping
    # find the true best stopping point for the *final* model.
    history = model.fit(X_train, y_train, 
                        epochs=100, # Train for longer, ES will stop it
                        batch_size=32, 
                        validation_split=0.1, 
                        verbose=1, 
                        callbacks=[early_stopping]) # Use early stopping
    print("Final training complete.")

    # --- 5. Make and Interpret a Prediction (Same as before) ---
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
    
    # Must pass custom loss to load_model
    dilate_loss_fn = create_dilate_loss(alpha=0.5)
    loaded_model = model_from_json(loaded_model_json, 
                                   custom_objects={'dilate_loss': dilate_loss_fn})
    
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")
    
    loaded_model.compile(optimizer='adam', loss=dilate_loss_fn) # Re-compile loaded model

    # Predict
    sample_input = X_test[0].reshape(1, N_PAST_HOURS, X_test.shape[2])
    predicted_scaled = loaded_model.predict(sample_input)
    predicted_water_levels = scaler_target.inverse_transform(predicted_scaled)
    actual_scaled = y_test[0].reshape(1, N_FUTURE_HOURS)
    actual_water_levels = scaler_target.inverse_transform(actual_scaled)

    print(f"\nForecast for the next {N_FUTURE_HOURS} hours:")
    for i in range(N_FUTURE_HOURS):
        print(f"  - Hour {i+1}: Predicted={predicted_water_levels[0][i]:.2f}m, Actual={actual_water_levels[0][i]:.2f}m")

