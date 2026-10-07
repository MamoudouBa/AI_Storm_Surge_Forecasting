#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
#
# Import the necessary packages
#
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import datetime as dt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping # <<< 1. IMPORT ADDED
#
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR) # Only print ERROR messages
#Suppress INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Only print ERROR messages
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#
# --- 1. Load Data ---
#
print("Loading data...")
data = pd.read_csv('surge_training_datasets.csv', parse_dates=['timestamp'], index_col='timestamp')
df = data.sort_values(by='timestamp')
print("Data Head:\n", df.head())


#
# --- 2. Define Custom DILATE-inspired Loss Function ---
#
def create_dilate_loss(alpha=0.5):
    """
    Factory function to create the DILATE-inspired loss.
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

    dilate_loss.__name__ = 'dilate_loss'
    return dilate_loss
#
#

#
# --- 3. Data Preprocessing Function ---
#
def prepare_data(df, n_past, n_future, target_col):
    """Prepares data for the LSTM model."""
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

# --- Main Execution ---
if __name__ == '__main__':
    #
    # --- 4. Configuration ---
    #
    N_PAST_HOURS = 24  # Use the last 24 hours of data
    N_FUTURE_HOURS = 12   # Predict the next 5 hours
    TARGET_COLUMN = 'water_level'

    #
    # --- 5. Prepare Data ---
    #
    X, y, scaler_features, scaler_target = prepare_data(df, N_PAST_HOURS, N_FUTURE_HOURS, TARGET_COLUMN)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

   #
    # --- 6. Define and Build Model (UPDATED) ---
    #
    print("\n--- Building and Training Model with Optimal Hyperparameters ---")

    # Optimal parameters: {'num_layers': 3, 'units_layer_1': 64, 'units_layer_2': 96, 'units_layer_3': 96}

    model = Sequential()

    # Layer 1
    model.add(GRU(units=64, activation='relu',
                  input_shape=(N_PAST_HOURS, X_train.shape[2]),
                  return_sequences=True)) # return_sequences=True because next layer is GRU

    # Layer 2
    model.add(GRU(units=96, activation='relu',
                  return_sequences=True)) # return_sequences=True because next layer is GRU

    # Layer 3
    model.add(GRU(units=96, activation='relu',
                  return_sequences=False)) # return_sequences=False because next layer is Dense

    # Output Layer
    model.add(Dense(N_FUTURE_HOURS))

    #
    # --- 7. Compile Model (UPDATED) ---
    #

    # Optimal parameter: {'learning_rate': 0.001}
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    dilate_loss_fn = create_dilate_loss(alpha=0.5)
    model.compile(optimizer=optimizer, loss=dilate_loss_fn)
    model.summary()

    # --- 8. CONFIGURE CALLBACKS ---
    # <<< 2. CALLBACK CONFIGURED
    # Stop training if 'val_loss' doesn't improve for 10 epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #
    # --- 9. Train Model ---
    #
    history = model.fit(
        X_train,
        y_train,
        epochs=100,      # Increased epochs, since EarlyStopping will find the best one
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[early_stopping] # <<< 3. CALLBACK ADDED
    )

    #
    # --- 10. Save Trained Model ---
    #
    print("\n--- Saving model to disk... ---")

    model_json_saved = f'gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_OPTIMAL.json'
    weights_file = f'gru_multi_steps_{N_PAST_HOURS}past_{N_FUTURE_HOURS}future_model_OPTIMAL.h5'

    fileout = model.to_json()
    with open(model_json_saved, "w") as json_file:
            json_file.write(fileout)

    model.save_weights(weights_file)

    print(f"Saved model architecture to: {model_json_saved}")
    print(f"Saved model weights to: {weights_file}")

