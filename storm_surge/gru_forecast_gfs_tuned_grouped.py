#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python

#https://www.geeksforgeeks.org/gated-recurrent-unit-networks/

#Import libraries
#
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
# --- NEW: Added for splitting groups ---
from sklearn.model_selection import train_test_split 

from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K # Import K for K.clear_session()
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import GRU, Dense, Dropout # Import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Required for date formatting
import random # Import the random library for manual tuning
import sys # To exit if file not found

#
# --- Custom DILATE-inspired Loss Class ---
# This class-based loss is robust for saving and loading.
#
date_format = "%Y-%m-%d %H %M %S"
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
        # For multi-step output (n_future > 1), we calculate differences along the time axis.
        # This assumes y_true and y_pred have shape (batch_size, n_future).
        # We need to handle the first element.

        # Calculate differences (derivatives)
        y_true_diff = y_true[:, 1:] - y_true[:, :-1]
        y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]

        # Pad the first element with zeros so shape_loss has the same sequence length as mse
        # tf.zeros_like(y_true[:, :1]) creates a tensor of zeros with the same shape as the first column.
        y_true_diff_padded = K.concatenate([tf.zeros_like(y_true[:, :1]), y_true_diff], axis=-1)
        y_pred_diff_padded = K.concatenate([tf.zeros_like(y_pred[:, :1]), y_pred_diff], axis=-1)

        shape_loss = K.mean(K.square(y_true_diff_padded - y_pred_diff_padded), axis=-1)

        # 3. Combine losses
        loss = self.alpha * mse + (1.0 - self.alpha) * shape_loss
        return loss

    def get_config(self):
        # This allows the loss to be saved and loaded correctly
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config

# --- START: Modified Data Loading and Preprocessing ---

print("--- Loading and Aligning Data ---")
# Opening input data
# Original observation data (contains actual water level and observed met features)
df_obs = pd.read_csv('/contrib/Mamoudou.Ba/storm_surge/st_petersburg_training_gfs_wl_data.csv', index_col='timestamp', parse_dates=False)
# Drop any rows with nan
df_obs = df_obs.dropna()
df_obs = df_obs.sort_values(by='timestamp')
# Drop any rows that had unparseable dates
df_obs = df_obs.dropna(subset=['timestamp'])

# **Explicitly convert index to datetime with the correct format**
df_obs.index = pd.to_datetime(df_obs.index, format='%Y-%m-%d %H %M %S')


# NWP forecast data (contains forecast met features, water_level column will be ignored for its values here)
df_nwp_raw = pd.read_csv('/contrib/Mamoudou.Ba/storm_surge/st_petersburg_gfs_data.csv', index_col='timestamp', parse_dates=False)
# Drop any rows with nan
df_nwp_raw = df_nwp_raw.dropna()
df_nwp_raw = df_nwp_raw.sort_values(by='timestamp')
# Drop any rows that had unparseable dates
df_nwp_raw = nwp_raw.dropna(subset=['timestamp'])

# **Explicitly convert index to datetime with the correct format**
df_nwp_raw.index = pd.to_datetime(df_nwp_raw.index, format='%Y-%m-%d %H %M %S') # Apply format here

# Ensure both dataframes cover the same time range for merging and alignment
common_timestamps = df_obs.index.intersection(df_nwp_raw.index)
df_obs = df_obs.loc[common_timestamps]
df_nwp_raw = df_nwp_raw.loc[common_timestamps]

# Define the features to be used for the model
# These are the *observed* meteorological features from df_obs for training
# and will be replaced by NWP *forecasts* during future prediction.
# water_level is the target.
features_for_model = ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust']
target_feature = 'water_level'
num_features = len(features_for_model) # 5 features

# Create the combined DataFrame for training (input features + target)
# Ensure the order of features is consistent for scaling and model input
df_training = df_obs[features_for_model + [target_feature]]

# --- NEW: Configuration for Grouping and Sequences ---
print("\n--- Grouping and Filtering Time Series Data ---")
num_steps = 24 # Number of past hours to look at for prediction
N_FUTURE_HORIZONS = 24 # Number of future hours to predict in one go
GAP_THRESHOLD_HOURS = 3.0 # Gap is defined as > 3 hours
# A group must be large enough to create at least ONE sequence
MIN_GROUP_SIZE = num_steps + N_FUTURE_HORIZONS # 12 + 24 = 36

# --- NEW: 1. Identify and filter groups ---
diffs = df_training.index.to_series().diff()
group_ids = (diffs > pd.Timedelta(hours=GAP_THRESHOLD_HOURS)).cumsum()
df_training['group'] = group_ids

group_sizes = df_training.groupby('group').size()
valid_group_ids = group_sizes[group_sizes >= MIN_GROUP_SIZE].index

print(f"Found {len(group_sizes)} total groups based on >{GAP_THRESHOLD_HOURS}hr gap.")
print(f"Keeping {len(valid_group_ids)} groups with >= {MIN_GROUP_SIZE} elements.")

if len(valid_group_ids) == 0:
    print(f"Error: No data groups found with at least {MIN_GROUP_SIZE} elements. Exiting.")
    sys.exit(1)

# Create the final DataFrame with only the valid groups
df_valid = df_training[df_training['group'].isin(valid_group_ids)].copy()

# --- NEW: 2. Split groups into train and test sets ---
# We split the *groups* to prevent data leakage.
# shuffle=False maintains the chronological split (like the original script)
train_group_ids, test_group_ids = train_test_split(
    valid_group_ids, 
    test_size=0.2, 
    random_state=42, 
    shuffle=False
)

df_train_groups = df_valid[df_valid['group'].isin(train_group_ids)]
df_test_groups = df_valid[df_valid['group'].isin(test_group_ids)]

print(f"Using {len(train_group_ids)} groups for Training.")
print(f"Using {len(test_group_ids)} groups for Testing.")

# --- NEW: 3. Fit scalers *only* on training data ---
print("Fitting scalers on training group data...")
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# Fit on the features and target from the training groups
scaler_x.fit(df_train_groups[features_for_model].values)
scaler_y.fit(df_train_groups[[target_feature]].values) # Note: Needs 2D array

# --- Functions for data transformation ---
def lstm_gru_data_transform(x_data_input, y_data_input, num_steps=100, n_future_horizons=1):
    """ Changes data to the format for LSTM training
    for sliding window approach, now for multi-step output """
    X, y = list(), list()
    # Check if we have enough data to create even one sequence
    if x_data_input.shape[0] < num_steps + n_future_horizons:
        return np.array(X), np.array(y) # Return empty arrays
        
    for i in range(x_data_input.shape[0]):
        end_ix = i + num_steps
        out_end_ix = end_ix + n_future_horizons -1 # Adjusted for multi-step future output

        if out_end_ix >= x_data_input.shape[0]: # Check if we have enough data for input and output sequence
            break

        seq_X = x_data_input[i:end_ix]
        seq_y = y_data_input[end_ix : end_ix + n_future_horizons, 0] # Extract n_future_horizons steps for y
        X.append(seq_X)
        y.append(seq_y)
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array


# --- NEW: 4. Create sequences from groups ---
print("Creating sequences from training groups...")
all_X_train, all_y_train = [], []

for group_id in train_group_ids:
    group_df = df_train_groups[df_train_groups['group'] == group_id]
    
    # Scale the data for this group
    x_data_group = group_df[features_for_model].values
    y_data_group = group_df[[target_feature]].values
    
    x_scaled = scaler_x.transform(x_data_group)
    y_scaled = scaler_y.transform(y_data_group)
    
    # Create sequences from this scaled group
    X_group, y_group = lstm_gru_data_transform(
        x_scaled, y_scaled, num_steps, N_FUTURE_HORIZONS
    )
    
    if X_group.shape[0] > 0:
        all_X_train.append(X_group)
        all_y_train.append(y_group)

if not all_X_train:
    print("Error: No training sequences were generated. Check data and group sizes. Exiting.")
    sys.exit(1)

# Concatenate all sequences from all training groups
x_train_transformed = np.concatenate(all_X_train, axis=0)
y_train_transformed = np.concatenate(all_y_train, axis=0)
print(f"Total training sequences generated: X={x_train_transformed.shape}, y={y_train_transformed.shape}")

print("Creating sequences from testing groups...")
all_X_test, all_y_test = [], []

for group_id in test_group_ids:
    group_df = df_test_groups[df_test_groups['group'] == group_id]
    
    # Scale the data for this group
    x_data_group = group_df[features_for_model].values
    y_data_group = group_df[[target_feature]].values
    
    x_scaled = scaler_x.transform(x_data_group)
    y_scaled = scaler_y.transform(y_data_group)
    
    # Create sequences from this scaled group
    X_group, y_group = lstm_gru_data_transform(
        x_scaled, y_scaled, num_steps, N_FUTURE_HORIZONS
    )
    
    if X_group.shape[0] > 0:
        all_X_test.append(X_group)
        all_y_test.append(y_group)

if not all_X_test:
    print("Warning: No testing sequences were generated. Validation will be empty.")
    # Create empty arrays with correct shapes for the model
    x_test_transformed = np.empty((0, num_steps, num_features))
    y_test_transformed = np.empty((0, N_FUTURE_HORIZONS))
else:
    # Concatenate all sequences from all testing groups
    x_test_transformed = np.concatenate(all_X_test, axis=0)
    y_test_transformed = np.concatenate(all_y_test, axis=0)
print(f"Total testing sequences generated: X={x_test_transformed.shape}, y={y_test_transformed.shape}")


# --- Model Building Function for Tuning ---
def build_model(input_shape, n_outputs, hp):
    """
    Builds and compiles the GRU model from a hyperparameter dictionary (hp).
    """
    model = Sequential()

    num_layers = hp['num_layers']
    dropout_rate = hp['dropout_rate']
    gru_activation = hp['gru_activation']

    for i in range(num_layers):
        hp_units = hp[f'units_layer_{i+1}']
        is_last_layer = (i == num_layers - 1)
        return_sequences = not is_last_layer # Only the last GRU layer should not return sequences

        if i == 0:
            model.add(GRU(units=hp_units, activation=gru_activation,
                          input_shape=input_shape,
                          return_sequences=return_sequences,
                          dropout=dropout_rate,
                          recurrent_dropout=dropout_rate))
        else:
            model.add(GRU(units=hp_units, activation=gru_activation,
                          return_sequences=return_sequences,
                          dropout=dropout_rate,
                          recurrent_dropout=dropout_rate))

    # Add a Dropout layer before the final Dense layer for more regularization
    # This is a common practice to prevent overfitting on the dense output layer
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=n_outputs)) # Output size is N_FUTURE_HORIZONS

    learning_rate = hp['learning_rate']
    dilate_alpha = hp['dilate_alpha']

    dilate_loss_fn = DilateLoss(alpha=dilate_alpha)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=dilate_loss_fn) # Use DilateLoss for tuning

    return model

# --- Main Execution (Now incorporating tuning) ---

if __name__ == '__main__':
    # Early stopping callback for tuning trials
    early_stopping_tuning = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=15, # Increased patience for thorough search
        verbose=0, # Set verbose=0 to keep trial logs clean
        mode='min',
        restore_best_weights=True
    )

    input_shape = (x_train_transformed.shape[1], x_train_transformed.shape[2])

    print("\n--- Starting Manual Hyperparameter Search (No KerasTuner) ---")

    # Define the full search space for this model
    possible_learning_rates = [1e-3, 5e-4, 1e-4, 5e-5] # Adjusted common range
    possible_num_layers = [1, 2, 3]
    possible_units = [32, 64, 96, 128] # Adjusted range for GRU units
    possible_dropout_rates = [0.1, 0.2, 0.3, 0.4] # Added dropout
    possible_gru_activations = ['tanh', 'relu'] # Added GRU activation
    possible_dilate_alphas = [0.3, 0.5, 0.7, 0.9] # Added DilateLoss alpha

    num_tuning_trials = 20 # Number of hyperparameter combinations to try
    tuning_results = []

    # Store the best model instance directly
    best_tuned_model_instance = None
    best_overall_val_loss_tuning = float('inf')
    
    # Check if we have validation data
    has_validation_data = x_test_transformed.shape[0] > 0
    if not has_validation_data:
        print("--- WARNING: No validation data. Model will be trained without validation. ---")

    for trial in range(num_tuning_trials):
        print(f"\n--- Tuning Trial {trial + 1}/{num_tuning_trials} ---")

        # 1. Randomly select hyperparameters
        hp = {}
        hp['learning_rate'] = random.choice(possible_learning_rates)
        hp['num_layers'] = random.choice(possible_num_layers)
        hp['dropout_rate'] = random.choice(possible_dropout_rates)
        hp['gru_activation'] = random.choice(possible_gru_activations)
        hp['dilate_alpha'] = random.choice(possible_dilate_alphas)

        for i in range(hp['num_layers']):
            hp[f'units_layer_{i+1}'] = random.choice(possible_units)

        print(f"Testing parameters: {hp}")

        # 2. Build and train the model
        K.clear_session() # Clear session to prevent memory leaks

        model_trial = build_model(input_shape, N_FUTURE_HORIZONS, hp)

        try:
            # Conditionally add validation data if it exists
            validation_data_arg = (x_test_transformed, y_test_transformed) if has_validation_data else None
            
            history = model_trial.fit(x_train_transformed, y_train_transformed,
                                      epochs=100, # High epoch count, EarlyStopping will handle it
                                      batch_size=32,
                                      validation_data=validation_data_arg, 
                                      callbacks=[early_stopping_tuning] if has_validation_data else None, # Only use callback if validating
                                      verbose=0) # Set verbose=0 to keep the log clean during trials

            # Evaluate the best model (weights restored by EarlyStopping) on the validation set
            if has_validation_data:
                current_val_loss = model_trial.evaluate(x_test_transformed, y_test_transformed, verbose=0)
                print(f"Trial {trial + 1} Best val_loss (on x_test_transformed): {current_val_loss:.4f}")
            else:
                # If no validation, just use the final training loss
                current_val_loss = history.history['loss'][-1]
                print(f"Trial {trial + 1} Final train_loss (no validation): {current_val_loss:.4f}")

            tuning_results.append((current_val_loss, hp))

            if current_val_loss < best_overall_val_loss_tuning:
                best_overall_val_loss_tuning = current_val_loss
                best_tuned_model_instance = model_trial # Store the model instance

        except Exception as e:
            print(f"!!! Tuning Trial {trial + 1} failed with error: {e} !!!")
            print("Skipping this trial.")
            tuning_results.append((float('inf'), hp)) # Append high loss for failed trials

    print("\n--- Manual Search Complete ---")

    # Get the optimal hyperparameters
    tuning_results.sort(key=lambda x: x[0]) # Sort by val_loss (ascending)

    if not tuning_results or tuning_results[0][0] == float('inf'):
        print("All tuning trials failed or no valid models were trained. Cannot proceed.")
        sys.exit(1)

    best_val_loss_from_search, best_hps = tuning_results[0]

    print(f"""
    The hyperparameter search is complete.
    Best Validation Loss achieved during search: {best_val_loss_from_search:.4f}
    The optimal parameters are: {best_hps}
    """)

    # --- Use the Best Model Found (Already Trained) ---
    # `best_tuned_model_instance` already holds the weights restored by EarlyStopping.
    # We can use this directly.
    if best_tuned_model_instance:
        loaded_model = best_tuned_model_instance
        print("\nUsing the best model instance found during the search.")
    else:
        # This fallback should ideally not be hit if tuning_results is not empty
        print("\nBuilding and retraining the best model from scratch (fallback). This should only happen if `best_tuned_model_instance` was not captured.")
        K.clear_session()
        loaded_model = build_model(input_shape, N_FUTURE_HORIZONS, best_hps)
        # Train with verbose output this time
        early_stopping_final_train = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=15, # Use same patience as tuning
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        
        validation_data_arg = (x_test_transformed, y_test_transformed) if has_validation_data else None

        loaded_model.fit(x_train_transformed, y_train_transformed,
                         epochs=200, # Allow more epochs for final training
                         batch_size=32,
                         validation_data=validation_data_arg,
                         callbacks=[early_stopping_final_train] if has_validation_data else None)
        print("Final training of best model complete.")


    # --- 5. Save the Best Model ---
    print("\n--- Saving the Best Model to disk ---")
    model_name_suffix = (
        f"lr{best_hps['learning_rate']}_layers{best_hps['num_layers']}"
        f"_units{best_hps.get('units_layer_1', 'N/A')}_dr{best_hps['dropout_rate']}"
        f"_act{best_hps['gru_activation']}_alpha{best_hps['dilate_alpha']}"
    )
    # --- ADDED '_grouped' to filenames ---
    model_json_saved = f'gru_multi_steps_{num_steps}past_{N_FUTURE_HORIZONS}future_tuned_grouped_{model_name_suffix}.json'
    weights_file = f'gru_multi_steps_{num_steps}past_{N_FUTURE_HORIZONS}future_tuned_grouped_{model_name_suffix}.h5'

    fileout = loaded_model.to_json()
    with open(model_json_saved, "w") as json_file:
        json_file.write(fileout)
    loaded_model.save_weights(weights_file)
    print(f"Saved best model and weights to disk: {model_json_saved} | {weights_file}")


    # --- WARNING: The function 'predict_future_water_level' is not defined in this script. ---
    # --- The following sections (6, 7, and final evaluation) are COMMENTED OUT ---
    # --- because they will cause a NameError. ---
    
    # # --- 6. Perform a future prediction using the tuned model ---
    # print("\n--- Starting future water level forecast simulation with tuned model ---")
    #
    # # Find the latest timestamp in your observation data
    # last_obs_timestamp = df_obs.index[-1]
    # print(f"Last observation timestamp from df_obs: {last_obs_timestamp}")
    #
    # print(f"df_nwp_raw first timestamp: {df_nwp_raw.index.min()}")
    # print(f"df_nwp_raw last timestamp: {df_nwp_raw.index.max()}")
    #
    # # Determine the actual number of horizons we can forecast based on available NWP data
    # # We are taking the last N_FUTURE_HORIZONS from the df_nwp_raw for *simulation* purposes
    # actual_forecast_length_for_simulation = min(N_FUTURE_HORIZONS, len(df_nwp_raw))
    #
    # if actual_forecast_length_for_simulation == 0:
    #     print("Error: No NWP forecast data available in df_nwp_raw to simulate future forecasts.")
    #     print("Please ensure your df_nwp_raw file has data.")
    #     sys.exit(1)
    #
    # nwp_future_forecast_df = df_nwp_raw.iloc[-actual_forecast_length_for_simulation:]
    #
    # print(f"Simulating future forecasts using {actual_forecast_length_for_simulation} entries from df_nwp_raw.")
    # print(f"Simulated NWP forecast data range: {nwp_future_forecast_df.index.min()} to {nwp_future_forecast_df.index.max()}")
    #
    # # Extract the initial observations needed for the first `num_steps` input sequence
    # initial_observations_for_forecast = df_obs.loc[df_obs.index <= last_obs_timestamp].tail(num_steps)
    #
    # # Perform the future prediction
    # # !!! THIS FUNCTION IS NOT DEFINED !!!
    # predicted_water_levels_list, full_predicted_sequence_df = predict_future_water_level(
    #     loaded_model, # Use the tuned model here
    #     initial_observations_for_forecast,
    #     nwp_future_forecast_df,
    #     scaler_x,
    #     scaler_y,
    #     num_steps,
    #     features_for_model,
    #     target_feature
    # )
    #
    # print("\nFuture Water Level Predictions (from tuned model):")
    # forecast_output_df = pd.DataFrame({
    #     'timestamp': nwp_future_forecast_df.index[:len(predicted_water_levels_list)],
    #     'predicted_water_level': predicted_water_levels_list
    # })
    # print(forecast_output_df)
    #
    # # --- 7. Plotting Results ---
    # print("\n--- Generating plot of observed vs. predicted water levels ---")
    # plt.figure(figsize=(15, 7))
    #
    # # Plot recent observed data
    # recent_obs_df = df_obs.tail(200) # Show last 200 hours of observations
    # plt.plot(recent_obs_df.index, recent_obs_df[target_feature], label='Observed Water Level', color='blue', alpha=0.7)
    #
    # # Plot the predicted future water levels
    # plt.plot(forecast_output_df['timestamp'], forecast_output_df['predicted_water_level'],
    #          label=f'Predicted Water Level (Next {actual_forecast_length_for_simulation} hours)',
    #          color='red', linestyle='--', marker='o', markersize=4)
    #
    # # Add a vertical line at the forecast start point for clarity
    # forecast_start_point = initial_observations_for_forecast.index[-1]
    # plt.axvline(x=forecast_start_point, color='green', linestyle=':', label='Forecast Start')
    #
    # plt.xlabel('Timestamp')
    # plt.ylabel('Water Level (m)')
    # plt.title(f'Water Level Forecast with Tuned GRU Model ({num_steps} past, {N_FUTURE_HORIZONS} future)')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout() # Adjust layout to prevent labels overlapping
    #
    # filename = f'gru_gfs_obs_prediction_tuned_num_steps_{num_steps}_future_{N_FUTURE_HORIZONS}.png'
    # plt.savefig(filename)
    # print(f"Plot saved to {filename}")
    # plt.show()
    #
    # # Calculate and print metrics on the test set for the best model
    # # Note: `loaded_model` here is the `best_tuned_model_instance`
    # if has_validation_data:
    #     print("\n--- Model Evaluation on Test Set (using best tuned model) ---")
    #     y_pred_test_scaled = loaded_model.predict(x_test_transformed, verbose=0)
    #     y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)
    #     y_true_test = scaler_y.inverse_transform(y_test_transformed)
    #
    #     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    #
    #     mae = mean_absolute_error(y_true_test, y_pred_test)
    #     rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    #     r2 = r2_score(y_true_test, y_pred_test)
    #
    #     print(f"Mean Absolute Error (MAE): {mae:.4f} m")
    #     print(f"Root Mean Squared Error (RMSE): {rmse:.4f} m")
    #     print(f"R-squared (R2): {r2:.4f}")
    # else:
    #     print("\n--- No Test Set for Final Evaluation ---")
