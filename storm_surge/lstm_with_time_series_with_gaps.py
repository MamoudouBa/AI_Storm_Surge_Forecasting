#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---
## 1. Setup and Data
# ---

# =============================================================================
# NEW FIX APPLIED HERE
# =============================================================================

# Load the data. Pandas will correctly use the first row as the header
# since we saw it in your head() output.
df = pd.read_csv(
    'surge_training_datasets.csv'
)

print(f"Loaded data with {len(df)} total rows.")

# ---
# Explicitly convert the 'timestamp' column WITH THE CORRECT FORMAT
# ---
# Your format is "YYYY-MM-DD HH MM SS"
date_format = "%Y-%m-%d %H %M %S"
df['timestamp'] = pd.to_datetime(df['timestamp'], format=date_format, errors='coerce')

# ---
# Drop any rows that *still* failed (e.g., "N/A" or corrupted text)
# ---
original_rows = len(df)
df = df.dropna(subset=['timestamp'])
new_rows = len(df)

if original_rows > new_rows:
    print(f"Dropped {original_rows - new_rows} rows with un-parseable dates.")

# This should now print a non-zero number
print(f"Data is clean and ready with {len(df)} rows.")

# =============================================================================
# END OF FIX
# =============================================================================


# ---
## 2. Split Data into Separate Sequences
# ---

# We identify a "new" sequence by finding where the time difference
# between rows is greater than our expected frequency (e.g., 1 hour)
time_diff = df['timestamp'].diff()

# Find the indices where a gap occurs (diff > 1 hour)
sequence_id = (time_diff > pd.Timedelta('1 hour')).cumsum()

# Group the DataFrame by this new sequence ID
groups = df.groupby(sequence_id)

# Create a list of numpy arrays. Each array is one continuous sequence.
sequences_raw = [group.drop(columns=['timestamp']).values for _, group in groups]

print(f"Split data into {len(sequences_raw)} separate sequences:")
for i, seq in enumerate(sequences_raw):
    print(f"  Sequence {i}: {seq.shape[0]} steps")

# ---
## 3. Scale and Create Windows
# ---

# Model parameters
N_STEPS_IN = 5  # Use 5 hours of history
N_STEPS_OUT = 1 # To predict 1 hour in the future
N_FEATURES = 6  # We have 6 features in the data

# It's important to fit the scaler on ALL data to have a consistent
# 0-1 range across all sequences.
scaler = MinMaxScaler()
# This variable should now have data
all_data_for_scaling = df.drop(columns=['timestamp']).values
scaler.fit(all_data_for_scaling)

# Now, scale each sequence individually
sequences_scaled = [scaler.transform(seq) for seq in sequences_raw]

# This function creates sliding windows from a single sequence
def create_windows(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        # Check if we are beyond the sequence
        if out_end_ix > len(data):
            break

        # Gather input and output parts
        seq_x = data[i:end_ix]
        seq_y = data[end_ix:out_end_ix]

        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Iterate over each of our 3 sequences and create windows
all_X, all_y = [], []
for seq_scaled in sequences_scaled:
    # Only create windows if the sequence is long enough
    if len(seq_scaled) >= N_STEPS_IN + N_STEPS_OUT:
        X, y = create_windows(seq_scaled, N_STEPS_IN, N_STEPS_OUT)
        all_X.append(X)
        all_y.append(y)

# Stack the windows from all sequences into one big training set
X_train = np.vstack(all_X)

if N_STEPS_OUT == 1:
    y_train = np.vstack(all_y).reshape(-1, N_FEATURES)
else:
    y_train = np.vstack(all_y) 

print(f"\nCreated training data with shapes:")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")

# ---
## 4. Build and Train the LSTM Model
# ---

model = Sequential()
model.add(LSTM(
    50,
    activation='relu',
    input_shape=(N_STEPS_IN, N_FEATURES)
))
model.add(Dense(
    N_FEATURES
))

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
print("\n--- Training Model ---")

# =============================================================================
# TYPO FIX: Corrected 'X_Ttrain' to 'X_train'
# =============================================================================
model.fit(
    X_train,
    y_train,
    epochs=100,
    verbose=1,
    batch_size=8
)

print("--- Model Training Complete ---")

# ---
## 5. Make a Prediction (Example)
# ---

test_input = X_train[0].reshape((1, N_STEPS_IN, N_FEATURES))
predicted_scaled = model.predict(test_input)

predicted_actual = scaler.inverse_transform(predicted_scaled)
actual_scaled = y_train[0].reshape(1, -1)
actual_actual = scaler.inverse_transform(actual_scaled)

print("\n--- Example Prediction ---")
print(f"Actual (scaled):    {actual_scaled[0]}")
print(f"Predicted (scaled): {predicted_scaled[0]}")
print("\n---")
print(f"Actual (real values):    {actual_actual[0]}")
print(f"Predicted (real values): {predicted_actual[0]}")
