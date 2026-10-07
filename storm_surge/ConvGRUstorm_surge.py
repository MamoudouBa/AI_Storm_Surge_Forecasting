#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ConvGRU2D, TimeDistributed
from tensorflow.keras.models import Model

# Custom ConvGRU Cell Implementation
class ConvGRUCell(Layer):
    """A simplified Convolutional GRU cell."""
    def __init__(self, filters, kernel_size, **kwargs):
        super(ConvGRUCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.state_size = (None, None, filters) # Height, Width, Channels

    def build(self, input_shape):
        # Convolutional layer for reset and update gates (z, r)
        self.gate_conv = Conv2D(
            filters=2 * self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid'
        )
        # Convolutional layer for the candidate hidden state (h_tilde)
        self.candidate_conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation='tanh'
        )
        super(ConvGRUCell, self).build(input_shape)

    def call(self, inputs, states):
        # `states` is a list containing the previous hidden state
        prev_h = states[0]
        
        # Concatenate input and previous hidden state
        x = tf.concat([inputs, prev_h], axis=-1)
        
        # Calculate gates
        gates = self.gate_conv(x)
        z, r = tf.split(gates, 2, axis=-1) # z: update gate, r: reset gate
        
        # Calculate candidate hidden state
        r_h = r * prev_h
        h_tilde_input = tf.concat([inputs, r_h], axis=-1)
        h_tilde = self.candidate_conv(h_tilde_input)
        
        # Calculate new hidden state
        new_h = z * prev_h + (1 - z) * h_tilde
        
        return new_h, [new_h]


### MODEL DEFINITION ###
def build_convgru_model(input_shape):
    """
    Builds the ConvGRU model for spatio-temporal forecasting.
    
    Args:
        input_shape (tuple): Shape of the input data (timesteps, height, width, channels).
    
    Returns:
        A Keras Model instance.
    """
    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Use an RNN layer to wrap our custom ConvGRU cell
    # This will run the cell over the time dimension
    # return_sequences=False because we want the final output for prediction
    '''
    gru_layer = tf.keras.layers.RNN(
        ConvGRUCell(filters=32, kernel_size=(3, 3)),
        return_sequences=False
    )(input_layer)
    '''
    gru_layer = tf.keras.layers.ConvGRU2D(filters=32, kernel_size=(3, 3), 
                # This is the correct 5D shape (timesteps, height, width, channels)
                return_sequences=False)(input_layer)

    # Add a final Conv2D layer to generate the output grid.
    # The number of filters should match the number of output variables you want to predict.
    # Here we predict all 6 variables for the next timestep.
    output_layer = Conv2D(
        filters=6, # Predicting all 6 channels for the next step
        kernel_size=(1, 1),
        activation='linear', # 'linear' for regression tasks like this
        padding='same'
    )(gru_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

### EXAMPLE USAGE ###
# 1. Define data parameters
timesteps = 10      # Use 10 past time steps to predict the next one
height = 64         # 64x64 grid
width = 64
channels = 6        # Your 6 weather variables
batch_size = 4

input_data = np.random.rand(batch_size,timesteps, height, width, channels).astype(np.float32)

# Input shape for the model
#input_shape = (timesteps, height, width, channels)
input_shape = input_data.shape
print('****  input_shape ***', input_shape)

# 2. Build the model
model = build_convgru_model(input_shape)

# 3. Compile the model
# Use a regression loss function like Mean Squared Error
model.compile(optimizer='adam', loss='mse')

# 4. Print model summary
print("✅ Model Built Successfully!")
model.summary()


# 5. Generate some dummy data to test the model's forward pass
# Batch size of 4 for demonstration
#batch_size = 4
dummy_input = np.random.rand(batch_size, timesteps, height, width, channels).astype(np.float32)

# Make a prediction
dummy_prediction = model.predict(dummy_input)

print(f"\n➡️ Input data shape: {dummy_input.shape}")
print(f"⬅️ Predicted output shape: {dummy_prediction.shape}")
