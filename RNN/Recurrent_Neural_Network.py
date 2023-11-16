# Recurrent Neural Network that predicts the air speed based on past measurements at the same point.
# Data collected in a 10x5 wind tunnel using single hot-wire anemometry in the study of flow with tripped Reynolds number boundary layer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Split the dataset into windows of size window_size + 1 (1 extra for the label)
    # Shift: how many steps forward, stride: every how many points to consider; 1==all.
    dataset = dataset.window(window_size + 1, shift=1, stride=1, drop_remainder=True)

    # Flatten the dataset of windows and group them into batches of size window_size + 1
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Split the flattened dataset into features (first window_size values: window[:-1])
    # and label (last value: window[-1]))
    # The shuffle method shuffles the order each windows appear. The content within the windows is not shuffled.
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))

    # Group data into batches of size batch_size and prefetch one batch at a time to optimize GPU utilization
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

import h5py
with h5py.File('hot_wire_data.mat', 'r') as f:
    U = list(f['U'])
    freq = list(f['samp_rate'])

split_ratio = 0.8
x = np.array(U)
x_data = x[:, 0:10000]
n = len(x_data[1, :])
x_train = x_data[1, 0:int(n * split_ratio)]
x_valid = x_data[1, int(n * split_ratio):]
time_freq = np.array(freq)
time_data = np.zeros(n)
for i in range(0, n - 1):
    time_data[i + 1] = time_data[i] + 1 / time_freq
time_train = time_data[0:int(n * split_ratio)]
time_valid = time_data[int(n * split_ratio):]

window_size = 50
batch_size = 16
shuffle_buffer_size = 500

train_dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
test_dataset = windowed_dataset(x_valid, window_size, batch_size, 1)

model = tf.keras.models.Sequential([
    # Add a Lambda layer to expand the dimensions of the input tensor
    # (to account for the time dimension in the input sequence)
    # Dataset is 2D and here we make it 3D which is required by the SimpleRNN layer.
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),

    # Add an LSTM layer with 40 units and return sequences for use in subsequent layers
    tf.keras.layers.LSTM(40, return_sequences=False),

    # Add a Dense layer with 1 unit (to output the predicted value)
    tf.keras.layers.Dense(1),

    # Add a Lambda layer to scale the output by 100
    # (since the values of the target variable are in the range of 0-100)
    tf.keras.layers.Lambda(lambda x: x * 100.0)  ## can be removed, BUT would take much longer to train.
])

# Define the optimizer used for training the model
optimizer = tf.keras.optimizers.Adam()

# Compile the model with the optimizer, Huber loss function, and mean absolute error metric
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.Huber(),
              metrics=['mape'])

start_time = time.time()
history = model.fit(train_dataset, epochs = 50)
end_time = time.time()
runtime = end_time-start_time

epochs = range(len(history.history['loss']))
plt.plot(epochs, history.history['mape'], label = 'Training')
plt.title('Training loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

forecast = model.predict(test_dataset)
plt.figure(figsize=(10, 6))
plt.plot(time_valid[window_size:],x_valid[window_size:], label='data')
plt.plot(time_valid[window_size:],forecast, label='RNN prediction on validation data')
plt.xlabel('time step')
plt.ylabel('label')
plt.title('RNN prediction')
plt.legend()
plt.show()