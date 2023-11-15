import tensorflow as tf
import pandas as pd
import time
import matplotlib.pyplot as plt

# Neural network that predicts mechanical properties of fibre composites
# The dataset 'CNT_data.csv' contains:
# - 200 PCA components for each microstructure (fibres with different quantity and thickness embedded in a matrix) calculated from the 2-point correlation function
# - geometric properties of the microstructure
# - mechanical properties of fibre and matrix
# and response variables:
# - Young's moduli and yield stresses (homogenized transverse Young’s moduli  E22 and E33, the transverse shear modulus G23 and the transverse normal yield strengths σ22 and σ33

dataset = pd.read_csv("CNT_data.csv")
test_dataset = dataset.sample(frac = 0.101, random_state = 0)
train_and_validation_dataset = dataset.drop(test_dataset.index)
train_dataset = train_and_validation_dataset.sample(frac = 0.9, random_state = 0)
val_dataset = train_and_validation_dataset.drop(train_dataset.index)
labels = ['E22', 'E33', 'G23', 'yield22', 'yield33']

train_labels = train_dataset['E22'].copy()
train_dataset = train_dataset.drop(labels, axis = 1)

val_labels = val_dataset['E22'].copy()
val_dataset = val_dataset.drop(labels, axis = 1)

test_labels = test_dataset['E22'].copy()
test_dataset = test_dataset.drop(labels, axis = 1)

activ = 'sigmoid'
Hidden_layers = 1
node_num1 = train_dataset.shape[1]
node_num2 = train_dataset.shape[1]
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(node_num1, activation = activ, input_shape = [train_dataset.shape[1]]))
for j in range(Hidden_layers):
    model.add(tf.keras.layers.Dense(node_num2, activation = activ))
    model.add(tf.keras.layers.Dense(8, activation = 'linear'))

model.summary()

start_time = time.time()
model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0005, momentum = 0.0, nesterov = False),
              loss = 'mse',
              metrics = ['mse'])
history = model.fit(train_dataset, train_labels,
                    batch_size = 32,
                    epochs = 200,
                    validation_data = (val_dataset, val_labels),
                    verbose = 1)
epochs = range(len(history.history['loss']))

end_time = time.time()
runtime = end_time-start_time

plt.plot(epochs, history.history['loss'], label='training')
plt.plot(epochs, history.history['val_loss'], label='validation')
plt.title('Training loss')
plt.show()

model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0005, momentum = 0.0, nesterov = False),
              loss = 'mse',
              metrics = ['mape'])

results_t = model.evaluate(test_dataset, test_labels)
print(results_t)