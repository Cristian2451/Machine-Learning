import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
sns.set(color_codes = True)

merged_data=pd.read_csv("merged_dataset_BearingTest_2.csv", index_col=0)
merged_data.plot(figsize = (18,6))
#merged_data.head()


dataset_train = merged_data['2004-02-12 11:02:39':'2004-02-13 23:52:39']
dataset_test = merged_data['2004-02-13 23:52:39':]
dataset_train.plot(figsize = (12,6))
dataset_train.head()


X_train = merged_data['2004-02-12 11:02:39':'2004-02-13 23:52:39']
X_test = merged_data['2004-02-13 23:52:39':]

model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation = 'elu', kernel_initializer = 'glorot_uniform',
                              input_shape = (X_train.shape[1],)),
        tf.keras.layers.Dense( 2, activation = 'elu', kernel_initializer = 'glorot_uniform'),
        tf.keras.layers.Dense(10, activation = 'elu', kernel_initializer = 'glorot_uniform'),
        tf.keras.layers.Dense(X_train.shape[1], kernel_initializer = 'glorot_uniform')])

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse', 'mape'])
model.summary()

history = model.fit(np.array(X_train),np.array(X_train), batch_size = 10, epochs = 200,
                    validation_split = 0.05, verbose = 1)

plt.plot(history.history['loss'],'b', label = 'Training loss')
plt.plot(history.history['val_loss'],'r', label = 'Validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.ylim([0,.000005])
plt.show()

X_pred = model.predict(np.array(X_train))
X_pred = pd.DataFrame(X_pred, columns = X_train.columns)
X_pred.index = X_train.index

scored = pd.DataFrame(index = X_train.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred - X_train), axis = 1) # this is the difference between output and original train data

plt.figure(figsize = (15,8))
sns.displot(scored['Loss_mae'], bins = 10, kde = True, color = 'blue')
plt.xticks(rotation = 45)
plt.show()

X_pred = model.predict(np.array(X_test))
X_pred = pd.DataFrame(X_pred, columns = X_test.columns)
X_pred.index = X_test.index

predictions = pd.DataFrame(index = X_test.index)
predictions['Loss_mae'] = np.mean(np.abs(X_pred - X_test), axis = 1)
predictions['Threshold'] = 0.003
predictions['Anomaly'] = predictions['Loss_mae'] > predictions['Threshold']
predictions.sample(10)

anomaly_alldata = pd.concat([scored, predictions])

# plot predictions
predictions.plot(logy = True, color = ['blue','red'])
plt.xticks(rotation = 45)
plt.show()

# Check start of anomalies
start_anomaly = np.where(predictions['Anomaly'])[0]
print('First anomaly detected {}'.format(dataset_test.index[start_anomaly[0]]))

_, axs = plt.subplots(nrows=2, ncols=1, sharex='col', figsize=(10, 8), layout='tight')
# plot test dataset
# plt.sca(axs[0])
dataset_test.plot(ax=axs[0])

# plot predictions
predictions.plot(logy = True,  ax=axs[1],  color = ['blue','red'])
plt.xticks(rotation = 45)

for ax in axs:
    ax.axvline( ls='-.', color='orange', alpha=0.4, x=start_anomaly[0])
plt.show()




