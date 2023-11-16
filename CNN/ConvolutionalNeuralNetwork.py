# CNN for prediction of Angle of Attack of airfoils from images, predicting the correct AoA at least 99% of the time.
# Database of over 6600 NACA 4-digit code airfoils. Airfoils have Reynolds of 50000, Mach of 0.3 and an AoA between -3° and 3°.

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset = pd.read_csv("airfoils_Re_03.csv")
dataset['AoA'] = dataset['AoA'].astype(str)
af_df = pd.DataFrame(dataset)
my_df_train = af_df.sample(frac=0.7,random_state=0)
my_df_test = af_df.drop(my_df_train.index)

indx = random.randint(0, len(my_df_train))
img = plt.imread(my_df_train.fpath[indx])
plt.imshow(img)
plt.show()
my_df_train.shape

datagen = ImageDataGenerator(rescale=1. / 255,
                             validation_split=0.3, )

# Parameters
img_size = (250, 350)
batch_size = 32

# Training Data Generator
train_generator = datagen.flow_from_dataframe(
    dataframe=my_df_train,
    directory=None,
    x_col="fpath",  # Change this to the actual column containing image paths
    y_col="AoA",  # Change this to the actual column containing labels
    target_size=img_size,
    class_mode='categorical',  # Change this based on your task (e.g., 'binary' for binary classification)
    batch_size=batch_size,
    subset='training'
)

# Validation Data Generator
valid_generator = datagen.flow_from_dataframe(
    dataframe=my_df_train,
    directory=None,
    x_col="fpath",
    y_col="AoA",
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size,
    subset = 'validation'
)

im_width  = train_generator.next()[0].shape[1]
im_height = train_generator.next()[0].shape[2]

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(6, (3, 3), activation='relu', input_shape=(im_width, im_height, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(6, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
              metrics=['accuracy'])

start_time = time.time()

history = model.fit(train_generator,
                    epochs = 4,
                    validation_data = valid_generator,
                    verbose = 1)
end_time = time.time()
runtime = end_time - start_time