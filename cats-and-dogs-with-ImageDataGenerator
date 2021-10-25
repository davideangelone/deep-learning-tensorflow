# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 08:37:11 2021

@author: dangelone
"""

#Disable GPU for memory issues (if needed)
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0" #0=DEBUG (default), 1=INFO, 2=WARINIG, 3=ERROR

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

print(keras.__version__)

#Download cats and dogs dataset
!wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

import os
import zipfile

local_zip = './cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()

petImagesPath = './cats_and_dogs_filtered'

image_size = (180, 180)
batch_size = 32

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.,
        rotation_range=40, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True, 
        fill_mode='nearest',         
        validation_split=0.2
        )
test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.,
        validation_split=0.2)


# --------------------
# Flow training images in batches using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory (
                                     directory=petImagesPath,
                                     class_mode='binary',
                                     color_mode='rgb',
                                     subset="training",
                                     seed=1337,
                                     target_size=image_size,
                                     batch_size=batch_size,
                                     )     
# --------------------
# Flow validation images in batches using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(
                                     directory=petImagesPath,
                                     color_mode='rgb',
                                     class_mode='binary',
                                     subset="validation",
                                     seed=1337,
                                     target_size=image_size,
                                     batch_size=batch_size,
                                     )

network = models.Sequential(name='Sequential_model')

network.add(layers.Conv2D(
    name='primo_strato_Conv2D', 
    filters=32, kernel_size=(3,3), 
    activation='relu', 
    padding='same',
    strides=2,
    input_shape=(180, 180, 3))
    )
network.add(layers.MaxPool2D(name='primo_MaxPool2D', pool_size=(2, 2)))
network.add(layers.Conv2D(
    name='secondo_strato_Conv2D', 
    filters=64, kernel_size=(3,3), 
    activation='relu', 
    padding='same',
    input_shape=(180, 180, 3))
    )
network.add(layers.MaxPool2D(name='secondo_MaxPool2D', pool_size=(2, 2)))

network.add(layers.Flatten())
network.add(layers.Dense(name='primo_strato_Dense', units=1024, activation='relu'))
network.add(layers.Dense(name='secondo_strato_Dense', units=512, activation='relu'))
network.add(layers.Dropout(name='dropout_strato_Dense', rate=.2))
network.add(layers.Dense(name='terzo_strato_Dense', units=1, activation='sigmoid'))
print(network.summary())

network.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = network.fit(x=train_generator,
                      validation_data=validation_generator,
                      #steps_per_epoch=100,
                      epochs=25,
                      #validation_steps=50,
                      verbose=1)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

import matplotlib.pyplot as plt

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_acc, 'g', label='Test accuracy')
plt.plot(epochs, val_loss, 'm', label='Test loss')
plt.title('Accuracy and loss')
plt.legend(loc=0)
plt.grid(True)
plt.figure(figsize=(10, 5))

plt.show()

