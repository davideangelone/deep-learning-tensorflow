# -*- coding: utf-8 -*-
"""
Created on Fri May 28 13:49:00 2021

@author: use
"""


import keras
keras.__version__
print(keras.__version__)

from keras.datasets import mnist
from keras import models
from keras import layers

# load data from MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preparess the training and test sets, reshaping them
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

print('train images shape', train_images.shape)
print('len train images shape', len(train_images))
print('train labels', train_labels)

print('test images shape', test_images.shape)
print('len test images shape', len(test_images))
print('test labels', test_labels)

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator(rescale = 1.0/255.)
test_datagen = ImageDataGenerator(rescale = 1.0/255.)


# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow(x=train_images, y=train_labels,
                                     batch_size=20,
                                     #class_mode='categorical'
                                     )     
# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow(x=test_images, y=test_labels,
                                          batch_size=20,
                                          #class_mode='categorical'
                                          )


# build an artificial neural network (ANN)
network = models.Sequential(name='Sequential_model')

network.add(layers.Conv2D(name='primo_strato_Conv2D', filters=16, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
network.add(layers.MaxPool2D(name='primo_MaxPool2D', pool_size=(2,2)))
network.add(layers.Conv2D(name='secondo_strato_Conv2D', filters=32, kernel_size=(3,3), activation='relu'))
network.add(layers.MaxPool2D(name='secondo_MaxPool2D', pool_size=(2,2)))

network.add(layers.Flatten())
network.add(layers.Dense(name='primo_strato_Dense', units=1024, activation='relu'))
network.add(layers.Dense(name='secondo_strato_Dense', units=512, activation='relu'))
network.add(layers.Dense(name='terzo_strato_Dense', units=10, activation='softmax'))
print(network.summary())

#from tensorflow import keras as k
#k.utils.plot_model(network, show_shapes=True)

# make the ANN operational 'compiling' it
network.compile(optimizer='rmsprop',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

from tensorflow.keras.utils import to_categorical

# converts the target variable values into a suitable form
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(train_labels)

# train ANN
#history = network.fit(train_images, train_labels, epochs=25, batch_size=1024)
history = network.fit(x=train_generator, 
                      validation_data=validation_generator,
                      steps_per_epoch=100,
                      epochs=25,
                      validation_steps=50,
                      verbose=2)


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_acc, 'g', label='Test accuracy')
plt.plot(epochs, val_loss, 'm', label='Test loss')
plt.title('Accuracy and loss')
plt.legend(loc=0)
plt.grid(True)
plt.figure(figsize=(10,5))

plt.show()
