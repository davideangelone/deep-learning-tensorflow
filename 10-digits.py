# -*- coding: utf-8 -*-

import keras
keras.__version__
print(keras.__version__)

from keras.datasets import mnist
from keras import models
from keras import layers

# load data from MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('original train images shape', train_images.shape)
print('original test images shape', test_images.shape)

# prepares the training and test sets, reshaping them
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

print('train images shape', train_images.shape)
print('len train images shape', len(train_labels))
print('train labels', train_labels)
print('test labels shape', test_labels.shape)
print('len test labels', len(test_labels))
print('test labels', test_labels)

# build an artificial neural network (ANN)
network = models.Sequential(name='Sequential_model')

network.add(layers.Conv2D(name='primo_strato_Conv2D', filters=16, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
network.add(layers.MaxPool2D(name='primo_MaxPool2D', pool_size=(2, 2)))
network.add(layers.Conv2D(name='secondo_strato_Conv2D', filters=32, kernel_size=(3,3), activation='relu'))
network.add(layers.MaxPool2D(name='secondo_MaxPool2D', pool_size=(2, 2)))

network.add(layers.Flatten())
network.add(layers.Dense(name='primo_strato_Dense', units=1024, activation='relu'))
network.add(layers.Dense(name='secondo_strato_Dense', units=512, activation='relu'))
network.add(layers.Dropout(name='dropout_strato_Dense', rate=.2))
network.add(layers.Dense(name='terzo_strato_Dense', units=10, activation='softmax'))
print(network.summary())

#import pydot
#from keras.utils.vis_utils import plot_model
#plot_model(network, show_shapes=True)

# make the ANN operational 'compiling' it
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

from tensorflow.keras.utils import to_categorical

# converts the target variable values into a suitable form
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(train_labels)

# train ANN
history = network.fit(train_images, train_labels, epochs=25, batch_size=1024)

# test the ANN using the test data set
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('Metric output ', network.metrics_names)

# print test results
print('Test accuracy:', test_acc)
print('Test loss: ', test_loss)


import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))
test_accs = [test_acc] * len(acc)
test_losses = [test_loss] * len(loss)

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, test_accs, 'g', label='Test accuracy')
plt.plot(epochs, test_losses, 'm', label='Test loss')
plt.title('Accuracy and loss')
plt.legend(loc=0)
plt.grid(True)
plt.figure(figsize=(10,5))

plt.show()
