# -*- coding: utf-8 -*-

#Disable GPU for memory issues (if needed)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0" #0=DEBUG (default), 1=INFO, 2=WARINIG, 3=ERROR

import keras
keras.__version__

from keras.datasets import mnist
from keras import Model
import tensorflow as tf

#Incompatibile
#from keras import layers

#Sostituzione
from tensorflow.keras import layers

# load data from MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Convert images to rgb
# Warning: huge memory needed!
train_images=tf.image.grayscale_to_rgb(tf.expand_dims(train_images, axis=3))
test_images=tf.image.grayscale_to_rgb(tf.expand_dims(test_images, axis=3))

from tensorflow import image as i
train_images = i.resize_with_pad(train_images, 75, 75)
test_images = i.resize_with_pad(test_images, 75, 75)


print('train images shape', train_images.shape)
print('len train images shape', len(train_images))
print('train labels', train_labels)
print('train labels shape', train_labels.shape)

print('test images shape', test_images.shape)
print('len test images shape', len(test_images))
print('test labels', test_labels)
print('test labels shape', test_labels.shape)


# build an artificial neural network (ANN)
from tensorflow.keras.applications.inception_v3 import InceptionV3

#Download InceptionV3 weights
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (75, 75, 3),
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False
  
#pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer: ', last_layer)
print('last layer output: ', last_layer.output)
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
print('last output shape: ', last_output.shape)

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(name='primo_strato_Dense', units=1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
#x = layers.Dense  (1, activation='sigmoid')(x)
x = layers.Dense(name='secondo_strato_Dense', units=512, activation='relu')(x)
x = layers.Dense(name='terzo_strato_Dense', units=10, activation='softmax')(x)

network = Model( pre_trained_model.input, x) 

#print(network.summary())
#print(network.layers)
print(network)

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
history = network.fit(train_images, train_labels, epochs=15, batch_size=2048)

# test the ANN using the test data set
test_loss, test_acc = network.evaluate(test_images, test_labels)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))
test_accs = [test_acc] * len(acc)
test_losses = [test_loss] * len(loss)

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, test_accs, 'b', label='Test accuracy')
#plt.plot(epochs, loss, 'b', label='Training loss')
#plt.plot(epochs, val_acc, 'g', label='Test accuracy')
#plt.plot(epochs, val_loss, 'm', label='Test loss')
plt.title('Accuracy')
plt.legend(loc=0)
plt.grid(True)
plt.figure(figsize=(10,5))

plt.show()

#plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, test_losses, 'b', label='Test loss')
#plt.plot(epochs, val_acc, 'g', label='Test accuracy')
#plt.plot(epochs, val_loss, 'm', label='Test loss')
plt.title('Loss')
plt.legend(loc=0)
plt.grid(True)
plt.figure(figsize=(10,5))

plt.show()

#plt.plot(epochs, acc, 'r', label='Training accuracy')
#plt.plot(epochs, loss, 'b', label='Training loss')
#plt.title('Accuracy and loss')
#plt.legend(loc=0)
#plt.grid(True)
#plt.figure(figsize=(10,5))
#plt.show()
