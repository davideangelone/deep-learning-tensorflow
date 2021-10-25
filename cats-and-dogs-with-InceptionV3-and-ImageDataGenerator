# -*- coding: utf-8 -*-

#Disable GPU for memory issues (if needed)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0" #0=DEBUG (default), 1=INFO, 2=WARINIG, 3=ERROR

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

train_datagen = train_datagen.flow_from_directory (
                                     directory=petImagesPath,
                                     class_mode='binary',
                                     color_mode='rgb',
                                     subset="training",
                                     seed=1337,
                                     target_size=image_size,
                                     batch_size=batch_size,
                                     )     

test_datagen =  test_datagen.flow_from_directory(
                                     directory=petImagesPath,
                                     color_mode='rgb',
                                     class_mode='binary',
                                     subset="validation",
                                     seed=1337,
                                     target_size=image_size,
                                     batch_size=batch_size,
                                     )

from tensorflow.keras.applications.inception_v3 import InceptionV3

#Download InceptionV3 weights
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(180, 180, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer: ', last_layer)
print('last layer output: ', last_layer.output)
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(name='primo_strato_Dense', units=1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
#x = layers.Dense(name='secondo_strato_Dense', units=512, activation='relu')(x)
x = layers.Dense(name='terzo_strato_Dense', units=1, activation='sigmoid')(x)

network = models.Model(inputs=pre_trained_model.input, outputs=x)

print(network.summary())

from tensorflow import keras as k

#k.utils.plot_model(network, show_shapes=True)

# make the ANN operational 'compiling' it
network.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])


# train ANN
history = network.fit(x=train_datagen,
                      validation_data=test_datagen,
                      #steps_per_epoch=100,
                      epochs=25,
                      #validation_steps=50,
                      verbose=1)

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
plt.figure(figsize=(10, 5))

plt.show()
