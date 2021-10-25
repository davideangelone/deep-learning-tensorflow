# -*- coding: utf-8 -*-

import numpy as np

#Paths
smsTxtPath = './SMSSpamCollection.txt'
plotSavePath = './plot/save/dir'

#Dataset settings
splitPerc=.2

#Vocab settings
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"


#Model settings
#denseLayers = [512, 128]
#denseLayers = [8, 4]
#denseLayers = [24]
#denseLayers = [6]
#denseLayers = [4]
#denseLayers = [2]
denseLayers = []
numEpochs=25


from datetime import datetime
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def savePlot(plt, suffix):
    plt.suptitle(now + ' - Dense layers ' + str(denseLayers))
    plt.legend(loc=0)
    plt.grid(True)
    plt.gcf().savefig(fname=plotSavePath + '/' + now + '_' + suffix + '.png', format='png')   
    plt.close()

def annotateMax(plt, epochs, val, color):
    ymax = max(val)
    xpos = val.index(ymax)
    xmax = epochs[xpos]
    plt.annotate(text=round(ymax,4), xy=(xmax, ymax), xytext=(xmax, ymax), bbox=(dict(facecolor=color, alpha=0.5)))

def annotateMin(plt, epochs, val, color):
    ymin = min(val)
    xpos = val.index(ymin)
    xmin = epochs[xpos]
    plt.annotate(text=round(ymin,4), xy=(xmin, ymin), xytext=(xmin, ymin), bbox=(dict(facecolor=color, alpha=0.5)))


file = open(smsTxtPath, 'r')
lines = [line.rstrip() for line in file]

messaggi = []
#Tipo messaggio: 0=OK, 1=SPAM
tipo = []
# Strips the newline character
for line in lines:
    tipo.append(0 if line.startswith('ham') else 1)
    messaggi.append(line[4:] if line.startswith('ham') else line[5:])

split_index=int(len(messaggi)*splitPerc)

messaggi_training=messaggi[:split_index]
tipo_training=tipo[:split_index]

messaggi_test=messaggi[split_index:]
tipo_test=tipo[split_index:]



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(messaggi_training)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(messaggi_training)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(messaggi_test)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

#print(testing_sequences)
#print(testing_padded)

#Conversion needed by tf
training_labels_final = np.array(tipo_training)
testing_labels_final = np.array(tipo_test)


#Print messages
#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#def decode_review(text):
#    return ' '.join([reverse_word_index.get(i, '?') for i in text])

#print(decode_review(padded[2]))
#print(sequences[2])


import tensorflow as tf

network = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
])
for neurons in denseLayers:
    network.add(tf.keras.layers.Dense(neurons, activation='relu', name='strato_dense_' + str(neurons)))
network.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
network.compile(loss='binary_crossentropy',optimizer='adam',
                metrics=['accuracy', 
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall'),
                            tf.keras.metrics.FalseNegatives(name='false_negatives'),
                            tf.keras.metrics.FalsePositives(name='false_positives')
                        ])
network.summary()

history = network.fit(x=padded,
                      y=training_labels_final,
                      validation_data=(testing_padded, testing_labels_final),
                      epochs=numEpochs,
                      shuffle=True,
                      verbose=1)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
precision = history.history['precision']
recall = history.history['recall']
false_negatives = history.history['false_negatives']
false_positives = history.history['false_positives']
val_precision = history.history['val_precision']
val_recall = history.history['val_recall']
val_false_negatives = history.history['val_false_negatives']
val_false_positives = history.history['val_false_positives']

epochs = range(len(acc))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(epochs, acc, 'r', label='Training accuracy (max=' + str(round(max(acc), 4)) + ')')
plt.plot(epochs, val_acc, 'g', label='Test accuracy (max=' + str(round(max(val_acc), 4)) + ')')
plt.plot(epochs, loss, 'b', label='Training loss (min=' + str(round(min(loss), 4)) + ')')
plt.plot(epochs, val_loss, 'm', label='Test loss (min=' + str(round(min(val_loss), 4)) + ')')
annotateMax(plt, epochs, acc, 'r')
annotateMax(plt, epochs, val_acc, 'g')
annotateMin(plt, epochs, loss, 'b')
annotateMin(plt, epochs, val_loss, 'm')
plt.title('Accuracy and loss')
savePlot(plt, 'accuracy_loss')



plt.figure(figsize=(10, 5))
plt.plot(epochs, precision, 'r', label='Training precision (max=' + str(round(max(precision), 4)) + ')')
plt.plot(epochs, recall, 'g', label='Training recall (max=' + str(round(max(recall), 4)) + ')')
plt.plot(epochs, val_precision, 'c', label='Test precision (max=' + str(round(max(val_precision), 4)) + ')')
plt.plot(epochs, val_recall, 'y', label='Test recall (max=' + str(round(max(val_recall), 4)) + ')')
annotateMax(plt, epochs, precision, 'r')
annotateMax(plt, epochs, recall, 'g')
annotateMax(plt, epochs, val_precision, 'c')
annotateMax(plt, epochs, val_recall, 'y')
plt.title('Precision and recall')
savePlot(plt, 'precision_recall')



plt.figure(figsize=(10, 5))
plt.plot(epochs, false_negatives, 'b', label='Training false_negatives (min=' + str(round(min(false_negatives), 4)) + ')')
plt.plot(epochs, false_positives, 'm', label='Training false_positives (min=' + str(round(min(false_positives), 4)) + ')')
plt.plot(epochs, val_false_negatives, '--', label='Test false_negatives (min=' + str(round(min(val_false_negatives), 4)) + ')')
plt.plot(epochs, val_false_positives, ':', label='Training false_positives (min=' + str(round(min(val_false_positives), 4)) + ')')
annotateMin(plt, epochs, false_negatives, 'b')
annotateMin(plt, epochs, false_positives, 'm')
annotateMin(plt, epochs, val_false_negatives, 'c')
annotateMin(plt, epochs, val_false_positives, 'r')
plt.title('False positive and negative')
savePlot(plt, 'falses')

#plt.show()
print('Done')
