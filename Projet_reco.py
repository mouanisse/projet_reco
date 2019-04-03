
import tensorflow as tf
from tensorflow import keras
import kapre
import numpy as np

training_data = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_train_set.npy')
testing_data = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_test_set.npy')
validation_data = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_valid_set.npy')

training_label = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_train_label.npy')
testing_label = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_test_label.npy')
validation_label = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_validation_label.npy')


# Variables

input_shape = (96, 128, 3)


# 2D Convolutional Model
# ('None' represents the batch_size).

model = keras.Sequential()
model.add(keras.layers.Permute((2,1,3)))

# On utilise la fonction Permute pr changer le format de l'entrée
# On aura alors la shape (128,96,3) , et ca donne des résultats bien meilleurs.


model.add(keras.layers.Conv2D(20, (5, 5), activation='relu')) 
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2, 1))


model.add(keras.layers.Conv2D(40, (3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2, 2))    # divise par 2 les dimensions de l'image



model.add(keras.layers.Conv2D(80, (3, 3), activation='relu')) 
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2, 2))



model.add(keras.layers.Conv2D(160, (3, 3), activation='relu')) 
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2, 2))



model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(5, activation='softmax'))


# Phase d'entrainement et de test


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(training_data, training_label, batch_size=100, epochs=10, validation_data=(validation_data, validation_label) ) 
#batch_size prend les echantillons 100 par 100

score = model.evaluate(testing_data, testing_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
