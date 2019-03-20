
import tensorflow as tf
from tensorflow import keras
import numpy as np

images = np.load('/content/drive/My Drive/Colab Notebooks/no_dog_dataset/train.npy')
labels = np.load('/content/drive/My Drive/Colab Notebooks/no_dog_dataset/labels.npy')

x_train = images[0:5000][:][:][:]
y_train = labels[0:5000]

x_test = images[5000:len(images)][:][:][:]
y_test = labels[5000:len(images)]

# Variables

img_rows, img_cols = 28, 28
input_shape = (96, 128, 3)


# 2D Convolutional Model
# ('None' represents the batch_size).

model = keras.Sequential()
model.add(keras.layers.Permute((2,1,3)))

# On utilise la fonction Permute pr changer le format de l'entrée
# On aura alors la shape (128,96,3) , et ca donne des résultats bien meilleurs.


model.add(keras.layers.Conv2D(20, (5, 1), activation='relu', input_shape=input_shape)) #5
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
model.add(keras.layers.Dense(64, activation='relu'))

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))


# Phase d'entrainement et de test

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test) ) #batch_size prend les echantillons 100 par 100

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
