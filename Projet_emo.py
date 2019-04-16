import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


#********************************************* Loading the dataset **************************************************

train_data = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/train_emotions.npy')
test_data = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/test_emotions.npy')
train_labels = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/train_labels.npy')
test_labels = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/test_labels.npy')



#********************************************* Building our CNN model *********************************************

model = keras.Sequential()

model.add(keras.layers.Conv1D(256, 5, padding='same', input_shape=(259, 1)))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Conv1D(128, 5,padding='same'))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.MaxPooling1D(pool_size=8))


model.add(keras.layers.Conv1D(128, 5,padding='same'))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Conv1D(128, 5,padding='same'))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(8, activation='softmax'))

#opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

model.summary()

#**************************************** Compile and train our CNN model *******************************************


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_labels, batch_size=100, epochs=30)

score = model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




