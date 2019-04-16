import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


#********************************************* Loading the dataset **************************************************

dataset_features = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/emotion_features.npy')
dataset_labels = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/emotion_labels.npy')


#********************************** Break data into training , validation and test sets *****************************************************


train_data, test_val_data, train_labels, test_val_labels = train_test_split(dataset_features, dataset_labels, test_size=0.20, random_state=42)
test_data, val_data, test_labels, val_labels = train_test_split(test_val_data, test_val_labels, test_size=0.50, random_state = 42)



#********************************************* Building our CNN model *********************************************

model = keras.Sequential()

model.add(keras.layers.Conv1D(256, 5, padding='same', input_shape=(216, 1)))
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


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_labels, batch_size=100, epochs=10, validation_data=(val_data, val_labels))

score = model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])





