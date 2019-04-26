import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


#********************************************* Loading the dataset **************************************************

train_data = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/train_emotions2.npy')
test_val_data = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/test_emotions2.npy')
train_labels = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/train_labels3.npy')
test_val_labels = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/test_labels3.npy')
test_data, val_data, test_labels, val_labels = train_test_split(test_val_data, test_val_labels, test_size=0.5, random_state=42)



#********************************************* Building our CNN model *********************************************

model = keras.Sequential()

model.add(keras.layers.Conv1D(32, 5, padding='same', input_shape=(517,1), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling1D(pool_size=4))

model.add(keras.layers.Conv1D(64, 5, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling1D(pool_size=4))

model.add(keras.layers.Conv1D(128, 5, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling1D(pool_size=4))

model.add(keras.layers.Conv1D(256, 5, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling1D(pool_size=4))

model.add(keras.layers.Conv1D(512, 5, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling1D(pool_size=4))


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(7, activation='softmax'))


model.summary()

#**************************************** Compile and train our CNN model *******************************************


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_labels, batch_size=500, epochs=100, validation_data=(val_data, val_labels))

score = model.evaluate(test_data, test_labels, verbose=0)
model.save("emotion_model_mfcc.h5")
print('Test loss:', score[0])
print('Test accuracy:', score[1])





