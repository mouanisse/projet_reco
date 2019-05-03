import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


#********************************************* Loading the dataset **************************************************

train_data = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/train_emotions_3.npy')
test_val_data = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/test_emotions_3.npy')
train_labels = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/train_labels_3.npy')
test_val_labels = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/test_labels_3.npy')
test_data, val_data, test_labels, val_labels = train_test_split(test_val_data, test_val_labels, test_size=0.5, random_state=42)



#********************************************* Building our CNN model *********************************************

model = keras.Sequential()

model.add(keras.layers.Conv1D(64, 5, padding='same', input_shape=(259,1), activation='relu'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv1D(128, 5, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling1D(pool_size=8))

model.add(keras.layers.Conv1D(256, 5, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv1D(256, 5, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv1D(256, 5, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv1D(512, 5, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling1D(pool_size=8))


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(3, activation='softmax'))


model.summary()

#**************************************** Compile and train our CNN model *******************************************

checkpoint = keras.callbacks.ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_labels, batch_size=100, epochs=70, callbacks=[checkpoint], validation_data=(val_data, val_labels))

score = model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])





