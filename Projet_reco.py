
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

num_classes = 2
a = 5
b = 6

img_rows, img_cols = 28, 28
input_shape = (96, 128, 3)



# 2D Convolutional Model
# At the output of the spectrogram we will have a vector of shape ('None',mel_dim,tim_steps,3)
# ('None' represents the batch_size).

model = keras.Sequential()
print("Shape before: ",model.output_shape)
model.add(keras.layers.Permute((2,1,3)))
print("Shape after: ",model.output_shape)

# On utilise la fonction Permute pr changer de ('None',mel_dim,tim_steps,3) à (batch_size ou 'None', sequence, vec_dim)

model.add(keras.layers.Conv2D(20, (5, 1), activation='relu', input_shape=input_shape)) #5
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2, 1))
#model.add(keras.layers.Dropout(0.02))



model.add(keras.layers.Conv2D(40, (3, 3), activation='relu')) #25
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2, 2))    # divise par 2 les dimensions de l'image
#model.add(keras.layers.Dropout(0.01))



model.add(keras.layers.Conv2D(80, (3, 3), activation='relu')) #150
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2, 2))
#model.add(keras.layers.Dropout(0.02))


model.add(keras.layers.Conv2D(160, (3, 3), activation='relu')) 
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2, 2))



model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))


# RNN Model


#model = keras.Sequential()
#model.add(keras.layers.Conv2D(10, (5, 1), activation='relu', input_shape= input_shape))
#model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.Conv2D(1, (5, 1), activation='relu'))
#model.add(keras.layers.BatchNormalization())

# A la sortie du réseau de convolution , on a un vecteur de dimension 4 , on utilise la methode squeeze pr eliminer une dim
# et avoir un vecteur a 3 dimensions

#model.add(keras.layers.Bidirectional(keras.layers.CuDNNLSTM(64, return_sequences = True)))
#model.add(keras.layers.Bidirectional(keras.layers.CuDNNLSTM(64)))

#model.add(keras.layers.Dense(64, activation='relu'))
#model.add(keras.layers.Dense(32, activation='relu'))
#model.add(keras.layers.Dense(2, activation='softmax'))



#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test) ) #batch_size prend les echantillons 100 par 100

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
