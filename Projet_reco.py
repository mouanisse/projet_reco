import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split



########################################### 3A DATASET ################################################


images = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/emotion_images_3A.npy')
labels = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/emotion_labels_3A.npy')


# Break data into training , validation and test sets
train_images, test_val_images, test_images, val_images, train_labels, test_val_labels, test_labels, val_labels = [], [], [], [], [], [], [], []
train_images, test_val_images, train_labels, test_val_labels = train_test_split(images, labels, test_size=0.20, random_state=42)
test_images, val_images, test_labels, val_labels = train_test_split(test_val_images, test_val_labels, test_size=0.50, random_state=42)

# Flatten data: the values are between -9 and 4 , so we should replace them by values between 0 and 1
def flatten(images):
    images = np.array(images)
    temp=max(abs(images.min()),abs(images.max()))
    images = (((images/temp)+1)/2)-0.5
    return images



train_data = flatten(train_images)
val_data = flatten(val_images)
test_data = flatten(test_images)

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)


train_images_res = train_data.reshape((-1, 129, 129, 1))
val_images_res = val_data.reshape((-1, 129, 129, 1))
test_images_res = test_data.reshape((-1, 129, 129, 1))



class Oyez_Oyez:

    def __init__(self, input_shape):
        self.input_shape = input_shape



    # 2D Convolutional Model
    # ('None' represents the batch_size).

    def create_word_model(self):
        "This function creates a CNN model for Speech Word Recognition"

        # 4CONV_LAYERS + 2FULLY_CONNECTED + 1SOFTMAX

        model = keras.Sequential()
        model.add(keras.layers.Permute((2, 1, 3)))

        # On utilise la fonction Permute pr changer le format de l'entrée
        # On aura alors la shape (128,96,3) , et ca donne des résultats bien meilleurs.

        model.add(keras.layers.Conv2D(20, (5, 1), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 1))        # divise par 2 les dimensions de l'image


        model.add(keras.layers.Conv2D(40, (3, 3), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))  # divise par 4 les dimensions de l'image


        model.add(keras.layers.Conv2D(80, (3, 3), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))
        
        
        model.add(keras.layers.Conv2D(160, (3, 3), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))


        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        
        model.add(keras.layers.Dense(5, activation='softmax'))

        return model


    def create_emotion_model(self):
        "This function creates a CNN model for Speech Emotion Recognition (SER) based on Dr.Somayeh Shahsavarani paper"

        # 2CONV_LAYERS + 2FULLY_CONNECTED + 1SOFTMAX


        model = keras.Sequential()

        model.add(keras.layers.Conv2D(8, (5, 5), strides=(1, 1), input_shape=(129, 129, 1), padding='same', activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))


        model.add(keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(17424, activation='relu'))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Dense(7, activation='softmax'))

        return model


    def train_word_model(self):
        "This function trains our model for Speech Word Recognition"

        model = self.create_word_model()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(self.training_word_data, self.training_word_label, batch_size=100, epochs=7, 
                  validation_data=(self.validation_word_data, self.validation_word_label) )

        score = model.evaluate(self.testing_word_data, self.testing_word_label, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save("word_model.h5")
        

    def train_emotion_model(self):
        "This function trains our model for Speech Emotion Recognition"

        model = self.create_emotion_model()
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])

        model.fit(train_images_res, train_labels, epochs=10, validation_data=(val_images_res, val_labels), verbose=1)
            
        score = model.evaluate(test_images_res, test_labels, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        model.save("/content/drive/My Drive/emotion_model_3A.h5")
        

    def load_trained_word_model(self):

        model = keras.models.load_model(self.word_model_path)
        return model


    def load_trained_emotion_model(self):

        model = keras.models.load_model(self.emotion_model_path)
        return model





obj = Oyez_Oyez((129,129,1))

obj.train_emotion_model()
