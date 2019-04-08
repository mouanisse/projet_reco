import tensorflow as tf
from tensorflow import keras
import numpy as np



training_word_data = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_train_set.npy')
testing_word_data = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_test_set.npy')
validation_word_data = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_valid_set.npy')

training_word_label = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_train_label.npy')
testing_word_label = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_test_label.npy')
validation_word_label = np.load('/content/drive/My Drive/Colab Notebooks/5_words_dataset/reduce_validation_label.npy')

training_emotion_data = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/training_emotion_images.npy')
testing_emotion_data = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/testing_emotion_images.npy')
validation_emotion_data = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/validation_emotion_images.npy')

training_emotion_label = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/training_emotion_label.npy')
testing_emotion_label = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/testing_emotion_label.npy')
validation_emotion_label = np.load('/content/drive/My Drive/Colab Notebooks/emotion_dataset/validation_emotion_label.npy')



class Oyez_Oyez:

    def __init__(self, training_word_data, testing_word_data, validation_word_data, training_word_label,
                 testing_word_label, validation_word_label, training_emotion_data, testing_emotion_data,
                 validation_emotion_data, training_emotion_label, testing_emotion_label, validation_emotion_label):

        self.training_word_data = training_word_data
        self.testing_word_data = testing_word_data
        self.validation_word_data = validation_word_data
        self.training_word_label = training_word_label
        self.testing_word_label = testing_word_label
        self.validation_word_label = validation_word_label

        self.training_emotion_data = training_emotion_data
        self.testing_emotion_data = testing_emotion_data
        self.validation_emotion_data = validation_emotion_data
        self.training_emotion_label = training_emotion_label
        self.testing_emotion_label = testing_emotion_label
        self.validation_emotion_label = validation_emotion_label


    # 2D Convolutional Model
    # ('None' represents the batch_size).

    def create_word_model(self):
        "This function creates a CNN model for Speech Word Recognition"

        # 4CONV_LAYERS + 2FULLY_CONNECTED + 1SOFTMAX

        model = keras.Sequential()
        model.add(keras.layers.Permute((2, 1, 3)))

        # On utilise la fonction Permute pr changer le format de l'entrée
        # On aura alors la shape (128,96,3) , et ca donne des résultats bien meilleurs.

        model.add(keras.layers.Conv2D(10, (5, 1), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 1))        # divise par 2 les dimensions de l'image


        model.add(keras.layers.Conv2D(20, (3, 3), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))  # divise par 4 les dimensions de l'image


        model.add(keras.layers.Conv2D(40, (3, 3), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))


        model.add(keras.layers.Conv2D(80, (3, 3), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))
        
        
        #model.add(keras.layers.Conv2D(160, (3, 3), activation='relu'))
        #model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.MaxPooling2D(2, 2))


        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        
        model.add(keras.layers.Dense(8, activation='softmax'))

        return model


    def create_emotion_model(self):
        "This function creates a CNN model for Speech Emotion Recognition (SER) based on Dr.Somayeh Shahsavarani paper"

        # 2CONV_LAYERS + 2FULLY_CONNECTED + 1SOFTMAX


        model = keras.Sequential()

        model.add(keras.layers.Conv2D(8, (5, 5), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))


        model.add(keras.layers.Conv2D(16, (5, 5), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))


        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(17424, activation='relu'))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Dense(8, activation='softmax'))

        return model


    def train_word_model(self):
        "This function trains our model for Speech Word Recognition"

        model = self.create_word_model()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # ModelCheckPoint will save the model with the best validation accuracy
        #checkpointer = keras.callbacks.ModelCheckpoint(filepath="/content/projet_reco/word_model.hdf5",
                                                       #monitor='val_acc', verbose=0, save_best_only=True,
                                                       #save_weights_only=False, mode='max', period=1)
        # Save the path to the CNN model
        #self.word_model_path = "/content/projet_reco/word_model.hdf5"

        model.fit(self.training_word_data, self.training_word_label, batch_size=100, epochs=7,
                  validation_data=(self.validation_word_data, self.validation_word_label))#, callbacks=[checkpointer])

        score = model.evaluate(testing_word_data, testing_word_label, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])



    def train_emotion_model(self):
        "This function trains our model for Speech Emotion Recognition"

        model = self.create_word_model()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # ModelCheckPoint will save the model with the best validation accuracy
        #checkpointer = keras.callbacks.ModelCheckpoint(filepath="/content/projet_reco/emotion_model.hdf5",
                                                       #monitor='val_acc', verbose=0, save_best_only=True,
                                                       #save_weights_only=False, mode='max', period=1)
        # Save the path to the CNN model
        #self.emotion_model_path = "/content/projet_reco/emotion_model.hdf5"

        model.fit(self.training_emotion_data, self.training_emotion_label, batch_size=100, epochs=14,
                  validation_data=(self.validation_emotion_data, self.validation_emotion_label))#, callbacks=[checkpointer])

        score = model.evaluate(self.testing_emotion_data, self.testing_emotion_label, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])



    def load_trained_word_model(self):

        model = keras.models.load_model(self.word_model_path)
        return model


    def load_trained_emotion_model(self):

        model = keras.models.load_model(self.emotion_model_path)
        return model





obj = Oyez_Oyez(training_word_data, testing_word_data, validation_word_data, training_word_label,
                 testing_word_label, validation_word_label, training_emotion_data, testing_emotion_data,
                 validation_emotion_data, training_emotion_label, testing_emotion_label, validation_emotion_label )

obj.train_emotion_model()
