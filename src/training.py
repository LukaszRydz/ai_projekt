import time
from images import Images
import tensorflow as tf
import os
from keras import backend as K
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LeakyReLU
from keras.callbacks import CSVLogger
from logs import Logs

from termcolor import colored

class Training:
    
    images = Images()                                                                               # Obiekt klasy Images  (obrazy)
    logs = Logs()                                                                                   # Obiekt klasy Logs    (logi)
    trainParameters = None                                                                          # Obiekt klasy TrainParameters (parametry treningu)
    CLASSIFICATION_MODEL = None                                                                     # Model klasyfikacji

    def __prepareImages__(self):
        self.images.start(self.trainParameters, self.logs)

    def __buildModel__(self):
        # Model sekwencyjny
        model = Sequential([
            # Warstwa konwolucyjna 2D z 32 filtrami o rozmiarze 3x3, z zachowaniem wymiarów, wejściowy kształt obrazu
            Conv2D(32, (3, 3), padding="same", input_shape=self.images.imageProperties),  
            Activation("relu"),                                                                     # Funkcja aktywacji ReLU
            BatchNormalization(axis=self.images.chanDim),                                           # Normalizacja wsadowa na kanale
            MaxPooling2D(pool_size=(3, 3)),                                                         # Warstwa poolingowa typu maksymalnego o rozmiarze 3x3
            Dropout(0.25),                                                                          # Warstwa dropoutu o współczynniku 0.25

            Conv2D(64, (3, 3), padding="same"),
            Activation("relu"),
            BatchNormalization(axis=self.images.chanDim),                                           # Normalizacja wsadowa na kanale

            Conv2D(64, (3, 3), padding="same"),
            Activation("relu"),
            BatchNormalization(axis=self.images.chanDim),
            MaxPooling2D(pool_size=(2, 2)),                                                         # Warstwa poolingowa typu maksymalnego o rozmiarze 2x2
            Dropout(0.25),                                                                          # Warstwa dropoutu o współczynniku 0.25

            Conv2D(128, (3, 3), padding="same"),
            Activation("relu"),
            BatchNormalization(axis=self.images.chanDim), 

            Conv2D(128, (3, 3), padding="same"),
            Activation("relu"),
            BatchNormalization(axis=self.images.chanDim),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Flatten(),                                                                              # Warstwa spłaszczająca
            Dense(1024),                                                                            # W pełni połączona warstwa z 1024 neuronami
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.5),

            Dense(2),                                                                               # W pełni połączona warstwa z 2 neuronami (klasyfikacja binarna (meżczyzna/kobieta))
            Activation("sigmoid")                                                                   # Funkcja aktywacji sigmoidalna
        ])

        self.CLASSIFICATION_MODEL = model
    
    def __trainModel__(self):
        print(colored("\n\n\nRozpoczęcie treningu...", 'yellow'))

        # Inicjalizacja optymalizatora
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=self.trainParameters.learning_rate, 
                                              decay=self.trainParameters.learning_rate/self.trainParameters.epochs)
        
        # Kompilacja modelu 
        self.CLASSIFICATION_MODEL.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc"])

        # Inicjalizacja generatora danych treningowych
        train_data_generator = self.images.IMAGE_AUGMENTER.flow(self.images.train_images, self.images.train_labels, 
                                                                batch_size=self.trainParameters.batch_size)

        # Inicjalizacja generatora zapisującego wyniki
        csv_logger = CSVLogger(os.path.join(self.logs.get_folder_path(), "epochs_history.csv"))

        # Train the model
        TRAINING_HISTORY = self.CLASSIFICATION_MODEL.fit(
            train_data_generator,                                                                   # Generator danych treningowych
            validation_data=(self.images.test_images, self.images.test_labels),                     # Dane do walidacji modelu
            steps_per_epoch=len(self.images.train_images) // self.trainParameters.batch_size,       # Liczba kroków (mini-batches) na epokę
            epochs=self.trainParameters.epochs,                                                     # Liczba epok
            verbose=1,                                                                              # Poziom wypisywanych informacji
            callbacks=[csv_logger]                                                                  # Lista callbacków, w tym przypadku CSVLogger
        )
        
        self.logs.savePlot(TRAINING_HISTORY, self.trainParameters)
        
        # Zapisanie modelu
        model_path = './model/gender_detection.model'
        self.CLASSIFICATION_MODEL.save(model_path)

    def __init__(self, conf):
        TOTAL_TRAINING_DURATION = time.time()

        # Pobranie parametrów treningu
        self.trainParameters = conf
        
        # Zapisanie parametrów treningu do pliku
        self.logs.add_value("Status", "Unfinished")
        self.logs.add_value("Training parameters: ", self.trainParameters.get_parameters())
        
        
        self.__prepareImages__()                                                                    # Przygotowanie obrazów
        self.__buildModel__()                                                                       # Zbudowanie modelu
        
        TIMER = time.time()
        self.__trainModel__()                                                                       # Trening modelu
        
        self.logs.add_value("Training duration: ", str(time.time() - TIMER) + "s")                  # Zapisanie czasu treningu
        self.logs.add_value("Total duration: ", str(time.time() - TOTAL_TRAINING_DURATION) + "s")   # Zapisanie całkowitego czasu treningu

        print(colored("Training finished", 'green'))
        input(colored("Press Enter to continue...", 'magenta'))

        self.logs.add_value("Status", "Finished")
        K.clear_session()                                                                           # Wyczyszczenie sesji