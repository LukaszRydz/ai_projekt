import numpy as np
import os
from tqdm import tqdm
import cv2
import time

from keras.utils import img_to_array, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend

from termcolor import colored

class Images:
    logs = None                                     # Obiekt klasy Logs

    IMAGE_PATHS = []                                # Tablica z scieżkami do obrazów
    IMAGE_DATA = []                                 # Lista z obrazami które zostaną przetworzone
    IMAGE_LABELS = []                               # Lista z etykietami dla przetworzonych obrazów
    AUGUMENTED_IMAGE_DATA = []                      # Lista z przetworzonymi obrazami
    
    train_images = []                               # Obrazy do uczenia
    train_labels = []                               # Etykiety obrazów do uczenia
    test_images = []                                # Obrazy do testowania
    test_labels = []                                # Etykiety obrazów do testowania

    chanDims = 0                                    # Oś kanałów
    imageProperties = ()                            # Wymiary obrazu

    trainParameters = None                          # Parametry uczenia

    def __loadImages__(self):
        print(colored("Rozpoczęcie ładowania zdjęć...", "yellow"))
        for root, dirs, files in os.walk('./database'):
            for file in files:
                file_path = os.path.join(root, file)
                if not os.path.isdir(file_path):
                    self.IMAGE_PATHS.append(file_path)

    def __convertImages__(self):
        
        for img in tqdm(self.IMAGE_PATHS, desc=colored("Konwertowanie zdjęć", "magenta")):
            # Wczytanie obrazu
            image = cv2.imread(img)
            # Zmiana rozmiaru obrazu na wymiary zdefiniowane
            image = cv2.resize(image, (self.trainParameters.width, self.trainParameters.height))
            # Konwertuje obraz na tablicę i dodaje go do listy obrazów
            image = img_to_array(image)
            self.IMAGE_DATA.append(image)

            # Etykietowanie obrazów 1 == kobieta, 0 == mężczyzna
            label = img.split(os.path.sep)[-2]
            if label == "women":
                label = 1
            else:
                label = 0
            self.IMAGE_LABELS.append([label])

        # Normalizacja która ma na celu skalowanie wartości pikseli do zakresu od 0 do 1, co ułatwia uczenie modelu.
        self.IMAGE_DATA = np.array(self.IMAGE_DATA, dtype="float") / 255.0
        # Konwersja etykiet do tablicy numpy
        self.IMAGE_LABELS = np.array(self.IMAGE_LABELS)

        # Podział danych na zbiór treningowy i testowy 20% dla testów, 80% dla treningu
        (self.train_images, self.test_images, self.train_labels, self.test_labels) = train_test_split(self.IMAGE_DATA, self.IMAGE_LABELS, test_size=0.2, random_state=42)
        
        # Konwersja etykiet do postaci kategorycznej
        self.train_labels = to_categorical(self.train_labels, num_classes=2)
        self.test_labels = to_categorical(self.test_labels, num_classes=2)

        # Ustawienie wymiarów wejściowych oraz osi kanałów
        if backend.image_data_format() == "channels_first":
            self.imageProperties = (self.trainParameters.channels, 
                                    self.trainParameters.height, 
                                    self.trainParameters.width)
            self.chanDim = 1
        else:   
            self.imageProperties = (self.trainParameters.height, 
                                    self.trainParameters.width, 
                                    self.trainParameters.channels)
            self.chanDim = -1

    def __imageAugmentation__(self):
        print(colored("Rozpoczęcie augmentacji zdjęć...", "yellow"))
        self.IMAGE_AUGMENTER = ImageDataGenerator(
            rotation_range=20, 
            width_shift_range=0.08, 
            height_shift_range=0.08, 
            shear_range=0.15, 
            zoom_range=0.15, 
            horizontal_flip=True, 
            fill_mode="nearest")
        print(colored("Zakończono augmentację zdjęć.", "green"))
    
    def start(self, trainParameters, logs):
        self.logs = logs
        self.trainParameters = trainParameters
        TOTAL_IMG_PREPARE_DURATION = time.time()
        
        TIMER = time.time()
        self.__loadImages__()
        self.logs.add_value("Load images duration", str(time.time() - TIMER) + "s")
        
        TIMER = time.time()
        self.__convertImages__()
        self.logs.add_value("Convert images duration", str(time.time() - TIMER) + "s")
        
        self.__imageAugmentation__()

        self.logs.add_value("Number of images: ", len(self.IMAGE_PATHS))
        self.logs.add_value("Total image prepare duration", str(time.time() - TOTAL_IMG_PREPARE_DURATION) + "s")