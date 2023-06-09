from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import datetime
import os
import time
import threading
# Wczytanie wytrenowanego modelu
model = load_model('./model/gender_detection.model')

import sys
sys.path.append('src')
from train_config import TrainConfig
config = TrainConfig("train_config.json")

# Inicjalizacja znacznika czasu
last_save_time = 0

def cameraApp():
    # Inicjalizacja zmiennej przechowującej ostatnią zapisaną wartość znacznika czasu
    global last_save_time

    # Inicjalizacja kamery
    camera = cv2.VideoCapture(0)
    genders = ['man','woman']

    while camera.isOpened():
        # Odczytanie klatki z kamery
        status, frame = camera.read()

        # Wykrycie twarzy na klatce
        face, confidence = cv.detect_face(frame)

        for idx, f in enumerate(face):
            # Prostokąta obejmująy twarz
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # Narysowanie prostokąta wokół twarzy
            cv2.rectangle(frame, (startX,startY), (endX,endY), (255,0,0), 2)

            # Przycięcie obszaru z wykrytą twarzą
            croped_image = np.copy(frame[startY:endY,startX:endX])

            # Sprawdzenie minimalnego rozmiaru przyciętego obrazu
            if (croped_image.shape[0]) < 10 or (croped_image.shape[1]) < 10:
                continue

            # Zmiana rozmiaru przyciętego obrazu na 96x96 pikseli
            croped_image = cv2.resize(croped_image, (config.get_width(), config.get_height()))
            # Przygotowanie kopii przyciętego obrazu do zapisu
            image_to_save = croped_image

            # Przygotowanie przyciętego obrazu dla modelu
            croped_image = croped_image.astype("float") / 255.0
            croped_image = img_to_array(croped_image)
            croped_image = np.expand_dims(croped_image, axis=0)

            # Predykcja płci na podstawie przyciętego obrazu
            conf = model.predict(croped_image)[0]

            # Pobranie indeksu o największej wartości predykcji
            idx = np.argmax(conf)
            # Pobranie etykiety odpowiadającej indeksowi
            label = genders[idx]

            # Przygotowanie napisów do wyświetlenia na klatce
            label_gender = "Gender: {}".format(label, )
            label_prob = "Prob: {:.1f}%".format(conf[idx] * 100)

            # Wyznaczenie pozycji napisów na klatce
            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # Narysowanie napisów na klatce
            cv2.putText(frame, label_gender, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, label_prob, (startX, endY + 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Zapisanie obrazu
            # if (time.time() - last_save_time >= 4) and (conf[idx] > 0.9920):
            #     last_save_time = time.time()
            #     threading.Thread(target=saveImage, args=(croped_image, image_to_save, genders[idx])).start()

        # Wyświetlenie klatki
        cv2.imshow("Camera app", frame)

        # Wciśnięcie klawisza "ESC" kończy działanie aplikacji
        if cv2.waitKey(1) == 27:
            # Zakończenie wątków
            os._exit(0)

    # Zwolnienie zasobów
    camera.release()
    cv2.destroyAllWindows()


def saveImage(image_to_analize, image_to_save, gender):
    # Inicjalizacja zmiennej przechowującej ostatnią zapisaną wartość znacznika czasu
    global last_save_time

    # Predykcja płci na podstawie obrazu
    conf = model.predict(image_to_analize)[0]

    # Pobranie indeksu o największej wartości predykcji
    idx = np.argmax(conf)

    # Sprawdzenie wartości predykcji - jeśli poniżej progu, to zakończ funkcję
    if conf[idx] < 0.9920:
        return

    # Zapisanie obrazu w odpowiednim folderze na podstawie płci
    if gender == 'man':
        path = os.path.join("new_data", "men", "men_{}.jpg".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        cv2.imwrite(path, image_to_save)
        print(path)
    elif gender == 'woman':
        path = os.path.join("new_data", "women", "women_{}.jpg".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        cv2.imwrite(path, image_to_save)

    
    # Zresetowanie znacznika czasu
    # last_save_time = 0
    
    return

def closeCameraApp():
    # Zakończenie działania aplikacji
    return 