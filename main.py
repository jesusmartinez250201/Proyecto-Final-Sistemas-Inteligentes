import cv2
import mediapipe

import Operations
from Recorder import Recorder
from Camera import Camera

import pandas as pd

if __name__ == "__main__":
    # Obtiene el dataset generado por el programa main_training_data.py
    filenameDataset = "gestures_dataset.pickle"
    reference_signs = pd.read_pickle(filenameDataset)

    #Objeto que almacena resultados de mediapipe y calcula similitudes de signos
    sign_recorder = Recorder(reference_signs)

    #Objeto que dibuja puntos clave y muestra resultados
    webcam_manager = Camera()

    #Enciende la camara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #Configurar el entorno de Mediapipe
    with mediapipe.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():

            #Leer fuente
            ret, frame = cap.read()

            #Hacer detecciones
            image, results = Operations.deteccion_mediapipe(frame, holistic)

            #Procesar resultados
            sign_detected, is_recording = sign_recorder.procesar_resultado(results)

            #Actualice el frame
            webcam_manager.actualizar(frame, results, sign_detected, is_recording)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("r"):  #Grabacion presionando r
                sign_recorder.registro()
            elif pressedKey == ord("q"):  #Detener presionando q
                break

        cap.release()
        cv2.destroyAllWindows()
