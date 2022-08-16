import re
from pytube import YouTube
import os
from shutil import copyfile
import pandas as pd
from tqdm import tqdm
import Operations

FOLDER = os.path.join("data", "videos")


print("Creando dataset... esto puede tardar varios minutos")
#Cree un dataset de los videos donde aún no se han extraído puntos de referencia
videos = Operations.cargar_dataset()

#Crea un marco de datos de signos de referencia (nombre, modelo, distancia)
Operations.cargar_referencia_señales(videos)
print("¡Dataset gestures_dataset creado correctamente!")