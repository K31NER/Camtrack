import cv2
import pandas as pd
import numpy as np
import streamlit as st
import os
import time
from conexiones import *

st.set_page_config(
    page_icon="https://cdn-icons-png.flaticon.com/128/10479/10479750.png",
    page_title="Deteccion de rostro",
    layout="wide"
)
with st.sidebar:
    st.image("https://axis.uninunez.edu.co/images/uninunez/vm/logoqwhite.svg", width=150, use_container_width=True)


st.logo("https://cdn-icons-png.flaticon.com/128/9806/9806214.png")
# Cargar el clasificador de rostro en cascada
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Crear un DataFrame para almacenar los rostros detectados
if not os.path.exists('rostros_detectados.csv'):
    df = pd.DataFrame(columns=["Nombre", "X", "Y", "Ancho", "Alto", "Timestamp"])
    df.to_csv('rostros_detectados.csv', index=False)

# Función para guardar los rostros en CSV
def save_faces_to_csv(faces, frame_id, min_distance=50, time_threshold=5):
    df = pd.read_csv('rostros_detectados.csv')
    rows_to_add = []
    current_time = time.time()
    
    for (x, y, w, h) in faces:
        # Filtrar las detecciones recientes cercanas (dentro de `min_distance`) y recientes (dentro de `time_threshold` segundos)
        recently_detected = df[
            (np.sqrt((df["X"] - x) ** 2 + (df["Y"] - y) ** 2) < min_distance) & 
            (current_time - df["Timestamp"] <= time_threshold)
        ]
        
        # Agregar solo si no hubo una detección cercana recientemente
        if recently_detected.empty:
            nombre = f'Rostro_{frame_id}'
            rows_to_add.append({"Nombre": nombre, "X": x, "Y": y, "Ancho": w, "Alto": h, "Timestamp": current_time})
    
    # Si hay filas nuevas, añadirlas al CSV
    if rows_to_add:
        new_faces_df = pd.DataFrame(rows_to_add)
        df = pd.concat([df, new_faces_df], ignore_index=True)
        df.to_csv('rostros_detectados.csv', index=False)

# Configuración de Streamlit
st.title(":orange[Detección de Rostros en Tiempo Real]")

stframe = st.empty()  # Espacio para mostrar el video
dataframe_placeholder = st.empty()  # Espacio para el DataFrame de rostros

# Iniciar la captura de video
cap = cv2.VideoCapture(url)

# Contador de frames
frame_id = 0

try:
    while True:
        # Leer un cuadro del video
        ret, frame = cap.read()
        if not ret:
            st.error("No se puede leer el video")
            break

        # Convertir el cuadro a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        # Dibujar rectángulos alrededor de los rostros detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Guardar los rostros en el CSV con control de duplicados
        save_faces_to_csv(faces, frame_id)

        # Mostrar el cuadro con los rostros detectados en Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Leer y mostrar el DataFrame actualizado en Streamlit
        df = pd.read_csv('rostros_detectados.csv')
        # Verificar si 'Timestamp' está en las columnas antes de eliminarla
        if 'Timestamp' in df.columns:
            df = df.drop(columns=["Timestamp"])
        dataframe_placeholder.dataframe(df, use_container_width=True)

        # Incrementar el contador de frames
        frame_id += 1

except KeyboardInterrupt:
    st.write("Detección detenida")

# Liberar la captura cuando se cierre la aplicación
cap.release()
