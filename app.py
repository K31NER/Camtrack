import os
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import time
from conexiones import *
from analisis import *

st.set_page_config(
    page_icon="web-cam.png",
    page_title="camtrack",
    layout="wide"
)
st.logo("color-circle.png")

with st.sidebar:
    st.image("https://axis.uninunez.edu.co/images/uninunez/vm/logoqwhite.svg", width=150,use_container_width=True)

#cargamos los datos
df_facultad = pd.read_csv("colores_facultad.csv")

# Rango de colores en HSV
color_ranges = {
    "Cian": ([80, 50, 70], [90, 255, 255]),
    "Azul Oscuro": ([100, 30, 25], [120, 70, 55]),
    "Marrón Dorado": ([20, 100, 100], [30, 200, 180]),
    "Azul Claro": ([100, 150, 200], [130, 255, 255]),
    "Azul Muy Claro": ([100, 0, 150], [130, 50, 255]),
    "Lavanda": ([130, 40, 120], [160, 120, 200])
}

font = cv2.FONT_HERSHEY_SIMPLEX
detected_colors = []  # Lista para almacenar los colores detectados
last_detection_time = {color: 0 for color in color_ranges}  # Tiempo de última detección por color

def dibujar(mask, color, nombre, frame):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            M = cv2.moments(c)
            if (M["m00"] == 0): M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M['m01'] / M['m00'])
            nuevoContorno = cv2.convexHull(c)
            cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(frame, nombre, (x + 10, y - 10), font, 0.75, color, 2, cv2.LINE_AA)
            
            # Agregar la detección si ha pasado suficiente tiempo desde la última
            current_time = time.time()
            if current_time - last_detection_time[nombre] > 5:  # 5 segundos de pausa
                detected_colors.append((current_time, nombre))
                last_detection_time[nombre] = current_time  # Actualizar el último tiempo de detección
                guardar_csv()  # Guardar cada vez que se detecta un color

def get_frame():
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Procesar cada color en el diccionario
        for nombre, (bajo, alto) in color_ranges.items():
            mask = cv2.inRange(frameHSV, np.array(bajo, np.uint8), np.array(alto, np.uint8))
            color_rgb = tuple(int(c) for c in np.array(bajo, np.uint8))  # Color aproximado para dibujar
            
            dibujar(mask, color_rgb, nombre, frame)  # Dibuja si cumple la condición de tiempo

        # Convierte el frame a formato RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame

    cap.release()

# Guardar los datos en dos CSV
def guardar_csv():
    # Asegurarse de que los archivos existan con los encabezados si están vacíos o no existen
    registro_columns = ["Timestamp", "Color"]
    facultad_columns = ["Timestamp", "Color", "Facultad"]
    
    if not os.path.exists("registro_colores.csv") or os.stat("registro_colores.csv").st_size == 0:
        pd.DataFrame(columns=registro_columns).to_csv("registro_colores.csv", index=False)
    
    if not os.path.exists("colores_facultad.csv") or os.stat("colores_facultad.csv").st_size == 0:
        pd.DataFrame(columns=facultad_columns).to_csv("colores_facultad.csv", index=False)
    
    # Crear DataFrame temporal con los colores detectados
    df = pd.DataFrame(detected_colors, columns=["Timestamp", "Color"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv("registro_colores.csv", index=False, mode="a", header=False)
    
    # Añadir la columna de "Facultad" usando el diccionario
    df['Facultad'] = df['Color'].map(facultades_por_color)
    df.to_csv("colores_facultad.csv", index=False, mode="a", header=False)

st.title(":blue[Cam]:red[Track]📸")

# Crea un espacio para mostrar el video
st.subheader("Vista :red[ESP32-CAM]", divider="red")

#Mostramos las estadisticas
estadisticas(df_facultad)
stframe = st.empty()
dataframe_placeholder = st.empty()  # Espacio para la tabla de datos
st.subheader("Analisis",divider="red")

# Definir estilos condicionales
# Definir estilos condicionales
def color_condicional(color):
    if color == "Cian":
        return 'color: cyan; font-weight: bold;'  # Estilo para cian
    elif color == "Azul Oscuro":
        return 'color:  #00008B;'  # Azul oscuro
    elif color == "Marrón Dorado":
        return 'color: #efb810;'  # Marrón dorado
    elif color == "Azul Claro":
        return 'color: #ADD8E6; ' # Azul claro
    elif color == "Azul Muy Claro":
        return 'color: #2271b3;'  # Azul muy claro
    elif color == "Lavanda":
        return 'color: #e6e6fa;'  # Lavanda
    return ''  # Sin estilo por defecto

# Aplicar estilos condicionales a las columnas específicas
styled_df = df_facultad.style.applymap(color_condicional, subset=["Color"])
st.dataframe(styled_df,use_container_width=True)

st.sidebar.subheader(":orange[Funciones]", divider="blue")
if st.sidebar.button(":orange[Graficar]", help="Haz clic para graficar", disabled=False,use_container_width=True):
    graficar(df_facultad)


# Captura y muestra el video
try:
    for frame in get_frame(): 
        stframe.image(frame, channels="RGB", width=800)
        
        # Actualiza la tabla de colores detectados
        if detected_colors:
            # Convertir la marca de tiempo a un formato legible
            df = pd.DataFrame(detected_colors, columns=["Timestamp", "Color"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
            dataframe_placeholder.dataframe(df, use_container_width=True)  
            
except KeyboardInterrupt:
    # Detener la captura y guardar el CSV al salir
    guardar_csv()
    st.write("Captura detenida. Colores guardados en colores_detectados.csv.")
