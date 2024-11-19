import pandas as pd
import streamlit as st 
import plotly.express as px


# Diccionario que asocia colores a facultades
facultades_por_color = {
    'Cian': 'Facultad de Ingenieria',
    'Azul Oscuro': 'Facultad de Medicina',
    'Marrón Dorado': 'Facultad de Ciencias contables',
    'Azul Claro': 'Facultad de Ingenieria',
    'Azul Muy Claro': 'Facultad de Medicina',
    'Lavanda': 'Facultad de Humanidades'
}

# Cargar el archivo CSV de detecciones de colores
df = pd.read_csv('registro_colores.csv')

# Agregar una nueva columna "Facultad" basada en el color detectado
df['Facultad'] = df['Color'].map(facultades_por_color)

# Guardar el archivo CSV actualizado
df.to_csv('colores_facultad.csv', index=False)
#lo volvemos un dataframe
df_facultad = pd.read_csv("colores_facultad.csv")

df_renombrado = df_facultad.rename(columns={"Color": "Registro"})

# Contar los ingresos por facultad (basado en los registros renombrados)
ingreso_facultad = df_renombrado.groupby("Facultad")["Registro"].count().reset_index()


def graficar(df):
    st.subheader("Gráficas", divider="red")
    st.dataframe(df, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)

    # Definir el color específico para cada facultad
    facultad_colores = {
        'Ingeniería en Sistemas': 'cyan',
        'Ciencias Contables': 'brown',
        'Ciencias Sociales': 'purple',
        'Medicina': 'blue'
    }

    # Definir el color específico para cada tipo de registro (Color)
    color_colores = {
        'Cian': 'cyan',
        'Azul Oscuro': '#00008B',
        'Marrón Dorado': '#efb810',
        'Azul Claro': '#ADD8E6',
        'Azul Muy Claro': '#2271b3',
        'Lavanda': '#e6e6fa'
    }

    # Gráfico de pastel para ver la proporción de registros por facultad
    with c1:
        registros_facultad = df.groupby('Facultad')['Color'].count().reset_index()
        registros_facultad.columns = ['Facultad', 'Registro total']
        
        fig_pie = px.pie(
            registros_facultad,
            values='Registro total',
            names='Facultad',
            title='Proporción de registros por facultad',
            color='Facultad',
            color_discrete_map=facultad_colores  # Usar colores de facultad
        )
        st.plotly_chart(fig_pie)

    # Gráfico de barras para contar los registros de cada color por facultad
    with c2:
        registros_color = df.groupby(['Facultad', 'Color']).size().reset_index(name='Cantidad')
        fig_bar = px.bar(
            registros_color,
            x='Facultad',
            y='Cantidad',
            color='Color',
            title='Cantidad de registros de colores por facultad',
            barmode='stack',
            color_discrete_map=color_colores  # Usar colores específicos para 'Color'
        )
        st.plotly_chart(fig_bar)

    # Gráfico de área para ver la distribución temporal de los registros de colores
    # Convertir Timestamp a datetime y extraer la fecha
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Fecha'] = df['Timestamp'].dt.date

    # Contar registros por día y facultad
    registros_por_dia_y_facultad = df.groupby(['Fecha', 'Facultad']).size().reset_index(name='Cantidad')

    # Gráfico de área apilada
    fig_area = px.area(
        registros_por_dia_y_facultad,
        x='Fecha',
        y='Cantidad',
        color='Facultad',
        title='Distribución acumulada de registros por facultad en el tiempo',
        labels={'Fecha': 'Fecha', 'Cantidad': 'Cantidad de Registros'},
        color_discrete_map=facultad_colores  # Usar colores de facultad
    )
    st.plotly_chart(fig_area)


def estadisticas(df_facultad):
    # Convertir la columna 'Timestamp' a tipo datetime
    df_facultad['Timestamp'] = pd.to_datetime(df_facultad['Timestamp'], errors='coerce')
    df_facultad['Fecha'] = df_facultad['Timestamp'].dt.date

    # Agrupar los registros por día para encontrar el día con más registros
    registros_por_dia = df_facultad.groupby('Fecha').size().reset_index(name='Cantidad')
    dia_con_mas_registros = registros_por_dia.loc[registros_por_dia['Cantidad'].idxmax(), 'Fecha']
    max_registros_dia = registros_por_dia['Cantidad'].max()

    # Agrupar por facultad y contar registros
    ingreso_facultad = df_facultad.groupby("Facultad")["Color"].count().reset_index(name="Registro")

    # Calcular la facultad con el mayor número de ingresos
    facultad_max_ingresos = ingreso_facultad.loc[ingreso_facultad["Registro"].idxmax(), "Facultad"]
    max_ingresos = ingreso_facultad["Registro"].max()

    # Calcular el promedio de registros por facultad
    promedio_registros = ingreso_facultad['Registro'].mean()

    # Visualizar las estadísticas en columnas
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Facultad con más ingresos", facultad_max_ingresos, f"Ingresos: {max_ingresos}")
        
    with c2:
        st.metric("Promedio de registros por facultad", promedio_registros)
    
    with c3:
        # Convertir el día con más registros a string para mostrarlo en st.metric
        st.metric("Día con más registros", str(dia_con_mas_registros), f"Registros: {max_registros_dia}")

        
