import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from dtreeviz import *
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

model = joblib.load('modelo_desordenes.mentales.joblib')#Cargo el modelo entrenado

data_md = pd.read_excel('Mental disorder symptoms.xlsx')  # Cargo los datos de entrenamiento
df_pordesorden= pd.read_csv('porDesorden.csv')

X = data_md.drop(columns=['Desorden'])# Divido los datos en características (X) y etiquetas (y)
y = data_md['Desorden']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# Divido los datos en conjuntos de entrenamiento y prueba

def predecir_desorden(data):# Creo una función para las predicciones
    return model.predict(data)

st.title('Predicción de Desórdenes Mentales')

# Introducción 
st.write('Esta aplicación permite predecir desórdenes mentales basados en características y síntomas. En la barra lateral se encuentra la lista de los desordenes mentales con su respectiva descripción')

# Formulario para ingresar datos
st.header('Ingresar Datos')
edad = st.slider('Edad', 0, 100, 30)  # Ajusto los valores iniciales y los límites

col1, col2 = st.columns(2)

sintomas = {
    'feeling.nervous': 'Síntoma de nerviosismo',
    'panic': 'Síntoma de ataques de pánico',
    'breathing.rapidly': 'Síntoma de respiración rápida',
    'sweating': 'Síntoma de sudoración',
    'trouble.in.concentration': 'Síntoma de problemas de concentración',
    'trouble.sleeping': 'Síntoma de problemas para dormir',
    'trouble.with.work': 'Síntoma de problemas con el trabajo',
    'hopelessness': 'Síntoma de desesperanza',
    'anger': 'Síntoma de enojo',
    'over.react': 'Síntoma de reacciones exageradas',
    'change.in.eating': 'Síntoma de cambios en la alimentación',
    'suicidal.thought': 'Síntoma de pensamientos suicidas',
    'feeling.tired': 'Síntoma de cansancio',
    'close.friend': 'Síntoma de falta de amigos cercanos',
    'social.media.addiction': 'Síntoma de adicción a las redes sociales',
    'weight.gain': 'Síntoma de aumento de peso',
    'introvert': 'Síntoma de introversión',
    'popping.up.stressful.memory': 'Síntoma de recuerdos estresantes que aparecen',
    'nightmares': 'Síntoma de pesadillas',
    'avoids.people.or.activities': 'Síntoma de evitar personas o actividades',
    'feeling.negative': 'Síntoma de pensamientos negativos',
    'trouble.concentrating': 'Síntoma de problemas de concentración',
    'blaming.yourself': 'Síntoma de culparse a sí mismo',
    'hallucinations': 'Síntoma de alucinaciones',
    'repetitive.behaviour': 'Síntoma de comportamiento repetitivo',
    'seasonally': 'Síntoma estacional',
    'increased.energy': 'Síntoma de aumento de energía'
}

valores = {}# Creo un diccionario para almacenar los valores ingresados

valores['age'] = edad

sintoma_keys = list(sintomas.keys())# Itero a través del diccionario de síntomas y coloco la mitad en cada columna
for i in range(len(sintoma_keys)):
    sintoma = sintoma_keys[i]
    descripcion = sintomas[sintoma]
    clave = sintoma  # Uso el nombre del síntoma como clave única
    if i < len(sintoma_keys) // 2:
        valores[sintoma] = col1.checkbox(descripcion, False, key=clave)
    else:
        valores[sintoma] = col2.checkbox(descripcion, False, key=clave)

data = pd.DataFrame(valores, index=[0])# Convierto los valores ingresados en un DataFrame

if st.button('Predecir'):# Realizo predicción
    resultado = predecir_desorden(data)
    st.success(f'Predicción: {resultado[0]}')  # Muestra el resultado en la barra lateral

# Información adicional
st.markdown("### Información Adicional")
st.write('Este modelo de predicción se basa en datos y síntomas recopilados de Kaggle. Este modelo se proporciona únicamente con fines informativos y de entretenimiento. No tiene fines médicos y no debe reemplazar la opinión de un profesional de la salud. Si experimentas cualquier síntoma o inquietud sobre tu salud mental, te recomendamos encarecidamente que consultes a un médico o profesional de la salud mental para recibir orientación y atención adecuada.')

st.header('Gráfica de Árbol de Decisión')

# URL de la imagen en GitHub
imagen_url = 'https://github.com/marcebalzarelli/Modelo_Desordenes_Mentales/raw/main/DTreeViz_1288.png'

st.image(imagen_url, caption='Gráfica de Árbol de decisión', use_column_width=True)# Muestro la imagen desde la URL de GitHub

nombres_clases = ["ADHD", "ASD", "ED", "Loneliness", "MDD", "OCD", "PDD", "PTSD", "anxiety", "bipolar", "psychot depresn", "sleep disord"]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

m = dtreeviz(# Gráfico de árbol de decisión
    model, 
    X_train, 
    y_train_encoded,
    target_name= nombres_clases,
    feature_names=list(X_train.columns)
)

if st.button('Descargar Árbol de Decisión'):# Agrego un botón para permitir a los usuarios descargar el gráfico
    m.save("arbol_decision.svg")
    st.success('El Árbol de Decisión ha sido descargado como imagen (arbol_decision.svg).')


# Elijo las columnas relevantes para el gráfico de radar
columns = ['sensación.de.nerviosismo', 'panico', 'respiración.rápida', 'sudoración', 'dificultad.en.la.concentración',
           'dificultad.para.dormir', 'dificultad.con.el.trabajo', 'desesperanza', 'enojo', 'reaccion.exagerada', 'cambio.en.la.alimentación',
           'pensamientos.suicidas', 'sensación.de.cansancio', 'falta.amistad', 'adicción.a.las.redes.sociales', 'aumento.de.peso',
           'introvertido', 'recuerdo.de.eventos.estresantes', 'pesadillas', 'evita.gente.o.actividades',
           'sensación.negativa', 'dificultad.para.concentrarse', 'culparse.a.sí.mismo', 'alucinaciones',
           'comportamiento.repetitivo', 'aumento.estacional.de.energía']

values = df_pordesorden[columns].values.tolist()# Creo una lista de valores para cada fila del DataFrame

fig = go.Figure()# Creo gráfico de radar interactivo

for i, row in enumerate(values):
    fig.add_trace(go.Scatterpolar(
        r=row,
        theta=columns,
        fill='toself',
        name=nombres_clases[i]
    ))

fig.update_layout(
    title="Gráfica de Radar de Desordenes Mentales",  
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100] 
        )),
    showlegend=True
)

st.plotly_chart(fig)

#Grafico de Barras interactivo
column_to_plot = 'dificultad.para.dormir' 

fig_bar = px.bar(df_pordesorden, x='Desorden', y='dificultad.para.dormir',
                 title=f'Gráfico de Barras de Síntoma de Dificultad para dormir',
                 color='Desorden', color_discrete_sequence=px.colors.qualitative.Set1)
fig_bar.update_xaxes(title_text='Tipo de Trastorno')
fig_bar.update_yaxes(title_text='dificultad.para.dormir')

st.plotly_chart(fig_bar)


# Gráfico interactivo de barras
fig = px.bar(data_md, x='Desorden', title='Cantidad de Pacientes por Tipo de Trastorno',
             color='Desorden', color_discrete_sequence=px.colors.qualitative.Set1)
fig.update_xaxes(title_text='Tipo de Trastorno')
fig.update_yaxes(title_text='Cantidad de Pacientes')

st.plotly_chart(fig)

# Gráfico interactivo de boxplot 
fig = px.box(data_md, x='Desorden', y='Edad',
             title='Boxplot de Edad por Desorden Mental',
             color='Desorden', color_discrete_sequence=px.colors.qualitative.Set1)
fig.update_xaxes(title_text='Desorden Mental')
fig.update_yaxes(title_text='Edad')

st.plotly_chart(fig)

# Gráfico de dispersión interactivo
fig = px.scatter(df_pordesorden, x='dificultad.en.la.concentración', y='adicción.a.las.redes.sociales', 
                 title='Relación entre Dificultad para Concentrarse y Adicción a Redes Sociales')

fig.update_xaxes(title_text='Dificultad para Concentrarse')
fig.update_yaxes(title_text='Adicción a Redes Sociales')

st.plotly_chart(fig)

#Lista de desordenes mentales de la barra lateral
st.sidebar.header("Lista de Desordenes Mentales")
st.sidebar.write("'MDD' (Major Depressive Disorder): También conocido como trastorno depresivo mayor o depresión clínica, es un trastorno del estado de ánimo caracterizado por una profunda tristeza, pérdida de interés o placer en las actividades y otros síntomas que afectan la capacidad de una persona para funcionar en la vida diaria.")
st.sidebar.write("'ASD' (Autism Spectrum Disorder): Se refiere al trastorno del espectro autista, que es un trastorno del neurodesarrollo que afecta la comunicación, la interacción social y el comportamiento de una persona. Puede variar en gravedad y síntomas.")
st.sidebar.write("'Loneliness': En este caso, se menciona la soledad, que no es un trastorno mental en sí, pero puede estar relacionada con problemas de salud mental cuando es crónica y debilitante.")
st.sidebar.write("'Bipolar': Se refiere al trastorno bipolar, que implica cambios extremos en el estado de ánimo, incluyendo episodios maníacos (elevado) y episodios depresivos (deprimido).")
st.sidebar.write("'Anxiety': Hace referencia a los trastornos de ansiedad, que incluyen trastornos como el trastorno de ansiedad generalizada (TAG) y el trastorno de pánico, caracterizados por la ansiedad excesiva y preocupación.")
st.sidebar.write("'PTSD' (Post-Traumatic Stress Disorder): Trastorno de estrés postraumático, que se desarrolla después de una experiencia traumática y puede involucrar síntomas como pesadillas, flashbacks y evitación de situaciones relacionadas con el trauma.")
st.sidebar.write("'Sleep Disord' (Sleep Disorder): Se refiere a los trastornos del sueño, que pueden incluir insomnio, apnea del sueño y otros trastornos que afectan la calidad del sueño.")
st.sidebar.write("'Psychot Depresn' (Psychotic Depression): Depresión psicótica, que es una forma de depresión mayor que involucra síntomas psicóticos como alucinaciones o delirios.")
st.sidebar.write("'ED' (Eating Disorder): Trastornos de la alimentación, que incluyen anorexia nerviosa, bulimia nerviosa y otros trastornos relacionados con la alimentación.")
st.sidebar.write("'ADHD' (Attention-Deficit/Hyperactivity Disorder): Trastorno por déficit de atención con hiperactividad, un trastorno del neurodesarrollo que afecta la atención, la impulsividad y la hiperactividad.")
st.sidebar.write("'PDD' (Pervasive Developmental Disorder): Se refiere a los trastornos generalizados del desarrollo, que incluyen el autismo y otros trastornos del espectro autista.")
st.sidebar.write("'OCD' (Obsessive-Compulsive Disorder): Trastorno obsesivo-compulsivo, caracterizado por pensamientos obsesivos y comportamientos compulsivos repetitivos.")

st.markdown('<hr>', unsafe_allow_html=True)

# Pie de página personalizado
st.write("@2023 Hecho por María Marcela Balzarelli")