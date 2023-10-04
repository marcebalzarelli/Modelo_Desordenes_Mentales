# Modelo de Predicción de Desórdenes Mentales

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo de aprendizaje automático que pueda predecir desórdenes mentales en individuos a partir de una serie de características y síntomas observados. Los desórdenes mentales son un problema de salud importante en todo el mundo, y contar con una herramienta de predicción precisa puede ayudar en la detección temprana y la intervención efectiva.

## Fuente de Datos

Los datos utilizados en este proyecto se obtuvieron de Kaggle y están disponibles en el siguiente enlace: [Enlace](https://www.kaggle.com/datasets/baselbakeer/mental-disorders-dataset). Estos datos contienen información sobre la edad de los individuos, así como varios síntomas y características relacionadas con la salud mental.

## Análisis Exploratorio de Datos (EDA)

Se realizó un detallado Análisis Exploratorio de Datos (EDA) como parte de este proyecto. Esto incluyó la visualización de datos, la identificación de tendencias y patrones, y la exploración de la relación entre las variables. Los resultados del EDA se utilizaron para comprender mejor los datos y tomar decisiones informadas durante el proceso de modelado.

## Diccionario de Datos y Descripción de Siglas

El proyecto incluye dos archivos en formato TXT que son importantes para comprender los datos:

- `diccionario de datos.txt`: Este archivo contiene un diccionario de datos que describe el significado y la interpretación de cada columna en el conjunto de datos. Proporciona información detallada sobre las variables utilizadas en el proyecto.

- `Desordenes mentales.txt`: En este archivo se encuentran las descripciones de las siglas utilizadas para representar los desórdenes mentales en los datos. Ayuda a comprender las abreviaturas y términos utilizados en el conjunto de datos.

## Metodología
El proyecto se desarrolló siguiendo los siguientes pasos:

1. **Carga de Datos**: Se obtuvieron los datos desde la fuente de Kaggle utilizando la biblioteca `pandas`. Los datos se almacenaron en un DataFrame para su análisis posterior.

2. **Preprocesamiento de Datos**: Se realizó una limpieza y transformación de los datos para asegurar que estuvieran en un formato adecuado para el modelado. Esto incluyó la corrección de errores ortográficos en los tipos de desórdenes y la renombración de columnas.

3. **Exploración de Datos**: Se llevaron a cabo análisis exploratorios para comprender mejor la distribución de las variables, identificar patrones y determinar las características más relevantes para el modelo.

4. **Creación y Evaluación del Modelo**: Se implementó un modelo de clasificación de Árbol de Decisión utilizando la biblioteca `scikit-learn`. Los datos se dividieron en conjuntos de entrenamiento y prueba, y se evaluó la precisión del modelo en el conjunto de prueba. Se generó un informe de clasificación para evaluar el rendimiento.

5. **Validación Cruzada y Optimización de Hiperparámetros**: Se realizó una validación cruzada para garantizar la estabilidad del modelo y se llevó a cabo una búsqueda en cuadrícula para encontrar los mejores hiperparámetros.

6. **Guardado del Modelo**: Una vez que se obtuvieron los mejores hiperparámetros, se entrenó un modelo final y se guardó utilizando `joblib` para su uso posterior.

## Despliegue en Streamlit

El modelo entrenado se implementó en una aplicación web interactiva utilizando Streamlit. La aplicación permite a los usuarios ingresar información sobre síntomas y características personales para predecir posibles desórdenes mentales. Link de la app de Streamlit: https://modelodesordenesmentales.streamlit.app/

## Instrucciones de Uso

Si deseas utilizar este modelo para realizar predicciones en nuevos datos, puedes seguir estos pasos:

1. Clona este repositorio en tu máquina local:

   ```bash
   git clone https://github.com/TU_USUARIO/nombre-del-repositorio.git

2. Asegúrate de tener Python 3.x instalado en tu entorno.

```bash
pip install pandas seaborn matplotlib numpy scikit-learn

```
Carga tus propios datos en un DataFrame de Pandas siguiendo el formato adecuado.

Utiliza el modelo entrenado para hacer predicciones en tus datos.

## Contribuciones

Si deseas contribuir a este proyecto, ¡eres bienvenido! Puedes abrir problemas, enviar solicitudes de extracción o ayudar con la documentación.

## Contacto

Para cualquier pregunta o comentario relacionado con este proyecto, puedes contactarme en:

- Correo Electrónico: marcebalzarelli@gmail.com
- LinkedIn: [Marcela Balzarelli](https://www.linkedin.com/in/marcela-balzarelli/)

   
