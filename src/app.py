import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar el modelo y el codificador
rf_model = joblib.load('src/rf_model.pkl')
encoder = joblib.load('src/encoder.pkl')

# Título principal y descripción del proyecto
st.title("🌟 Predicción de Calificación de Café 🌟")
st.markdown("""
¡Bienvenido a nuestra aplicación de predicción de calificaciones para cafés! ☕  

### Descripción del proyecto:
Esta herramienta interactiva utiliza un conjunto de datos que organiza reseñas globales de café recopiladas entre **2017 y 2022**, basado en factores como:
- **Nombre de la mezcla**
- **Tipo de tueste**
- **Precio por 100 gramos**
- **Origen geográfico de los granos de café**

Nuestro objetivo es predecir la **calificación promedio** que un café recibiría, según las características seleccionadas. Para ello, usamos un modelo de **Random Forest**, entrenado con datos reales para garantizar precisión en las predicciones. 

### ¿Cómo funciona?
1. Selecciona las características de tu café (tueste, precio, origen, etc.).
2. Haz clic en el botón **"Predecir"**.
3. Obtendrás la calificación predicha en una escala de 0 a 100.  

Esta aplicación combina análisis de datos y aprendizaje automático para ayudarte a entender qué hace que un café sea altamente valorado. ¡Disfruta explorando! 🌎☕  
""")

# Divider (línea para dividir secciones)
st.markdown("---")

# Entrada de usuario
st.subheader("Selecciona las características del café")
roast = st.selectbox("Selecciona el tueste:", ["Light", "Medium", "Medium-Dark", "Dark", "Unknown"])
price = st.slider("Precio por 100g (USD):", min_value=0.1, max_value=50.0, step=0.1)
country = st.selectbox("País del tostador:", ["United States", "Ethiopia", "Colombia", "Unknown"])
origin = st.selectbox("Origen del café:", ["Africa", "Asia", "South America", "Unknown"])

# Divider
st.markdown("---")

# Predicción
st.subheader("Resultado de la predicción")
if st.button("Predecir"):
    # Crear el DataFrame con los valores ingresados
    input_data = pd.DataFrame([[roast, price, country, origin]], columns=['roast', '100g_USD', 'loc_country', 'origin_1'])
    
    # Transformar las características categóricas usando el codificador
    input_encoded = encoder.transform(input_data[['roast', 'loc_country', 'origin_1']])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(['roast', 'loc_country', 'origin_1']))

    # Concatenar las columnas codificadas con la característica numérica
    final_input = pd.concat([input_encoded_df, input_data[['100g_USD']].reset_index(drop=True)], axis=1)

    # Realizar la predicción
    prediction = rf_model.predict(final_input)
    st.success(f"La calificación predicha para este café es: **{prediction[0]:.2f}**")

