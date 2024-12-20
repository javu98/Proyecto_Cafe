import pandas as pd

# Cargar el archivo CSV
file_path = r"D:\Documentos\Proyecto_cafe\coffee_analysis.csv"
coffee_data = pd.read_csv(file_path)

# Mostrar las primeras filas
print("Primeras filas del dataset:")
print(coffee_data.head())

# Información general del dataset
print("\nInformación del dataset:")
print(coffee_data.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(coffee_data.describe(include='all'))

# Manejar valores nulos
# Reemplazar valores nulos en 'roast' con 'Unknown'
coffee_data['roast'].fillna('Unknown', inplace=True)

# Reemplazar valores nulos en 'desc_3' con un texto genérico
coffee_data['desc_3'].fillna('No description available', inplace=True)

# Verificar que no hay valores nulos
print("\nValores nulos por columna después del reemplazo:")
print(coffee_data.isnull().sum())

# Verificar duplicados
duplicates = coffee_data.duplicated()
print(f"\nNúmero de filas duplicadas: {duplicates.sum()}")

# Si existen duplicados, eliminarlos
coffee_data = coffee_data.drop_duplicates()
print(f"Después de eliminar duplicados: {coffee_data.shape[0]} filas")

# Convertir la columna review_date a formato datetime
coffee_data['review_date'] = pd.to_datetime(coffee_data['review_date'], errors='coerce')

# Verificar si hay valores no válidos
print(f"\nNúmero de fechas no válidas: {coffee_data['review_date'].isnull().sum()}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Selección de columnas relevantes
features = coffee_data[['roast', '100g_USD', 'loc_country', 'origin_1']]
target = coffee_data['rating']

# Codificación One-Hot para columnas categóricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
features_encoded = encoder.fit_transform(features[['roast', 'loc_country', 'origin_1']])

# Convertir One-Hot Encoding a DataFrame
encoded_columns = encoder.get_feature_names_out(['roast', 'loc_country', 'origin_1'])
features_encoded_df = pd.DataFrame(features_encoded, columns=encoded_columns)

# Concatenar columnas codificadas con las características numéricas
features_final = pd.concat([features_encoded_df, features[['100g_USD']].reset_index(drop=True)], axis=1)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features_final, target, test_size=0.2, random_state=42)

# Verificar formas de los conjuntos
print("\nFormas de los conjuntos:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluación del modelo:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

from sklearn.ensemble import RandomForestRegressor

# Crear y entrenar un modelo de Random Forest
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
rf_y_pred = rf_model.predict(X_test)

# Evaluar el modelo
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

print("\nEvaluación del modelo Random Forest:")
print(f"Mean Squared Error (MSE): {rf_mse:.2f}")
print(f"R2 Score: {rf_r2:.2f}")

import joblib

# Guardar el modelo y el codificador
joblib.dump(rf_model, 'src/rf_model.pkl')
joblib.dump(encoder, 'src/encoder.pkl')  # Guarda el codificador

