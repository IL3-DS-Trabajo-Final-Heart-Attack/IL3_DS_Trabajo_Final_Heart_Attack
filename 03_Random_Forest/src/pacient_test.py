import pandas as pd
import joblib
import os

# Obtener la ruta del directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Obtener la ruta del directorio padre (03_Random_Forest)
project_dir = os.path.dirname(script_dir)

# Diccionario con los valores del paciente (modifica estos valores según el caso real)
valores_paciente = {
    "age": 60,
    "gender": 1, 
    "hr": 80,
    "sbp": 120,
    "dbp": 92,
    "bs": 100, 
    "ckmb": 12,
    "trop": 0.014
}

# Crear DataFrame
nuevo_paciente = pd.DataFrame([valores_paciente])

# Cargar el modelo
model = joblib.load(os.path.join(project_dir, "models/rf_model.joblib"))

# Predecir
prediccion = model.predict(nuevo_paciente)
proba = model.predict_proba(nuevo_paciente)

print("Valores del paciente:")
print(valores_paciente)
print("\nResultado de la predicción:")
print("¿Heart Attack?:", "Sí" if prediccion[0] == 1 else "No")
print("Probabilidad (No, Sí):", proba[0])
