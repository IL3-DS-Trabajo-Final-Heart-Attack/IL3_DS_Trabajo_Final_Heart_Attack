import pandas as pd
import joblib

# Diccionario con los valores del paciente (modifica estos valores según el caso real)
valores_paciente = {
    "age": 60,
    "gender": 1,   # 1 = hombre, 0 = mujer
    "hr": 80,
    "sbp": 120,
    "dbp": 92,
    "bs": 100,     # valor de azúcar en sangre
    "ckmb": 12,
    "trop": 0.014
}

# Crear DataFrame
nuevo_paciente = pd.DataFrame([valores_paciente])

# Cargar el modelo
model = joblib.load("03_Random_Forest/models/rf_model.joblib")

# Predecir
prediccion = model.predict(nuevo_paciente)
proba = model.predict_proba(nuevo_paciente)

print("Valores del paciente:")
print(valores_paciente)
print("\nResultado de la predicción:")
print("¿Heart Attack?:", "Sí" if prediccion[0] == 1 else "No")
print("Probabilidad (No, Sí):", proba[0])
