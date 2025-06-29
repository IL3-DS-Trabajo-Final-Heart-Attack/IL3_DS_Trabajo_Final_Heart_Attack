import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Cargar datos y modelo
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
model = joblib.load("models/rf_model.joblib")

# Evaluar
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


# Guardar y mostrar resultados
with open("reports/classification_report.txt", "w") as f:
    f.write(report)
print(f"🔎 Accuracy: {acc:.4f}")
print("Reporte guardado:\n", report)