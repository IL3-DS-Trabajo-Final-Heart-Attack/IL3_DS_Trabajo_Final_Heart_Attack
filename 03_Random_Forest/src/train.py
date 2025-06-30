import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

# Obtener la ruta del directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Obtener la ruta del directorio padre (03_Random_Forest)
project_dir = os.path.dirname(script_dir)

# Cargar datos
X_train = pd.read_csv(os.path.join(project_dir, "data/processed/X_train.csv"))
y_train = pd.read_csv(os.path.join(project_dir, "data/processed/y_train.csv")).squeeze()

# Entrenar modelo
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    max_features= 'sqrt',
    class_weight="balanced",
    random_state=42,
    oob_score=True,    
)
model.fit(X_train, y_train)

# Guardar modelo
joblib.dump(model, os.path.join(project_dir, "models/rf_model.joblib"))