import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Cargar datos
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

# Entrenar modelo
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    max_features= 'sqrt',
    class_weight="balanced",
    random_state=42,
    oob_score=True
)
model.fit(X_train, y_train)

# Guardar modelo
joblib.dump(model, "models/rf_model.joblib")