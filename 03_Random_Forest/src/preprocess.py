import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Obtener la ruta del directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Obtener la ruta del directorio padre (03_Random_Forest)
project_dir = os.path.dirname(script_dir)

# Cargar dataset
df = pd.read_csv(os.path.join(project_dir, 'data/processed/medicaldataset.csv'))

# Separar features (X) y target (y)
X = df.drop('res', axis=1)
y = df['res']

# Dividir en train y test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  
)

# Guardar datasets procesados
X_train.to_csv(os.path.join(project_dir, 'data/processed/X_train.csv'), index=False)
X_test.to_csv(os.path.join(project_dir, 'data/processed/X_test.csv'), index=False)
y_train.to_csv(os.path.join(project_dir, 'data/processed/y_train.csv'), index=False)
y_test.to_csv(os.path.join(project_dir, 'data/processed/y_test.csv'), index=False)
