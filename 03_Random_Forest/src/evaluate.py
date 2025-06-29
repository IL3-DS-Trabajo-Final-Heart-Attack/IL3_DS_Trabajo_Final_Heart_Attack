import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score, balanced_accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import export_graphviz, plot_tree
import os

# Cargar datos y modelo
X_test = pd.read_csv("03_Random_Forest/data/processed/X_test.csv")
y_test = pd.read_csv("03_Random_Forest/data/processed/y_test.csv").squeeze()
model = joblib.load("03_Random_Forest/models/rf_model.joblib")

# Evaluar
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


# Guardar y mostrar resultados
with open("03_Random_Forest/reports/classification_report.txt", "w") as f:
    f.write(report)
print(f"🔎 Accuracy: {acc:.4f}")
print("Reporte guardado:\n", report)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
np.savetxt("03_Random_Forest/reports/confusion_matrix.txt", cm, fmt='%d')
print("Matriz de confusión:\n", cm)

# Balanced Accuracy y Cohen's Kappa
bal_acc = balanced_accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")

# Importancia de variables
importances = model.feature_importances_
features = X_test.columns
with open("03_Random_Forest/reports/feature_importances.txt", "w") as f:
    for feat, imp in zip(features, importances):
        f.write(f"{feat}: {imp:.4f}\n")
print("Importancia de variables:")
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.4f}")


y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.savefig("03_Random_Forest/reports/roc_curve.png")
plt.close()
print(f"AUC: {auc:.4f}")

# Guardar datos de la curva ROC para Streamlit
roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
roc_df.to_csv("03_Random_Forest/reports/roc_curve_data.csv", index=False)

# Guardar los primeros 10 árboles del Random Forest como imágenes PNG
os.makedirs("03_Random_Forest/reports/trees", exist_ok=True)
for i, estimator in enumerate(model.estimators_[:10]):
    plt.figure(figsize=(20, 10))
    plot_tree(estimator, feature_names=X_test.columns, class_names=[str(c) for c in np.unique(y_test)], filled=True, rounded=True, fontsize=8, max_depth=3)
    img_filename = f"03_Random_Forest/reports/trees/tree_{i+1}.png"
    plt.savefig(img_filename, bbox_inches='tight')
    plt.close()
    print(f"Árbol {i+1} guardado como imagen en {img_filename}")
