import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, roc_auc_score
from reportes import show_classification_report_table, show_feature_importance_chart, show_roc_curve_section
from trees import show_tree_images
from prediction import show_patient_prediction_section
from prediction_utils import load_model, predict
from creditos import show_credits_section

# Obtener la ruta del directorio actual del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_feature_importance(model, features, ax=None):
    importances = model.feature_importances_
    if ax is None:
        fig, ax = plt.subplots()
    ax.barh(features, importances, color='skyblue')
    ax.set_xlabel('Importancia')
    ax.set_ylabel('Variable')
    ax.set_title('Importancia de las variables en el modelo')
    return ax

def main():
    st.set_page_config(page_title="Predicción de Infarto", layout="wide")
    st.title("Predicción de Riesgo de Infarto")

    menu = st.sidebar.radio(
        "Navegación",
        [
            "Predicción de paciente",
            "Reportes del Modelo",
            "Árboles de decisión",
            "Créditos"
        ]
    )

    if menu == "Predicción de paciente":
        show_patient_prediction_section()

    elif menu == "Reportes del Modelo":
        st.subheader("Reportes del Modelo")
        col1, col2, col3 = st.columns(3)
        with col1:
            show_classification_report_table(os.path.join(SCRIPT_DIR, "reports/classification_report.txt"))
        with col2:
            # Cargar datos de test
            X_test = pd.read_csv(os.path.join(SCRIPT_DIR, "data/processed/X_test.csv"))
            y_test = pd.read_csv(os.path.join(SCRIPT_DIR, "data/processed/y_test.csv")).squeeze()
            model = load_model(os.path.join(SCRIPT_DIR, "models/rf_model.joblib"))
            show_roc_curve_section(model, X_test, y_test)
        with col3:
            model = load_model(os.path.join(SCRIPT_DIR, "models/rf_model.joblib"))
            show_feature_importance_chart(model, ["age", "gender", "hr", "sbp", "dbp", "bs", "ckmb", "trop"])

    elif menu == "Árboles de decisión":
        st.subheader("Primeros 9 árboles de decisión del modelo")
        show_tree_images(os.path.join(SCRIPT_DIR, "reports/trees"))

    elif menu == "Créditos":
        show_credits_section()

if __name__ == "__main__":
    main()  