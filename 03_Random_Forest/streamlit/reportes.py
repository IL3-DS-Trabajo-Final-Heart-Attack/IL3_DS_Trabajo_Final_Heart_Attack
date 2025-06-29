import streamlit as st
import pandas as pd
import os
from sklearn.metrics import roc_curve, roc_auc_score
from prediction_utils import load_model


def show_classification_report_table(report_path):
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip() for line in f if line.strip()]
        # Buscar encabezado
        header = None
        for line in lines:
            if line.strip().startswith('precision'):
                header = line.split()
                break
        if not header:
            st.info("No se encontró encabezado en el classification report.")
            return
        data = []
        index = []
        for line in lines:
            if line.startswith('precision') or line.startswith('recall') or line.startswith('f1-score') or line.startswith('support'):
                continue
            parts = line.split()
            if not parts:
                continue
            # Clases normales (0, 1, etc)
            if parts[0].isdigit():
                index.append(parts[0])
                data.append(parts[1:])
            # accuracy
            elif parts[0] == 'accuracy':
                index.append('accuracy')
                # accuracy row: ['', '', accuracy, support]
                row = ['', '', parts[1], parts[2]]
                data.append(row)
            # macro avg y weighted avg
            elif parts[0] in ('macro', 'weighted') and parts[1] == 'avg':
                index.append(f'{parts[0]} avg')
                data.append(parts[2:])
        # Convertir a DataFrame
        df = pd.DataFrame(data, index=index, columns=header)
        st.subheader("Classification Report")
        st.dataframe(df)
        st.markdown("""
        **¿Qué significa cada métrica?**
        - **Precision:** Proporción de verdaderos positivos entre todos los casos predichos como positivos. Indica cuántas de las predicciones positivas fueron correctas.
        - **Recall (Sensibilidad):** Proporción de verdaderos positivos entre todos los casos realmente positivos. Indica cuántos de los positivos reales fueron identificados correctamente.
        - **F1-score:** Media armónica entre precision y recall. Es una métrica balanceada que considera ambos aspectos.
        - **Support:** Número de muestras reales de cada clase en los datos de prueba.
        
        **¿Qué significa cada fila?**
        - **0:** Métricas calculadas para la clase 0 ("No infarto").
        - **1:** Métricas calculadas para la clase 1 ("Infarto").
        - **accuracy:** Precisión global del modelo considerando todas las clases.
        - **macro avg:** Promedio simple de las métricas para todas las clases, sin ponderar por el número de muestras.
        - **weighted avg:** Promedio ponderado de las métricas para todas las clases, considerando la cantidad de muestras de cada clase.
        """)
    else:
        st.info("No se encontró el classification report.")


def show_feature_importance_chart(model, features):
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'Importancia': importances}, index=features)
    imp_df = imp_df.sort_values('Importancia', ascending=False)
    st.subheader("Importancia de las variables")
    st.bar_chart(imp_df)
    st.markdown("""
    **¿Qué muestra este gráfico?**
    Este gráfico indica la relevancia de cada variable en la predicción del modelo. Las variables a la izquierda tienen mayor influencia en la decisión final del modelo Random Forest.
    """)


def show_roc_curve_section(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    st.subheader("Curva ROC")
    st.line_chart(roc_df.set_index("FPR"))
    st.caption("Eje X: False Positive Rate (FPR) | Eje Y: True Positive Rate (TPR)")
    st.metric(label="AUC", value=f"{auc:.2f}")
    st.markdown("""
    **¿Qué es la curva ROC?**
    La curva ROC (Receiver Operating Characteristic) es una herramienta gráfica que permite evaluar el desempeño de un modelo de clasificación binaria, mostrando la relación entre la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR) a diferentes umbrales de decisión.

    **¿Qué muestra esta gráfica?**
    - **False Positive Rate (FPR):** Proporción de negativos reales que son clasificados incorrectamente como positivos (falsos positivos / total de negativos reales).
    - **True Positive Rate (TPR):** Proporción de positivos reales que son correctamente identificados (verdaderos positivos / total de positivos reales).
    - **AUC (Área Bajo la Curva):** Mide la capacidad del modelo para distinguir entre clases. Un valor de 1.0 indica un modelo perfecto; 0.5 indica un modelo sin capacidad de discriminación.
    """)

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    model = load_model("models/rf_model.joblib") 