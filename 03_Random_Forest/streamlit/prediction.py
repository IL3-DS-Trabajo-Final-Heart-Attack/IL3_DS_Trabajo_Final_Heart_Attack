import streamlit as st
import pandas as pd
import os
from prediction_utils import load_model, predict

# Obtener la ruta del directorio actual del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def show_patient_prediction_section():
    col1, col2 = st.columns([1.2, 2])
    with col1:
        st.write("### Ingreses los datos del paciente")
        with st.form("patient_form"):
            age = st.number_input("Edad", min_value=18, max_value=100, value=55, help="Edad del paciente en años")
            gender = st.selectbox("Género", options=[("Hombre", 1), ("Mujer", 0)], format_func=lambda x: x[0], help="Género del paciente")[1]
            hr = st.number_input("Frecuencia cardíaca (latidos/min)", min_value=40, max_value=180, value=75, help="Frecuencia cardíaca en reposo")
            sbp = st.number_input("Presión sistólica (mmHg)", min_value=80, max_value=220, value=130, help="Presión arterial sistólica")
            dbp = st.number_input("Presión diastólica (mmHg)", min_value=50, max_value=120, value=85, help="Presión arterial diastólica")
            bs = st.number_input("Azúcar en sangre (mg/dL)", min_value=60, max_value=400, value=95, help="Nivel de glucosa en sangre")
            ckmb = st.number_input("CK-MB (ng/mL)", min_value=0.0, max_value=50.0, value=3.5, step=0.1, help="Nivel de creatina quinasa-MB")
            trop = st.number_input("Troponina (ng/mL)", min_value=0.0, max_value=5.0, value=0.01, step=0.001, format="%.3f", help="Nivel de troponina cardíaca")
            submitted = st.form_submit_button("Predecir")

    with col2:
        st.write("### Valores de referencia")
        ref_data = [
            ["HR (reposo)", "bpm", "60–100", "-", "< 60 (bradicardia), > 100 (taquicardia)"],
            ["SBP", "mm Hg", "< 120", "120–129 (elevado)", "≥ 130 etapa 1–2; ≥ 180 crisis"],
            ["DBP", "mm Hg", "< 80", "< 60 hipotensión; 80–89 etapa 1", "≥ 90 etapa 2; ≥ 120 crisis"],
            ["Glucemia ayuno", "mg/dL", "70–99", "< 70 hipoglucemia; 100–125 prediabetes", "≥ 126 diabetes"],
            ["CK‑MB", "ng/mL", "< 5", "5–10 elevado", ">10 infarto extenso"],
            ["Troponina T", "ng/mL", "< 0.01", "0.01–0.13 elevado", "≥ 0.14"],
            ["hs‑Troponin T", "ng/mL", "< 0.014", "0.014–0.052 elevado", "≥ 0.053"],
        ]
        ref_df = pd.DataFrame(ref_data, columns=["Variable", "Unidad", "Normal", "Elevado / Bajo", "Crítico / IAM / Crisis"])
        st.dataframe(ref_df, hide_index=True, use_container_width=True)

        if 'submitted' in locals() and submitted:
            valores_paciente = {
                "age": age,
                "gender": gender,
                "hr": hr,
                "sbp": sbp,
                "dbp": dbp,
                "bs": bs,
                "ckmb": ckmb,
                "trop": trop
            }
            nuevo_paciente = pd.DataFrame([valores_paciente])
            model = load_model(os.path.join(SCRIPT_DIR, "models/rf_model.joblib"))
            pred, proba = predict(model, nuevo_paciente)

            st.write("### Resultado de la predicción")
            st.write("**¿Riesgo de infarto?:**", "Sí" if pred == 1 else "No")
            st.write(f"**Probabilidad:** No: {proba[0]:.2%}, Sí: {proba[1]:.2%}")
            
            # Mensaje de advertencia importante
            st.warning("""
            **⚠️ ADVERTENCIA IMPORTANTE**
            
            Esta predicción se basa únicamente en las variables biométricas y bioquímicas ingresadas (edad, género, frecuencia cardíaca, presión arterial, glucosa, CK-MB y troponina). 
            
            **IMPORTANTE CONSIDERAR:**
            - El diagnóstico de infarto agudo del miocardio requiere evaluación médica completa
            - Se deben considerar síntomas clínicos, historia médica, electrocardiograma, y otros exámenes
            - Factores como dolor torácico, disnea, sudoración, náuseas, y otros síntomas son cruciales
            - El contexto clínico completo es fundamental para un diagnóstico preciso
            
            **ESTE MODELO:**
            - Fue desarrollado únicamente con fines educativos y de investigación
            - NO pretende suplantar o reemplazar el criterio médico profesional
            - Los resultados deben interpretarse con precaución
            
            **En caso de síntomas sospechosos de infarto, busque atención médica inmediata.**
            """)
 
        else:
            st.write("### Resultado de la predicción")
            st.info("Complete el formulario y presione 'Predecir' para ver el resultado.") 