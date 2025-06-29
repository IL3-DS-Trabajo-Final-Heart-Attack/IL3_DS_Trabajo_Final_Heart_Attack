import streamlit as st
import pandas as pd
from prediction_utils import load_model, predict

def show_patient_prediction_section():
    col1, col2 = st.columns([1.2, 2])
    with col1:
        st.write("### Ingrese los datos del paciente")
        with st.form("patient_form"):
            age = st.number_input("Edad", min_value=1, max_value=120, value=60)
            gender = st.selectbox("Género", options=[("Hombre", 1), ("Mujer", 0)], format_func=lambda x: x[0])[1]
            hr = st.number_input("Frecuencia cardíaca (latidos/min)", min_value=30, max_value=200, value=80)
            sbp = st.number_input("Presión sistólica (mmHg)", min_value=50, max_value=250, value=120)
            dbp = st.number_input("Presión diastólica (mmHg)", min_value=30, max_value=150, value=92)
            bs = st.number_input("Azúcar en sangre (mg/dL)", min_value=40, max_value=500, value=100)
            ckmb = st.number_input("CK-MB (ng/mL)", min_value=0.0, max_value=100.0, value=12.0)
            trop = st.number_input("Troponina (ng/mL)", min_value=0.0, max_value=10.0, value=0.014, format="%f")
            submitted = st.form_submit_button("Predecir")

    with col2:
        st.write("### Valores de referencia")
        ref_data = [
            ["HR (reposo)", "bpm", "60–100", "-", "< 60 (bradicardia), > 100 (taquicardia)"],
            ["SBP", "mm Hg", "< 120", "120–129 (elevado)", "≥ 130 etapa 1–2; ≥ 180 crisis"],
            ["DBP", "mm Hg", "< 80", "< 60 hipotensión; 80–89 etapa 1", "≥ 90 etapa 2; ≥ 120 crisis"],
            ["Glucemia ayuno", "mg/dL", "70–99", "< 70 hipoglucemia; 100–125 prediabetes", "≥ 126 diabetes"],
            ["CK‑MB", "ng/mL", "< 5", "5–10 elevado", ">10 infarto extenso"],
            ["Troponina T", "ng/mL", "< 0.01", "0.01–0.13 elevado", "≥ 0.14"],
            ["hs‑Troponin T", "ng/mL", "< 0.014", "0.014–0.052 elevado", "≥ 0.053"],
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
            model = load_model("../models/rf_model.joblib")
            pred, proba = predict(model, nuevo_paciente)

            st.write("### Resultado de la predicción")
            st.write("**¿Riesgo de infarto?:**", "Sí" if pred == 1 else "No")
            st.write(f"**Probabilidad:** No: {proba[0]:.2%}, Sí: {proba[1]:.2%}")
 
        else:
            st.write("### Resultado de la predicción")
            st.info("Complete el formulario y presione 'Predecir' para ver el resultado.") 