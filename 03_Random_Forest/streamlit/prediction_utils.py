import streamlit as st
import joblib

def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"No se encontró el modelo en {path}")
        st.stop()

def predict(model, input_df):
    pred = model.predict(input_df)
    proba = model.predict_proba(input_df)
    return pred[0], proba[0] 