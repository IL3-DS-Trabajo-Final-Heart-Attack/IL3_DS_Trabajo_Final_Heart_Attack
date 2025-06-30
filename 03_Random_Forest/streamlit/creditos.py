import streamlit as st

def show_credits_section():
    st.title("Créditos")
    st.markdown("""
    Esta aplicación web es fruto de un proyecto investigativo y educacional de ciencia de datos, desarrollado en el marco del curso **Ciencia de los Datos (Data Science), Aplicaciones en Biología y Medicina con Python y R** en el **Institut de Formació Contínua - Universitat de Barcelona**.
    
    **Autores:**
    - Oscar Alejandro Rojas Rodríguez
    - Àlex Francisco Sigüencia
    - Maya Carvalho-Evans
    - Weronika Balcerzak
    
    [Repositorio del proyecto en GitHub](https://github.com/IL3-DS-Trabajo-Final-Heart-Attack/IL3_DS_Trabajo_Final_Heart_Attack)
    
    ---
    
    :page_facing_up: Este proyecto está licenciado bajo los términos de la licencia MIT. Consulta el archivo LICENSE para más detalles.
    """) 