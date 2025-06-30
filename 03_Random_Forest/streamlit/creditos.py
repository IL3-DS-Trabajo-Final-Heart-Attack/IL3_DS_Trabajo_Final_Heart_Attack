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
    - Luis Antelo Laguna
    """)
    
    # Bloque destacado para el repositorio
    st.markdown(
        """
        <div style=' padding: 20px; border-radius: 10px; border: 2px solid #24292f; text-align: center; margin-bottom: 20px;'>
            <h3 style='margin-bottom: 10px;'>Repositorio del proyecto en GitHub</h3>
            <a href='https://github.com/IL3-DS-Trabajo-Final-Heart-Attack/IL3_DS_Trabajo_Final_Heart_Attack' target='_blank' style='display: inline-block; background-color: #24292f; color: #fff; padding: 12px 28px; border-radius: 6px; font-size: 18px; font-weight: bold; text-decoration: none; margin-top: 10px;'>
                ⭐ Visítanos en GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("""
    ---
    
    :page_facing_up: Este proyecto está licenciado bajo los términos de la licencia MIT. Consulta el archivo LICENSE para más detalles.
    """) 