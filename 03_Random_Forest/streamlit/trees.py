import streamlit as st
import os

def show_tree_images(tree_dir, n=9):
    st.markdown("""
    A continuación se muestran los primeros 9 árboles individuales que componen el modelo Random Forest. Cada árbol es una representación gráfica de cómo el modelo toma decisiones a partir de las variables de entrada. Estos árboles ayudan a entender cómo el modelo combina reglas simples para realizar predicciones complejas.
    """)
    tree_files = [f"tree_{i}.png" for i in range(1, n+1)]
    for row in range(3):
        cols = st.columns(3)
        for col, idx in zip(cols, range(row*3, (row+1)*3)):
            tree_path = os.path.join(tree_dir, tree_files[idx])
            if os.path.exists(tree_path):
                col.image(tree_path, caption=f"Árbol {idx+1}", use_container_width=True)
            else:
                col.info(f"No se encontró el árbol {idx+1}") 