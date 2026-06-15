import os
import pickle
import pandas as pd
import streamlit as st

#Configuración de la interfaz
st.set_page_config(
    page_title="Clasificador Fotométrico TFG",
    page_icon="🌌",
    layout="centered"
)

#Título principal y descripción
st.title("Clasificador Fotométrico de Cuerpos Celestes")
st.write("""
Este sistema utiliza los modelos de Inteligencia Artificial optimizados para determinar si un objeto celeste 
es una **Galaxia**, un **Cuásar (QSO)** o una **Estrella** a partir de sus magnitudes fotométricas (SDSS).
""")

st.markdown("---")

st.sidebar.header("Configuración del Modelo")
modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona el modelo:",
    ["Random Forest (Bagging)", "XGBoost (Boosting)"]
)

st.sidebar.markdown("---")
st.sidebar.write("**Estado del Sistema:**")
st.sidebar.success("Modelos cargados correctamente.")


st.subheader("Entrada de Magnitudes Fotométricas")
st.write("Introduce los valores de las bandas de luz del objeto celeste (u,g,r,i,d):")

col1, col2 = st.columns(2)

with col1:
    u = st.number_input("Filtro Ultravioleta (u):", value=23.39351, step=0.01, format="%.4f")
    g = st.number_input("Filtro Verde (g):", value=22.3446, step=0.01, format="%.4f")
    r = st.number_input("Filtro Rojo (r):", value=20.29405, step=0.01, format="%.4f")

with col2:
    i = st.number_input("Filtro Infrarrojo Cercano (i):", value=19.42002, step=0.01, format="%.4f")
    z = st.number_input("Filtro Infrarrojo Extremo (z):", value=18.96489, step=0.01, format="%.4f")


u_g = u - g
g_r = g - r
r_i = r - i
i_z = i - z

#Estructura exacta que esperan los modelos
datos_entrada = {
    'u': u, 'g': g, 'r': r, 'i': i, 'z': z,
    'u_g': u_g, 'g_r': g_r, 'r_i': r_i, 'i_z': i_z
}
df_input = pd.DataFrame([datos_entrada])

st.markdown("---")

if st.button("Clasificar Objeto Celeste", type="primary"):
    
    ruta_models = 'models'
    
    try:
        if "Random Forest" in modelo_seleccionado:
            #Cargar y predecir con Random Forest
            with open(os.path.join(ruta_models, 'best_random_forest.pkl'), 'rb') as f:
                model = pickle.load(f)
            
            #Asegurar el orden de las columnas que guardó el entrenamiento
            df_input = df_input[model.feature_names_in_]
            prediccion_final = model.predict(df_input)[0]
            
            #Traducir la respuesta para la interfaz
            dicc_traduccion = {'GALAXY': 'GALAXIA', 'QSO': 'CUÁSAR', 'STAR': 'ESTRELLA'}
            resultado_texto = dicc_traduccion.get(prediccion_final, prediccion_final)
            
        else:
            #Cargar y predecir con XGBoost
            with open(os.path.join(ruta_models, 'best_xgboost.pkl'), 'rb') as f:
                model = pickle.load(f)
            with open(os.path.join(ruta_models, 'label_encoder.pkl'), 'rb') as f:
                le = pickle.load(f)
                
            df_input = df_input[model.feature_names_in_]
            pred_num = model.predict(df_input)[0]  # Devuelve número (0, 1, 2)
            
            #Traducir usando el LabelEncoder por la determinación numerica (0, 1, 2)
            pred_texto_original = le.inverse_transform([pred_num])[0]
            dicc_traduccion = {'GALAXY': 'GALAXIA', 'QSO': 'CUÁSAR', 'STAR': 'ESTRELLA'}
            resultado_texto = dicc_traduccion.get(pred_texto_original, pred_texto_original)

        #Mostrar el resultado final
        st.subheader("Resultado de la Clasificación:")
        
        if resultado_texto == "GALAXIA":
            st.info(f"El modelo **{modelo_seleccionado.split(' ')[0]}** ha clasificado el objeto como una: **{resultado_texto}**")
        elif resultado_texto == "CUÁSAR":
            st.error(f"El modelo **{modelo_seleccionado.split(' ')[0]}** ha clasificado el objeto como un: **{resultado_texto}**")
        else:
            st.success(f"El modelo **{modelo_seleccionado.split(' ')[0]}** ha clasificado el objeto como una: **{resultado_texto}**")
            
    except Exception as e:
        st.error(f"No se pudo realizar la predicción. Error: {e}")