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

#CONFIGURACIÓN
st.sidebar.header("Configuración del Modelo")
modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona el modelo:",
    ["Random Forest (Bagging)", "XGBoost (Boosting)"]
)

#Implementación del umbral de confianza
st.sidebar.markdown("---")
st.sidebar.header("Filtro de Indeterminación")
umbral_confianza = st.sidebar.slider(
    "Umbral de confianza mínimo (%):",
    min_value=50,
    max_value=100,
    value=75,
    step=5,
    help="Si la probabilidad del modelo es menor a este porcentaje, el objeto se clasificará como INDETERMINADO."
) / 100.0

st.sidebar.markdown("---")
st.sidebar.write("**Estado del Sistema:**")
st.sidebar.success("Modelos cargados correctamente.")


#ENTRADA DE DATOS
st.subheader("Entrada de Magnitudes Fotométricas")
st.write("Introduce los valores de las bandas de luz del objeto celeste (u,g,r,i,z):")

col1, col2 = st.columns(2)

with col1:
    u = st.number_input("Filtro Ultravioleta (u):", value=23.39351, step=0.01, format="%.4f")
    g = st.number_input("Filtro Verde (g):", value=22.3446, step=0.01, format="%.4f")
    r = st.number_input("Filtro Rojo (r):", value=20.29405, step=0.01, format="%.4f")

with col2:
    i = st.number_input("Filtro Infrarrojo Cercano (i):", value=19.42002, step=0.01, format="%.4f")
    z = st.number_input("Filtro Infrarrojo Extremo (z):", value=18.96489, step=0.01, format="%.4f")

#Ingeniería de características
u_g = u - g
g_r = g - r
r_i = r - i
i_z = i - z

datos_entrada = {
    'u': u, 'g': g, 'r': r, 'i': i, 'z': z,
    'u_g': u_g, 'g_r': g_r, 'r_i': r_i, 'i_z': i_z
}
df_input = pd.DataFrame([datos_entrada])

st.markdown("---")

#PROCESAMIENTO DE LA PREDICCIÓN
if st.button("Clasificar Objeto Celeste", type="primary"):
    
    ruta_models = 'models'
    dicc_traduccion = {'GALAXY': 'GALAXIA', 'QSO': 'CUÁSAR', 'STAR': 'ESTRELLA'}
    
    try:
        if "Random Forest" in modelo_seleccionado:
            #Carga de Random Forest
            with open(os.path.join(ruta_models, 'best_random_forest.pkl'), 'rb') as f:
                model = pickle.load(f)
            
            df_input = df_input[model.feature_names_in_]
            
            #Extraemos el array de probabilidades
            probabilidades = model.predict_proba(df_input)[0]
            max_proba = probabilidades.max()  # Mayor probabilidad detectada
            pred_idx = probabilidades.argmax()  # Índice de la clase ganadora
            pred_texto_original = model.classes_[pred_idx]
            
        else:
            #Carga de XGBoost
            with open(os.path.join(ruta_models, 'best_xgboost.pkl'), 'rb') as f:
                model = pickle.load(f)
            with open(os.path.join(ruta_models, 'label_encoder.pkl'), 'rb') as f:
                le = pickle.load(f)
                
            df_input = df_input[model.feature_names_in_]
            
            #Extraemos el array de probabilidades
            probabilidades = model.predict_proba(df_input)[0]
            max_proba = probabilidades.max()
            pred_num = probabilidades.argmax()
            pred_texto_original = le.inverse_transform([pred_num])[0]

        #Evaluación del filtro de confianza estricto (RF-4)
        if max_proba < umbral_confianza:
            resultado_texto = "INDETERMINADO"
        else:
            resultado_texto = dicc_traduccion.get(pred_texto_original, pred_texto_original)

        #SALIDA DE RESULTADOS EN LA INTERFAZ
        st.subheader("Resultado de la Clasificación:")
        nombre_modelo_corto = modelo_seleccionado.split(' ')[0]
        
        if resultado_texto == "INDETERMINADO":
            st.warning(
                f"**Objeto Indeterminado**: El modelo **{nombre_modelo_corto}** estimó una certeza "
                f"del **{max_proba * 100:.2f}%**, valor que no supera el umbral de seguridad exigido "
                f"del **{umbral_confianza * 100:.0f}%**."
            )
        elif resultado_texto == "GALAXIA":
            st.info(f"El modelo **{nombre_modelo_corto}** ha clasificado el objeto como una: **{resultado_texto}** *(Confianza: {max_proba * 100:.2f}%)*")
        elif resultado_texto == "CUÁSAR":
            st.error(f"El modelo **{nombre_modelo_corto}** ha clasificado el objeto como un: **{resultado_texto}** *(Confianza: {max_proba * 100:.2f}%)*")
        else:
            st.success(f"El modelo **{nombre_modelo_corto}** ha clasificado el objeto como una: **{resultado_texto}** *(Confianza: {max_proba * 100:.2f}%)*")
            
    except Exception as e:
        st.error(f"No se pudo realizar la predicción de ingeniería. Error técnico: {e}")