import os
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def comparativa_caracteristicas():
    """Compara el rendimiento de XGBoost usando solo bandas vs bandas + índices de color"""
    print("Iniciando comparación de ingeniería de características...")
    
    ruta_desarrollo = os.path.join('data', 'processed', 'desarrollo.csv')
    if not os.path.exists(ruta_desarrollo):
        raise FileNotFoundError("No se encuentra desarrollo.csv. Ejecutar primero src/data_processing.py")
    
    df = pd.read_csv(ruta_desarrollo)
    
    if 'u_g' not in df.columns and 'u-g' not in df.columns:
        print("Generando índices de color...")
        df['u_g'] = df['u'] - df['g']
        df['g_r'] = df['g'] - df['r']
        df['r_i'] = df['r'] - df['i']
        df['i_z'] = df['i'] - df['z']

    X = df.drop(columns=['class'])
    y = df['class']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    bandas_puras = ['u', 'g', 'r', 'i', 'z']
    
    tamanos_entrenamiento = [5000, 10000, 20000, 40000, 70000]
    
    raw_galaxy, raw_qso, raw_star = [], [], []
    
    eng_galaxy, eng_qso, eng_star = [], [], []
    
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y_encoded, test_size=10000, random_state=42, stratify=y_encoded
    )
    
    for tamano in tamanos_entrenamiento:
        print(f"Evaluando tramo de {tamano} filas con XGBoost...")
        X_sub = X_train_full.head(tamano)
        y_sub = y_train_full[:tamano]
        
        modelo_raw = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
        modelo_raw.fit(X_sub[bandas_puras], y_sub)
        preds_raw = modelo_raw.predict(X_val[bandas_puras])
        f1_raw = f1_score(y_val, preds_raw, average=None, labels=[0, 1, 2])
        
        raw_galaxy.append(f1_raw[0] * 100)
        raw_qso.append(f1_raw[1] * 100)
        raw_star.append(f1_raw[2] * 100)
        
        modelo_eng = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
        modelo_eng.fit(X_sub, y_sub)
        preds_eng = modelo_eng.predict(X_val)
        f1_eng = f1_score(y_val, preds_eng, average=None, labels=[0, 1, 2])
        
        eng_galaxy.append(f1_eng[0] * 100)
        eng_qso.append(f1_eng[1] * 100)
        eng_star.append(f1_eng[2] * 100)

    ruta_figuras = os.path.join('reports', 'figures')
    os.makedirs(ruta_figuras, exist_ok=True)
    
    print("Generando gráficas comparativas para XGBoost...")
    
    config_graficas = [
        ('GALAXIAS', raw_galaxy, eng_galaxy, '#2ecc71', 'comparativa_galaxias_XGB.png'),
        ('CUÁSARES', raw_qso, eng_qso, '#e67e22', 'comparativa_cuasares_XGB.png'),
        ('ESTRELLAS', raw_star, eng_star, '#9b59b6', 'comparativa_estrellas_XGB.png')
    ]
    
    for titulo, datos_raw, datos_eng, color_eng, nombre_archivo in config_graficas:
        plt.figure(figsize=(7.5, 5))
        
        plt.plot(tamanos_entrenamiento, datos_raw, color='#7f8c8d', linestyle='--', linewidth=1.8, label='Solo Bandas (u, g, r, i, z)', alpha=0.8, zorder=2)
        plt.scatter(tamanos_entrenamiento, datos_raw, color='#7f8c8d', s=80, edgecolors='black', zorder=3)
        
        plt.plot(tamanos_entrenamiento, datos_eng, color=color_eng, linestyle='-', linewidth=2.2, label='Bandas + Índices de Color', zorder=4)
        plt.scatter(tamanos_entrenamiento, datos_eng, color=color_eng, s=120, edgecolors='black', zorder=5)
        
        plt.title(f'INGENIERÍA DE DATOS (XGBOOST): {titulo}', fontsize=11, fontweight='bold')
        plt.xlabel('Filas', fontsize=10)
        plt.ylabel('F1-Score (%)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
        plt.xlim(2000, 73000)
        plt.ylim(70, 100)
        plt.legend(loc='lower right', fontsize=9, frameon=True, shadow=True)
        
        plt.savefig(os.path.join(ruta_figuras, nombre_archivo), dpi=300, bbox_inches='tight')
        plt.close()

    print("Generando CSV de la comparativa XGBoost...")
    df_comparativo = pd.DataFrame({
        'Filas_Entrenamiento': tamanos_entrenamiento,
        'GALAXY_Solo_Bandas': raw_galaxy,
        'GALAXY_Con_Indices': eng_galaxy,
        'QSO_Solo_Bandas': raw_qso,
        'QSO_Con_Indices': eng_qso,
        'STAR_Solo_Bandas': raw_star,
        'STAR_Con_Indices': eng_star
    })
    
    df_comparativo.to_csv(os.path.join(ruta_figuras, 'valores_comparativa_features_XGB.csv'), index=False, sep=';')
    print("Proceso terminado")

if __name__ == "__main__":
    try:
        comparativa_caracteristicas()
    except Exception as e:
        print(f"Error en la ejecución: {e}")