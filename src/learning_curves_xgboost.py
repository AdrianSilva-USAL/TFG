import os
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def generar_curvas_aprendizaje_xgboost():
    """Crea las curvas con parámetros de fábrica  y exporta graficas y CSV de resultados"""
    print("Iniciando simulación de curvas de aprendizaje...")
    
    #Cargar los datos procesados de desarrollo.csv
    ruta_desarrollo = os.path.join('data', 'processed', 'desarrollo.csv')
    if not os.path.exists(ruta_desarrollo):
        raise FileNotFoundError("No se encuentra desarrollo.csv. Ejecutar primero src/data_processing.py")
    
    df = pd.read_csv(ruta_desarrollo)
    X = df.drop(columns=['class'])
    y = df['class']
    
    #Traducir texto ('GALAXY', 'QSO', 'STAR') a números (0, 1, 2) necesario por XGBoost
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    #Escala exponencial de tamaños de aprendizaje
    tamanos_entrenamiento = [5000, 10000, 20000, 40000, 70000]
    
    historial_galaxy = []
    historial_qso = []
    historial_star = []
    
    #Examen de validación 10.000 filas fijas
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y_encoded, test_size=10000, random_state=42, stratify=y_encoded
    )
    
    #Bucle de entrenamiento
    for tamano in tamanos_entrenamiento:
        print(f"Evaluando tramo de {tamano} filas con XGBoost...")
        X_sub = X_train_full.head(tamano)
        y_sub = y_train_full[:tamano]
        
        #Modelo XGBoost de fábrica 
        modelo = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
        modelo.fit(X_sub, y_sub)
        
        preds = modelo.predict(X_val)
        f1_cada_cuerpo = f1_score(y_val, preds, average=None, labels=[0, 1, 2])
        
        #Cálculo de porcentajes de f1 de cada cuerpo
        historial_galaxy.append(f1_cada_cuerpo[0] * 100)
        historial_qso.append(f1_cada_cuerpo[1] * 100)
        historial_star.append(f1_cada_cuerpo[2] * 100)

    #Crear carpeta de reports (si no esta creada)
    ruta_figuras = os.path.join('reports', 'figures')
    os.makedirs(ruta_figuras, exist_ok=True)
    
    print("Generando graficas XGBoost...")

    #GALAXIAS
    plt.figure(figsize=(7, 4.5))
    plt.plot(tamanos_entrenamiento, historial_galaxy, color='#2980b9', linestyle='-', linewidth=2, alpha=0.7, zorder=2)
    plt.scatter(tamanos_entrenamiento, historial_galaxy, color='#2980b9', s=120, edgecolors='black', zorder=3)
    plt.title('GALAXIAS (XGBoost de fábrica)', fontsize=11, fontweight='bold')
    plt.xlabel('Filas', fontsize=10)
    plt.ylabel('F1-Score (%)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    plt.xlim(2000, 73000)
    plt.ylim(60, 100)
    plt.savefig(os.path.join(ruta_figuras, 'curva_aprendizaje_galaxias_XGB.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #CUÁSARES
    plt.figure(figsize=(7, 4.5))
    plt.plot(tamanos_entrenamiento, historial_qso, color='#c0392b', linestyle='-', linewidth=2, alpha=0.7, zorder=2)
    plt.scatter(tamanos_entrenamiento, historial_qso, color='#c0392b', s=120, edgecolors='black', zorder=3)
    plt.title('CUÁSARES (XGBoost de fábrica)', fontsize=11, fontweight='bold')
    plt.xlabel('Filas', fontsize=10)
    plt.ylabel('F1-Score (%)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    plt.xlim(2000, 73000)
    plt.ylim(60, 100)
    plt.savefig(os.path.join(ruta_figuras, 'curva_aprendizaje_cuasares_XGB.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #ESTRELLAS
    plt.figure(figsize=(7, 4.5))
    plt.plot(tamanos_entrenamiento, historial_star, color='#d4ac0d', linestyle='-', linewidth=2, alpha=0.7, zorder=2)
    plt.scatter(tamanos_entrenamiento, historial_star, color='#d4ac0d', s=120, edgecolors='black', zorder=3)
    plt.title('ESTRELLAS (XGBoost de fábrica)', fontsize=11, fontweight='bold')
    plt.xlabel('Filas', fontsize=10)
    plt.ylabel('F1-Score (%)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    plt.xlim(2000, 73000)
    plt.ylim(60, 100)
    plt.savefig(os.path.join(ruta_figuras, 'curva_aprendizaje_estrellas_XGB.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #Creación y exportación del CSV de resultados
    print("Generando CSV de resultados...")
    df_metricas = pd.DataFrame({
        'Filas_Entrenamiento': tamanos_entrenamiento,
        'GALAXY_F1_Porcentaje': historial_galaxy,
        'QSO_F1_Porcentaje': historial_qso,
        'STAR_F1_Porcentaje': historial_star
    })
    
    ruta_csv_valores = os.path.join(ruta_figuras, 'valores_curva_aprendizaje_XGB.csv')
    df_metricas.to_csv(ruta_csv_valores, index=False, sep=';')

if __name__ == "__main__":
    try:
        generar_curvas_aprendizaje_xgboost()
    except Exception as e:
        print(f"Error en la ejecución de XGBoost: {e}")