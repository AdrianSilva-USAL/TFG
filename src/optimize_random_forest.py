import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, classification_report

def optimizar_random_forest():
    """Optimizacion del modelo Random Forest ccon modificaciones en los 3 parametros principales, nº arboles, profundidad y muestras minimas para la división, utilizando la validación cruzada de 5 bloques"""

    print("Iniciando optimización para Random Forest con validacion cruzada de 5 bloques...")
    
    #Cargar los datos procesados de desarrollo.csv
    ruta_desarrollo = os.path.join('data', 'processed', 'desarrollo.csv')
    if not os.path.exists(ruta_desarrollo):
        raise FileNotFoundError("No se encuentra desarrollo.csv. Ejecutar primero src/data_processing.py")
    
    df = pd.read_csv(ruta_desarrollo)
    X = df.drop(columns=['class'])
    y = df['class']
    
    #Examen de validación 10.000 filas fijas
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )
    
    #Rejilla de 3 opciones por hiperparametro (27 combinaciones)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    
    combinaciones_totales = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])
    print(f"Evaluando las  {combinaciones_totales} combinaciones de hiperparámetros.")
    print(f"Ejecutando Validación Cruzada con 5 bloques. {combinaciones_totales * 5} entrenamientos en total...")
    
    #Configurar la búsqueda en rejilla
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    #Proceso de optimización del modelo con la rejilla de combinaciones
    grid_search.fit(X_train_full, y_train_full)
    
    print("\nOptimización completada")
    print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
    
    #Guarda el mejor modelo (mejor combinacón)
    mejor_modelo = grid_search.best_estimator_
    
    print("\nTop 5 configuraciones para análisis...")
    resultados_cv = pd.DataFrame(grid_search.cv_results_)

    #Ordenar por el ranking y coger los indices de los 5 mejores
    indices_top_5 = resultados_cv.sort_values(by='rank_test_score').index[:5]
    
    datos_top_5 = []
    
    for i, idx in enumerate(indices_top_5, 1):
        params = resultados_cv.loc[idx, 'params']
        print(f"Evaluando Top {i}: {params}")
        
        #Entrenar un modelo rápido con 70.000 filas con la configuración específica
        model_temp = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model_temp.fit(X_train_full, y_train_full)
        
        #Validar con las 10.000 filas fijas de examen
        preds_temp = model_temp.predict(X_val)
        f1_cuerpos = f1_score(y_val, preds_temp, average=None, labels=['GALAXY', 'QSO', 'STAR'])
        
        #Etiqueta para el eje X de la gráfica
        nombre_x = f"Top {i}\nArboles:{params['n_estimators']}\nProfundidad:{params['max_depth']}\nMuestras/Div:{params['min_samples_split']}"
        
        datos_top_5.append({
            'Configuracion': nombre_x,
            'GALAXIA': f1_cuerpos[0] * 100,
            'CUÁSAR': f1_cuerpos[1] * 100,
            'ESTRELLA': f1_cuerpos[2] * 100
        })
    
    df_top_5 = pd.DataFrame(datos_top_5)
    
    #Generar la gráfica de barras agrupadas desglosada
    ruta_figuras = os.path.join('reports', 'figures')
    os.makedirs(ruta_figuras, exist_ok=True)
    
    plt.figure(figsize=(11, 6))
    ax = df_top_5.set_index('Configuracion').plot(
        kind='bar', 
        color=['#3498db', '#e74c3c', '#f1c40f'], 
        edgecolor='black',
        alpha=0.85,
        width=0.75,
        figsize=(11, 6),
        zorder=2
    )
    
    plt.title('COMPARATIVA DEL TOP 5 DE MODELOS RANDOM FOREST', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Configuración de Hiperparámetros', fontsize=10, labelpad=10)
    plt.ylabel('F1-Score Final (%)', fontsize=10)
    plt.ylim(75, 105)
    plt.grid(True, linestyle='--', alpha=0.4, zorder=1)
    plt.xticks(rotation=0, fontsize=9)
    plt.legend(title="Cuerpos Celestes", loc='upper right', framealpha=0.9)
    
    #Añadir el valor exacto a cada barra de la gráfica
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.5), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points', 
                    fontsize=8, 
                    fontweight='bold')

    ruta_grafica_top = os.path.join(ruta_figuras, 'top_5_modelos_RF.png')
    plt.savefig(ruta_grafica_top, dpi=300, bbox_inches='tight')
    plt.close()
    
    #Guardar datos del Top 5 en CSV
    ruta_csv_top = os.path.join(ruta_figuras, 'valores_top_5_RF.csv')
    df_top_5.to_csv(ruta_csv_top, index=False, sep=';')
    
    #Guardar el mejor modelo
    ruta_modelos = 'models'
    os.makedirs(ruta_modelos, exist_ok=True)
    ruta_archivo_modelo = os.path.join(ruta_modelos, 'best_random_forest.pkl')
    
    with open(ruta_archivo_modelo, 'wb') as f:
        pickle.dump(mejor_modelo, f)
        
    print(f"\nProceso terminado")
    print(f"Gráfica TOP 5 guardada en: '{ruta_grafica_top}'")
    print(f"Valores del Top 5 guardada en: '{ruta_csv_top}'")
    print(f"Archivo del modelo ganador guardado en: '{ruta_archivo_modelo}'")

if __name__ == "__main__":
    try:
        optimizar_random_forest()
    except Exception as e:
        print(f"Error durante la optimización: {e}")