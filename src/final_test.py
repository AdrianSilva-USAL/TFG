import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

def comparar_modelos():
    """Se compara la eficiencia de las mejores versiones de cada modelo (Random Forest y XGBoost) con el 20% de lineas del test final (datos que nunca se han utilizado para el aprendizaje)"""
    print(f"Iniciando la evaluación comparativa final sobre el 20% de los datos de test...")
    
    #Cargar el datasheet de test final (El 20% separado)
    ruta_test = os.path.join('data', 'processed', 'test_final.csv')
    if not os.path.exists(ruta_test):
        raise FileNotFoundError("No se encuentra test_final.csv. Ejecuta primero src/data_processing.py")
        
    df_test = pd.read_csv(ruta_test)
    X_test = df_test.drop(columns=['class'])
    y_test = df_test['class']  # Contiene las etiquetas en texto ('GALAXY', 'QSO', 'STAR')
    
    #Cargar los modelos entrenados y el codificador
    ruta_models = 'models'
    
    with open(os.path.join(ruta_models, 'best_random_forest.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
        
    with open(os.path.join(ruta_models, 'best_xgboost.pkl'), 'rb') as f:
        xgb_model = pickle.load(f)
        
    with open(os.path.join(ruta_models, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
        
    # Ejecutar predicciones sobre el 20% de datos nuevos
    print("Evaluando Random Forest óptimo...")
    preds_rf = rf_model.predict(X_test)
    
    print("Evaluando XGBoost óptimo...")
    preds_xgb_num = xgb_model.predict(X_test)
    #Traducción de los números de XGBoost (0,1,2) a texto original para analizar
    preds_xgb = le.inverse_transform(preds_xgb_num)
    
    #Calcular métricas de cada cuerpo (F1-Score)
    cuerpos_orden = ['GALAXY', 'QSO', 'STAR']
    f1_rf = f1_score(y_test, preds_rf, average=None, labels=cuerpos_orden) * 100
    f1_xgb = f1_score(y_test, preds_xgb, average=None, labels=cuerpos_orden) * 100
    
    #Calcular las medias Macro generales
    macro_rf = f1_score(y_test, preds_rf, average='macro') * 100
    macro_xgb = f1_score(y_test, preds_xgb, average='macro') * 100
    
    #Estructurar los resultados para realizar la grafica
    datos_comparativa = {
        'Cuerpo Celeste': ['GALAXIA', 'CUÁSAR', 'ESTRELLA', 'MEDIA GLOBAL (MACRO)'],
        'Random Forest': [f1_rf[0], f1_rf[1], f1_rf[2], macro_rf],
        'XGBoost': [f1_xgb[0], f1_xgb[1], f1_xgb[2], macro_xgb]
    }
    df_comp = pd.DataFrame(datos_comparativa).set_index('Cuerpo Celeste')
    
    ruta_figuras = os.path.join('reports', 'figures')
    os.makedirs(ruta_figuras, exist_ok=True)
    
    #Grafica de la comparación
    ax = df_comp.plot(
        kind='bar',
        color=['#2ecc71', "#0077ff"],
        edgecolor='black',
        alpha=0.9,
        width=0.7,
        figsize=(11, 6),
        zorder=2
    )
    
    plt.title('EVALUACIÓN DEFINITIVA: RANDOM FOREST VS XGBOOST (20% TEST)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Cuerpos Celestes y Rendimiento Global', fontsize=10, labelpad=10)
    plt.ylabel('F1-Score (%)', fontsize=10)
    plt.ylim(75, 100)
    plt.grid(True, linestyle='--', alpha=0.4, zorder=1)
    plt.xticks(rotation=0, fontsize=9)
    plt.legend(title="Algoritmos", loc='upper right', framealpha=0.9)
    
    #Colocar los valores encima de cada barra
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.3), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points', 
                    fontsize=8, 
                    fontweight='bold')
                    
    ruta_grafica_final = os.path.join(ruta_figuras, 'comparativa_final_modelos.png')
    plt.savefig(ruta_grafica_final, dpi=300, bbox_inches='tight')
    plt.close()
    
    #Guardar los resultados en un CSV
    ruta_csv_final = os.path.join(ruta_figuras, 'tabla_comparativa_final.csv')
    df_comp.to_csv(ruta_csv_final, sep=';')
    
    print(f"\nComparativa terminada")
    print(f"Gráfica final guardada en: '{ruta_grafica_final}'")
    print(f"Resultados numéricos guardados en: '{ruta_csv_final}'")

if __name__ == "__main__":
    try:
        comparar_modelos()
    except Exception as e:
        print(f"Error durante la comparativa: {e}")