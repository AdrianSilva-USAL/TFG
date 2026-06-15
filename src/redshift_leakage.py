import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

def demostar_leakage():
    """Analizar con los valores base de cada modelo, la comparativa de precisión al utilizar o no la columna redshift"""
    print("🧪 Iniciando experimento de control: Comparativa Con vs Sin Redshift por modelo...")
    
    #Cargar el dataset original (raw)
    ruta_raw = os.path.join('data', 'raw', 'star_classification.csv') 
    if not os.path.exists(ruta_raw):
        ruta_raw = 'star_classification.csv'
        if not os.path.exists(ruta_raw):
            raise FileNotFoundError("No se encuentra el archivo CSV original con la columna 'redshift'.")

    df = pd.read_csv(ruta_raw)
    
    #Generacion de variables necesarias (índices de color)
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']
    
    #Listas de características
    caracteristicas_sin = ['u', 'g', 'r', 'i', 'z', 'u_g', 'g_r', 'r_i', 'i_z']
    caracteristicas_con = caracteristicas_sin + ['redshift']
    
    #Codificación de variables (necesaria para XGBoost)
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['class']) # 0=GALAXY, 1=QSO, 2=STAR
    
    #Particion de los 100.000 datos en 80% aprendizaje y 20% de test (f1-score)
    X_train_con, X_test_con, y_train, y_test = train_test_split(
        df[caracteristicas_con], y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    #Derivamos los conjuntos sin redshift a partir de la misma partición exacta
    X_train_sin = X_train_con[caracteristicas_sin]
    X_test_sin = X_test_con[caracteristicas_sin]
    
    print("Evaluando Random Forest...")
    rf_con = RandomForestClassifier(random_state=42, n_jobs=-1).fit(X_train_con, y_train)
    preds_rf_con = rf_con.predict(X_test_con)
    f1_rf_con = f1_score(y_test, preds_rf_con, average=None) * 100
    
    rf_sin = RandomForestClassifier(random_state=42, n_jobs=-1).fit(X_train_sin, y_train)
    preds_rf_sin = rf_sin.predict(X_test_sin)
    f1_rf_sin = f1_score(y_test, preds_rf_sin, average=None) * 100
    
    print("Evaluando XGBoost...")
    xgb_con = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss').fit(X_train_con, y_train)
    preds_xgb_con = xgb_con.predict(X_test_con)
    f1_xgb_con = f1_score(y_test, preds_xgb_con, average=None) * 100
    
    xgb_sin = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss').fit(X_train_sin, y_train)
    preds_xgb_sin = xgb_sin.predict(X_test_sin)
    f1_xgb_sin = f1_score(y_test, preds_xgb_sin, average=None) * 100
    
    #Creación de gráficas
    ruta_figuras = os.path.join('reports', 'figures')
    os.makedirs(ruta_figuras, exist_ok=True)
    cuerpos = ['GALAXIA', 'CUÁSAR', 'ESTRELLA']
    
    df_rf = pd.DataFrame({
        'Con Redshift (Data Leakage)': f1_rf_con,
        'Sin Redshift (Modelo Entrenado)': f1_rf_sin
    }, index=cuerpos)
    
    ax1 = df_rf.plot(
        kind='bar', 
        color=['#e74c3c', '#2ecc71'],
        edgecolor='black', alpha=0.9, width=0.6, figsize=(10, 6), zorder=2
    )
    plt.title('RANDOM FOREST: COMPARATIVA DEL USO DE REDSHIFT', fontsize=11, fontweight='bold', pad=15)
    plt.xlabel('Cuerpos Celestes', fontsize=10, labelpad=10)
    plt.ylabel('F1-Score Obtenido (%)', fontsize=10)
    plt.ylim(65, 105)
    plt.grid(True, linestyle='--', alpha=0.4, zorder=1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right', framealpha=0.9)
    
    #Añadir los valores  sobre cada barra
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.2f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 6), 
                    textcoords='offset points', fontsize=8, fontweight='bold')
    
    ruta_rf_out = os.path.join(ruta_figuras, 'comparativa_leakage_RF.png')
    plt.savefig(ruta_rf_out, dpi=300, bbox_inches='tight')
    plt.close()
    
    df_xgb = pd.DataFrame({
        'Con Redshift (Data Leakage)': f1_xgb_con,
        'Sin Redshift (Modelo Entrenado)': f1_xgb_sin
    }, index=cuerpos)
    
    ax2 = df_xgb.plot(
        kind='bar', 
        color=['#e74c3c', "#eeff00"],
        edgecolor='black', alpha=0.9, width=0.6, figsize=(10, 6), zorder=2
    )
    plt.title('XGBOOST: COMPARATIVA DEL USO DE REDSHIFT', fontsize=11, fontweight='bold', pad=15)
    plt.xlabel('Cuerpos Celestes', fontsize=10, labelpad=10)
    plt.ylabel('F1-Score Obtenido (%)', fontsize=10)
    plt.ylim(65, 105)
    plt.grid(True, linestyle='--', alpha=0.4, zorder=1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right', framealpha=0.9)
    
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.2f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 6), 
                    textcoords='offset points', fontsize=8, fontweight='bold')
                    
    ruta_xgb_out = os.path.join(ruta_figuras, 'comparativa_leakage_XGB.png')
    plt.savefig(ruta_xgb_out, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDemostración finalizada")
    print(f"Gráfica Random Forest guardada en: '{ruta_rf_out}'")
    print(f"Gráfica XGBoost guardada en: '{ruta_xgb_out}'")

if __name__ == "__main__":
    try:
        demostar_leakage()
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")