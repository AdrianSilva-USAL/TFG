import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def importancia_caracteristicas():
    """Evaluar los modelos haciendo uso o no de la caracteristica redshift para saber que peso tiene a la hora de tomar decisiones"""
    print("Generando gráficos circularees...")
    
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
    
    #Lista de características
    columnas_base = ['u', 'g', 'r', 'i', 'z', 'u_g', 'g_r', 'r_i', 'i_z']
    
    #CON REDSHIFT
    X_con = df[columnas_base + ['redshift']].dropna()
    y_con = df.loc[X_con.index, 'class']
    
    #SIN REDSHIFT
    X_sin = df[columnas_base].dropna()
    y_sin = df.loc[X_sin.index, 'class']
    
    #Codificación de variables (necesaria para XGBoost)
    le = LabelEncoder()
    y_con_encoded = le.fit_transform(y_con)
    y_sin_encoded = le.transform(y_sin)

    #Diccionario de nombres para los gráficos
    nombres_variables = {
        'redshift': 'Redshift', 'u': 'Filtro u', 'g': 'Filtro g', 'r': 'Filtro r', 
        'i': 'Filtro i', 'z': 'Filtro z', 'u_g': 'Índice u-g', 'g_r': 'Índice g-r', 
        'r_i': 'Índice r-i', 'i_z': 'Índice i-z'
    }

    ruta_figuras = os.path.join('reports', 'figures')
    os.makedirs(ruta_figuras, exist_ok=True)
    
    #Paleta de colores fija
    colores = ['#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#95a5a6', '#d35400', '#e74c3c']

    print("Entrenando y graficando Random Forest...")
    rf_con = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100).fit(X_con, y_con_encoded)
    rf_sin = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100).fit(X_sin, y_sin_encoded)
    
    imp_rf_con = pd.Series(rf_con.feature_importances_, index=X_con.columns).rename(nombres_variables).sort_values(ascending=False)
    imp_rf_sin = pd.Series(rf_sin.feature_importances_, index=X_sin.columns).rename(nombres_variables).sort_values(ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    #Random Forest con Redshift
    ax1.pie(
        imp_rf_con, labels=imp_rf_con.index, autopct='%1.1f%%', startangle=140, colors=colores,
        textprops={'fontsize': 9, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.8, 'alpha': 0.85}
    )
    ax1.set_title('Con Redshift (Data Leakage)', fontsize=12, fontweight='bold', pad=15)
    
    #Random Forest sin Redshift
    ax2.pie(
        imp_rf_sin, labels=imp_rf_sin.index, autopct='%1.1f%%', startangle=90, colors=colores,
        textprops={'fontsize': 9, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.8, 'alpha': 0.85}
    )
    ax2.set_title('Sin Redshift (Modelo entrenado)', fontsize=12, fontweight='bold', pad=15)
    
    plt.suptitle('RANDOM FOREST: IMPORTANCIA DE LAS CARACTERÍSTICAS', fontsize=14, fontweight='bold', y=0.98)
    ruta_rf = os.path.join(ruta_figuras, 'importancia_redshift_RF.png')
    plt.tight_layout()
    plt.savefig(ruta_rf, dpi=300, bbox_inches='tight')
    plt.close()

    print("Entrenando y graficando XGBoost...")
    xgb_con = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss').fit(X_con, y_con_encoded)
    xgb_sin = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss').fit(X_sin, y_sin_encoded)
    
    imp_xgb_con = pd.Series(xgb_con.feature_importances_, index=X_con.columns).rename(nombres_variables).sort_values(ascending=False)
    imp_xgb_sin = pd.Series(xgb_sin.feature_importances_, index=X_sin.columns).rename(nombres_variables).sort_values(ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    #XGB Con Redshift
    ax1.pie(
        imp_xgb_con, labels=imp_xgb_con.index, autopct='%1.1f%%', startangle=140, colors=colores,
        textprops={'fontsize': 9, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.8, 'alpha': 0.85}
    )
    ax1.set_title('Con Redshift (Data Leakage)', fontsize=12, fontweight='bold', pad=15)
    
    #XGB Sin Redshift
    ax2.pie(
        imp_xgb_sin, labels=imp_xgb_sin.index, autopct='%1.1f%%', startangle=90, colors=colores,
        textprops={'fontsize': 9, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.8, 'alpha': 0.85}
    )
    ax2.set_title('Sin Redshift (Modelo entrenado)', fontsize=12, fontweight='bold', pad=15)
    
    plt.suptitle('XGBOOST: IMPORTANCIA DE LAS CARACTERÍSTICAS', fontsize=14, fontweight='bold', y=0.98)
    ruta_xgb = os.path.join(ruta_figuras, 'importancia_redshift_XGB.png')
    plt.tight_layout()
    plt.savefig(ruta_xgb, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráficas generadas")
    print(f"Imagen Random Forest: '{ruta_rf}'")
    print(f"Imagen XGBoost: '{ruta_xgb}'")

if __name__ == "__main__":
    try:
        importancia_caracteristicas()
    except Exception as e:
        print(f"❌ Error al ejecutar el script: {e}")