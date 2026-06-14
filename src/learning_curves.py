import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def generar_curvas_unitarias_puntos():
    """Crea las curvas con los parámetros de fábrica del modelo RandomForest"""
    print("Iniciando simulación de Curvas de Aprendizaje Unitarias (Solo Puntos)...")
    
    #Cargar los datos procesados de desarrollo.csv y dividimos las caracteristicas(x) del objetivo(y)
    ruta_desarrollo = os.path.join('data', 'processed', 'desarrollo.csv')
    if not os.path.exists(ruta_desarrollo):
        raise FileNotFoundError("No se encuentra desarrollo.csv. Ejecutar primero scr/data_processing.py")
    
    df = pd.read_csv(ruta_desarrollo)
    X = df.drop(columns=['class'])
    y = df['class']
    
    #Escala exponencial de tamaños de aprendizaje
    tamanos_entrenamiento = [5000, 10000, 20000, 40000, 70000]
    
    historial_star = []
    historial_galaxy = []
    historial_qso = []
    
    #Examen de validación 10.000 filas fijas
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )
    
    # 2. Bucle de entrenamiento acumulativo
    for tamano in tamanos_entrenamiento:
        print(f"  🧠 Evaluando tramo de {tamano} filas...")
        X_sub = X_train_full.head(tamano)
        y_sub = y_train_full.head(tamano)
        
        # Modelo de fábrica
        modelo = RandomForestClassifier(random_state=42, n_jobs=-1)
        modelo.fit(X_sub, y_sub)
        
        preds = modelo.predict(X_val)
        f1_por_clase = f1_score(y_val, preds, average=None, labels=['GALAXY', 'QSO', 'STAR'])
        
        historial_galaxy.append(f1_por_clase[0] * 100)
        historial_qso.append(f1_por_clase[1] * 100)
        historial_star.append(f1_por_clase[2] * 100)

    # 3. Creación y guardado de las 3 gráficas independientes
    ruta_figuras = os.path.join('reports', 'figures')
    os.makedirs(ruta_figuras, exist_ok=True)
    
    print("🎨 Generando archivos de imagen independientes...")

    # --- GRÁFICA 1: ESTRELLAS ---
    plt.figure(figsize=(7, 4.5))
    # plt.scatter dibuja solo los puntos sueltos. s=120 es el tamaño del punto.
    plt.scatter(tamanos_entrenamiento, historial_star, color='#f1c40f', s=120, edgecolors='black', zorder=3)
    plt.title('Evolución del Aprendizaje: Estrellas (STAR)', fontsize=11, fontweight='bold')
    plt.xlabel('Cantidad de Datos de Entrenamiento (Filas)', fontsize=10)
    plt.ylabel('Rendimiento (F1-Score %)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    plt.ylim(50, 102)
    plt.savefig(os.path.join(ruta_figuras, 'curva_aprendizaje_estrellas.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRÁFICA 2: GALAXIAS ---
    plt.figure(figsize=(7, 4.5))
    plt.scatter(tamanos_entrenamiento, historial_galaxy, color='#3498db', s=120, edgecolors='black', zorder=3)
    plt.title('Evolución del Aprendizaje: Galaxias (GALAXY)', fontsize=11, fontweight='bold')
    plt.xlabel('Cantidad de Datos de Entrenamiento (Filas)', fontsize=10)
    plt.ylabel('Rendimiento (F1-Score %)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    plt.ylim(50, 102)
    plt.savefig(os.path.join(ruta_figuras, 'curva_aprendizaje_galaxias.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRÁFICA 3: CUÁSARES ---
    plt.figure(figsize=(7, 4.5))
    plt.scatter(tamanos_entrenamiento, historial_qso, color='#e74c3c', s=120, edgecolors='black', zorder=3)
    plt.title('Evolución del Aprendizaje: Cuásares (QSO)', fontsize=11, fontweight='bold')
    plt.xlabel('Cantidad de Datos de Entrenamiento (Filas)', fontsize=10)
    plt.ylabel('Rendimiento (F1-Score %)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    plt.ylim(50, 102)
    plt.savefig(os.path.join(ruta_figuras, 'curva_aprendizaje_cuasares.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n💾 ¡Las 3 gráficas se han guardado por separado en '{ruta_figuras}':")
    print("   ├── curva_aprendizaje_estrellas.png")
    print("   ├── curva_aprendizaje_galaxias.png")
    print("   └── curva_aprendizaje_cuasares.png")

if __name__ == "__main__":
    try:
        generar_curvas_unitarias_puntos()
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")