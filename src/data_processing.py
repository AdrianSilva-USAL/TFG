import os
import pandas as pd
from sklearn.model_selection import train_test_split

def cargar_datos_brutos(ruta_csv):
    """Carga el dataset original del SDSS."""
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el dataset original en: {ruta_csv}\n"
                                f"Por favor, asegúrate de haber descargado el archivo de Kaggle "
                                f"y haberlo renombrado a 'star_classification.csv'.")
    print(f"Cargando datos brutos desde {ruta_csv}...")
    return pd.read_csv(ruta_csv)

def limpiar_caracteristicas(df):
    """Elimina columnas innecesarias, evitar el Data Leakage y calcula índices de color."""
    print("Iniciando limpieza y estrucuturado de caracteristicas")
    
    #Clonar para no modificar el datasheet original
    df_limpio = df.copy()
    
    #Eliminación de columnas inecesarias
    columnas_a_eliminar = [
        'obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 
        'spec_obj_ID', 'plate', 'MJD', 'fiber_ID', 'alpha', 'delta', 'redshift'
    ]
    df_limpio = df_limpio.drop(columns=columnas_a_eliminar, errors='ignore')
    
    #Cálculo de los indices de color
    df_limpio['u_g'] = df_limpio['u'] - df_limpio['g']
    df_limpio['g_r'] = df_limpio['g'] - df_limpio['r']
    df_limpio['r_i'] = df_limpio['r'] - df_limpio['i']
    df_limpio['i_z'] = df_limpio['i'] - df_limpio['z']
    
    print("Eliminadas las columnas innecesarias para el entrenamiento y para evitar el Data Leakage")
    print("Añadidas las columnas de los indices de color: ([u-g], [g-r], [r-i], [i-z]).")
    return df_limpio

def ejecutar_particionado_y_guardado(df, ruta_salida):
    """Divide el dataset en un 80/20 de datos y genera los archivos CSV finales de cada bloque."""
    print("Separando el dataset en Bloque de Desarrollo (80%) y Test Final (20%)...")
    
    #Garantizar que el % de estrellas, galaxias y cuásares sea igual en ambos bloques
    df_desarrollo, df_test = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42, 
        stratify=df['class']
    )
    
    #Extraer 50 filas para usar como archivo de demostración en la interfaz
    df_demo = df_test.sample(n=50, random_state=42)
    
    #Crear la carpeta destino 'data/processed' si el sistema operativo no la ha creado aún
    os.makedirs(ruta_salida, exist_ok=True)
    
    #Guardar los archivos resultantes en las rutas especificas
    df_desarrollo.to_csv(os.path.join(ruta_salida, 'desarrollo.csv'), index=False)
    df_test.to_csv(os.path.join(ruta_salida, 'test_final.csv'), index=False)
    df_demo.to_csv(os.path.join(ruta_salida, 'datos_demo.csv'), index=False)
    
    print(f"\nArchivos generados en '{ruta_salida}':")
    print(f"─ desarrollo.csv -> {len(df_desarrollo)} filas: Entrenamiento y curvas de Machine Learning (80%)")
    print(f"─ test_final.csv -> {len(df_test)} filas: Evaluación final (20%)")
    print(f"─ datos_demo.csv -> {len(df_demo)} filas:  Pruebas para la interfaz de la aplicación")

if __name__ == "__main__":
    #Definición de rutas relativas locales
    RUTA_ORIGEN = os.path.join('data', 'raw', 'star_classification.csv')
    RUTA_DESTINO = os.path.join('data', 'processed')
    
    try:
        df_original = cargar_datos_brutos(RUTA_ORIGEN)
        df_procesado = limpiar_caracteristicas(df_original)
        ejecutar_particionado_y_guardado(df_procesado, RUTA_DESTINO)
        print("\nLos datos limpios estan listos para el aprendizaje.")
    except Exception as e:
        print(f"\nFallo en el procesamiento de datos: {e}")