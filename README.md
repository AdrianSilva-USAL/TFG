# 🌌 Clasificador Fotométrico de Cuerpos Celestes (TFG)

Este proyecto constituye el Trabajo de Fin de Grado (TFG) enfocado en el desarrollo de un sistema inteligente de clasificación astronómica multiclase. Utilizando datos fotométricos del **Sloan Digital Sky Survey (SDSS)**, el sistema es capaz de identificar y clasificar objetos celestes en tres categorías: **Galaxias (GALAXY)**, **Cuásares (QSO)** y **Estrellas (STAR)**.

La solución incluye la ingeniería de datos, el entrenamiento optimizado de modelos basados en conjuntos de árboles (**Random Forest** y **XGBoost**), un análisis empírico del fenómeno de *Data Leakage*, y el despliegue de una interfaz gráfica interactiva contenerizada con **Docker**.

---

## 📁 Estructura del Proyecto
```text
Desarrollo_TFG/
├── data/                     # Datasets del proyecto (Raw y Processed)
├── models/                   # Modelos entrenados en formato serializado (.pkl)
├── reports/                  # Informes y figuras generadas para la memoria
│   └── figures/              # Gráficas de rendimiento e importancia de variables
├── src/                      # Código fuente del proyecto
│   ├── app.py                # Interfaz gráfica (GUI) con Streamlit
│   ├── compare_models.py     # Script de evaluación de modelos (GridSearchCV)
│   ├── data_processing.py    # Pipeline de limpieza e ingeniería de características
│   ├── demo_redshift_leakage.py # Script del experimento de control (Data Leakage)
│   └── plot_importance_pie.py   # Generador de gráficos de tarta de importancia
├── Dockerfile                # Configuración de la imagen de Docker
├── docker-compose.yml        # Automatización del entorno multi-contenedor
├── requirements.txt          # Dependencias del proyecto de Python
└── README.md                 # Guía de uso y documentación (este archivo)
```
## 🔄 Flujo de Trabajo y Ejecución de Scripts Básicos
Si deseas replicar el ciclo de vida completo del desarrollo del modelo (desde la limpieza de los datos brutos hasta el entrenamiento definitivo) de forma nativa, ejecuta los scripts en la terminal siguiendo este orden cronológico:
    Preprocesamiento e Ingeniería de Características:
        Limpia el dataset original y calcula de forma automatizada los índices de color ($u-g$, $g-r$, $r-i$, $i-z$) guardando los archivos resultantes en la carpeta data/processed/.
            python src/data_processing.py
    Entrenamiento y Optimización de Modelos (GridSearchCV):
        Ejecuta la validación cruzada y la búsqueda en rejilla para encontrar los hiperparámetros óptimos de Random Forest y XGBoost:
            python src/optimize_random_forest.py
            python src/optimize_xgboost.py
        Al finalizar, guarda las métricas de rendimiento en reports/ y los mejores cerebros (.pkl) en models/.
        Y luego podemos hacer el examen final comparando el mejor modelo de cada tipo con:
            python src/final_test.py

## 🛠️ Requisitos PreviosAntes de ejecutar la aplicación web o los experimentos, asegúrate de tener instalado:
    Python 3.10 o superior.Docker y Docker Desktop (si deseas ejecutar la versión contenerizada).

## 🚀 Instrucciones de Ejecución de la Aplicación (GUI)Tienes dos métodos para arrancar y probar la interfaz gráfica interactiva:
    Opción A: Ejecución Nativa (Entorno Local)Clonar el repositorio y acceder a la carpeta del proyecto:
    git clone [https://github.com/TU_USUARIO/Desarrollo_TFG.git](https://github.com/TU_USUARIO/Desarrollo_TFG.git)
    cd Desarrollo_TFG

    Instalar las dependencias requeridas:
        pip install -r requirements.txt

    Lanzar la Aplicación Web (Streamlit):
        streamlit run src/app.py

    Opción B: Ejecución Automática (Con Docker 🐋)Esta opción no requiere que tengas instalado Python ni ninguna librería en tu equipo. Docker se encarga de aislar y empaquetar todo el sistema con sus puertos mapeados automáticamente.Asegúrate de tener Docker Desktop abierto.Abre la terminal en la raíz del proyecto y enciende la máquina con un único comando:
        docker compose up
    Abre tu navegador e ingresa a: http://localhost:8501
    Para apagar y detener el contenedor de forma segura, ejecuta:
        docker compose down

## 🧪 Experimentos de Control Incluidos
    El repositorio incluye herramientas para reproducir los hitos científicos descritos en la memoria académica relativos al sesgo por Data Leakage:
        Demostración de Data Leakage (F1-Score con Redshift):
            Muestra cómo la inclusión de la variable espectroscópica redshift genera un modelo sesgado y artificialmente perfecto del 99.9% usando las 100.000 líneas de datos.
                python src/redshift_leakage.py

        Análisis de Importancia de Variables (Gráficos ciculares):
            Genera las figuras comparativas que demuestran cómo el redshift fagocita la atención de ambos algoritmos frente al reparto limpio del enfoque fotométrico.
                python src/importance_features.py

## Trabajo de Fin de Grado | GIISI | 2ªConvocatoria de 2026 | Adrián Silva Sánchez