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

🛠️ Requisitos Previos
Antes de ejecutar el proyecto, asegúrate de tener instalado en tu máquina local:

Python 3.10 o superior.

Docker y Docker Desktop (si deseas ejecutar la versión contenerizada).

Git (para la gestión del repositorio).

🚀 Instrucciones de Ejecución
Tienes dos métodos para arrancar y probar este proyecto:

Opción A: Ejecución Nativa (Entorno Local)
Clonar el repositorio y acceder a la carpeta del proyecto:

Bash
git clone [https://github.com/TU_USUARIO/Desarrollo_TFG.git](https://github.com/TU_USUARIO/Desarrollo_TFG.git)
cd Desarrollo_TFG
Crear y activar un entorno virtual (Recomendado para evitar conflictos de librerías):

Bash
# En Windows:
python -m venv .venv
.venv\Scripts\activate
Instalar las dependencias requeridas:

Bash
pip install -r requirements.txt
Lanzar la Aplicación Web (Streamlit):

Bash
streamlit run src/app.py
La aplicación se abrirá automáticamente en tu navegador en la dirección: http://localhost:8501

Opción B: Ejecución Profesional Automática (Con Docker 🐋)
Esta opción no requiere que tengas instalado Python ni ninguna librería en tu equipo. Docker se encarga de aislar y empaquetar todo el sistema de forma portátil.

Asegúrate de tener Docker Desktop abierto.

Abre la terminal en la raíz del proyecto y enciende la máquina con un único comando:

Bash
docker compose up
Una vez finalice la construcción automática, abre tu navegador e ingresa a:

Plaintext
http://localhost:8501
Para apagar y detener el contenedor de forma segura, ejecuta en la terminal:

Bash
docker compose down
🧪 Experimentos de Control Incluidos
El repositorio incluye herramientas para reproducir los hitos científicos descritos en la memoria académica:

Demostración de Data Leakage (F1-Score con Redshift):
Para comprobar cómo la inclusión de la variable espectroscópica redshift genera un modelo sesgado e artificialmente perfecto del 99.9%, ejecuta:

Bash
python src/demo_redshift_leakage.py
Análisis de Importancia de Variables (Gráficos de Tarta):
Para regenerar la comparativa visual de cómo los modelos distribuyen sus decisiones con y sin la presencia del redshift, ejecuta:

Bash
python src/plot_importance_pie.py
💡 Trabajo de Fin de Grado desarrollado en la convocatoria de 2026 | Adrián Silva Sánchez