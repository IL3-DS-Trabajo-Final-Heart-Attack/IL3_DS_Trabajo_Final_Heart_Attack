# Proyecto de Ciencia de Datos: Predicción de Ataques Cardíacos

Este proyecto tiene como objetivo analizar y predecir la ocurrencia de ataques cardíacos utilizando técnicas de ciencia de datos. El flujo de trabajo se divide en tres etapas principales:

## 1. Análisis Exploratorio

En esta fase se realiza una exploración exhaustiva de los datos para comprender su estructura, distribución y posibles relaciones entre variables. El análisis exploratorio es fundamental para detectar patrones, valores atípicos, errores y guiar el modelado posterior. Las actividades principales incluyen:

- **Carga y visualización inicial del dataset:**  
  Se inspeccionan las primeras filas y la estructura general de los datos, identificando el tipo de cada variable (edad, género, presión arterial, marcadores bioquímicos, etc.) y la presencia de valores atípicos o vacíos. Se utiliza la función `View` y resúmenes de estructura (`str`, `summary`).

- **Estadísticas descriptivas y resumen global:**  
  Se calculan medidas como media, mediana, desviación estándar, mínimos y máximos para todas las variables relevantes. Se emplea la librería `skimr` para obtener resúmenes detallados y agrupados por resultado (evento cardíaco sí/no) y género, permitiendo comparar tendencias entre subgrupos.

- **Distribución y comprobación de normalidad:**  
  Se analiza la distribución de las variables principales mediante histogramas, boxplots y tests de normalidad (por ejemplo, Shapiro-Wilk). Esto permite decidir qué técnicas estadísticas son apropiadas para el análisis posterior.

- **Análisis por subgrupos y segmentación:**  
  Se exploran diferencias en las variables según el resultado (evento cardíaco sí/no) y el género, identificando patrones relevantes y posibles factores de riesgo diferenciados. Se emplean agrupamientos y resúmenes estadísticos por grupo.

- **Estudio detallado de variables bioquímicas:**  
  Se realiza un análisis específico de las variables bioquímicas clave (glucosa en sangre, CK-MB y troponina), comparando sus valores entre los distintos grupos y visualizando sus distribuciones. Se utilizan gráficos comparativos y tablas resumen para facilitar la interpretación.

- **Visualizaciones:**  
  Se emplean diversas técnicas gráficas (histogramas, boxplots, gráficos de barras, etc.) para ilustrar la distribución de las variables y las diferencias entre grupos. Estas visualizaciones ayudan a identificar relaciones no evidentes y a comunicar los hallazgos de manera clara.

- **Detección de valores atípicos y calidad de datos:**  
  Se identifican posibles outliers y se evalúa la calidad de los datos, lo que permite tomar decisiones informadas sobre limpieza y preprocesamiento antes del modelado.

Este análisis exploratorio proporciona una base sólida para el análisis multivariante y el desarrollo de modelos predictivos, asegurando que las decisiones posteriores se fundamenten en una comprensión profunda de los datos.

## 2. Análisis Multivariante

Aquí se profundiza en el estudio de las relaciones entre múltiples variables, utilizando técnicas estadísticas y visualizaciones avanzadas. El objetivo es identificar combinaciones de factores que puedan estar asociadas con un mayor riesgo de ataque cardíaco.

## 3. Entrenamiento de Random Forest

Finalmente, se emplea el algoritmo de Random Forest para construir un modelo predictivo capaz de estimar la probabilidad de un ataque cardíaco en función de las variables analizadas. Se evalúa el desempeño del modelo y se interpretan los resultados obtenidos.

> **Novedad:**  
> Los scripts de preprocesamiento, entrenamiento y evaluación han sido actualizados para funcionar correctamente tanto si se ejecutan desde el Makefile como si se ejecutan directamente con Python, gracias al uso de rutas absolutas basadas en la ubicación del script.

El directorio `03_Random_Forest` contiene:

- `data/`: Datos en crudo (`raw/`), procesados (`processed/`) y referencias externas (`external/`).
- `models/`: Modelos entrenados en formato `.joblib`.
- `notebooks/`: Notebooks de experimentación y análisis.
- `reports/`: Resultados de la evaluación del modelo, incluyendo:
  - `classification_report.txt`: Reporte de clasificación.
  - `confusion_matrix.txt`: Matriz de confusión.
  - `feature_importances.txt`: Importancia de variables.
  - `roc_curve.png`: Curva ROC del modelo.
  - `trees/tree_*.png`: Imágenes de los primeros 10 árboles del Random Forest.
- `src/`: Scripts fuente para preprocesamiento, entrenamiento y evaluación.
- `Makefile`: Automatización de tareas.
- `environment.yml`: Dependencias del entorno Conda.

Al ejecutar `make evaluate`, se generan automáticamente los archivos de reporte y las imágenes de los árboles en la carpeta `reports/`. Las imágenes de los árboles (`tree_*.png`) muestran solo los primeros niveles para facilitar la interpretación. Si deseas visualizar los árboles completos o en otros formatos, puedes modificar el script `src/evaluate.py` o exportar a `.dot` y usar Graphviz.

### Visualización interactiva y predicción con Streamlit

El proyecto incluye una aplicación web interactiva desarrollada con Streamlit, ubicada en `03_Random_Forest/streamlit/app.py`. Esta app permite:

- Visualizar la importancia de las variables.
- Consultar la curva ROC y los reportes de clasificación.
- Explorar los primeros 9 árboles individuales del modelo Random Forest.
- Realizar predicciones para nuevos pacientes ingresando sus datos manualmente.

#### Requisitos para la app

Además del entorno principal (`environment.yml`), la app de Streamlit requiere los paquetes listados en `03_Random_Forest/streamlit/requirements.txt`. Puedes instalar estos requisitos adicionales con:

```bash
pip install -r 03_Random_Forest/streamlit/requirements.txt
```

#### Ejecución de la app

Desde la carpeta `03_Random_Forest/streamlit`, ejecuta:

```bash
streamlit run app.py
```

Esto abrirá la interfaz web en tu navegador, donde podrás explorar los resultados y realizar predicciones.

---

> **Nota:**  
> Las imágenes de los árboles (`reports/trees/tree_*.png`) pueden visualizarse tanto en los reportes generados como en la app de Streamlit, facilitando la interpretación del modelo.

### Configuración del entorno

Para asegurar la correcta ejecución del proyecto, es recomendable crear un entorno de Conda a partir del archivo de requerimientos:

```bash
conda env create -f 03_Random_Forest/environment.yml
conda activate rf_env
```

### Uso de Makefile

El proyecto utiliza un archivo `Makefile` para automatizar tareas comunes como el preprocesamiento, entrenamiento y evaluación del modelo. Esto facilita la ejecución de los scripts con simples comandos, sin necesidad de recordar rutas o parámetros.

#### Comandos disponibles

Desde la carpeta `03_Random_Forest`, puedes ejecutar:

- Preprocesar datos:

  ```bash
  make preprocess
  ```

- Entrenar el modelo:

  ```bash
  make train
  ```

- Evaluar el modelo:

  ```bash
  make evaluate
  ```

- Limpiar archivos generados:

  ```bash
  make clean
  ```

#### Instalación de Make en Windows

En sistemas Windows, `make` no está disponible por defecto. Puedes instalarlo usando [Chocolatey](https://chocolatey.org/) ejecutando la terminal como administrador:

```powershell
choco install make
```

Si tienes problemas de permisos, asegúrate de abrir PowerShell o CMD como **Administrador**.

---

**Nota:**  
Para la estandarización de los nombres de las variables utilizadas en el proyecto, consulta el archivo [`name_references.md`](03_Random_Forest/data/external/name_references.md), donde se detallan las correspondencias entre los nombres originales y los nombres estandarizados empleados en el análisis.

**Referencias:**  

- Archivo de entorno: [`environment.yml`](03_Random_Forest/environment.yml)  
- Automatización de tareas: [`Makefile`](03_Random_Forest/Makefile)

---
