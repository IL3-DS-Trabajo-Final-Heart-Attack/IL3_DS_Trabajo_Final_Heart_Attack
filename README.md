# Proyecto de Ciencia de Datos: Predicción de Ataques Cardíacos

Este proyecto tiene como objetivo analizar y predecir la ocurrencia de ataques cardíacos utilizando técnicas de ciencia de datos. El flujo de trabajo se divide en tres etapas principales.

> **¡Prueba la app web interactiva!**
> Accede online a la aplicación de predicción y visualización en:
> [https://heart-attack-ds-rforest.streamlit.app/](https://heart-attack-ds-rforest.streamlit.app/)

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

El análisis exploratorio no solo permite familiarizarse con la estructura y calidad de los datos, sino que también es clave para formular hipótesis y orientar el análisis estadístico posterior. En esta etapa, se presta especial atención a la identificación de posibles sesgos, la distribución de los factores de riesgo clásicos (como edad, presión arterial, glucosa, marcadores cardíacos) y la comparación entre pacientes con y sin eventos cardíacos. Además, se documentan los criterios de exclusión/inclusión de datos y se justifica cualquier transformación o limpieza realizada. Este proceso asegura que los modelos predictivos se construyan sobre una base sólida y representativa de la población estudiada.

## 2. Análisis Multivariante

En esta sección se realiza un análisis multivariante para explorar las relaciones entre múltiples variables clínicas y su asociación con el riesgo de ataque cardíaco. El flujo de trabajo incluye:

- **Carga y preprocesamiento de datos:** Se cargan los datos clínicos, se verifica la ausencia de valores nulos y se transforman variables categóricas a formato numérico.
- **Análisis exploratorio:** Se visualizan las distribuciones de las variables numéricas mediante histogramas y se analiza la matriz de correlación, tanto general como específicamente respecto al resultado clínico ("Result").
- **Visualización de relaciones:** Se emplean pairplots y mapas de calor para identificar patrones y relaciones entre variables.
- **Reducción de dimensionalidad (PCA):** Se aplica Análisis de Componentes Principales (PCA) para visualizar la estructura de los datos y entender qué variables contribuyen más a la variabilidad observada.
- **Clustering:** Se utiliza K-means para identificar grupos de pacientes con características similares, determinando el número óptimo de clusters mediante el método del codo y visualizando los resultados en el espacio de componentes principales.
- **Interpretación de clusters:** Se analizan las características promedio de cada grupo y se visualizan mediante boxplots, facilitando la interpretación clínica de los patrones encontrados.

Este enfoque permite identificar combinaciones de factores asociados a un mayor riesgo de ataque cardíaco y segmentar a los pacientes en grupos con perfiles clínicos diferenciados.

El análisis multivariante profundiza en la comprensión de cómo interactúan múltiples variables simultáneamente y cómo estas interacciones pueden influir en el riesgo de ataque cardíaco. Se exploran relaciones complejas que no serían evidentes en análisis univariantes o bivariantes. Además de las técnicas mencionadas (PCA y clustering), se pueden emplear métodos adicionales como análisis discriminante, regresión logística multivariable o árboles de decisión para validar los hallazgos y robustecer las conclusiones. Las visualizaciones multivariantes, como los mapas de calor y los gráficos de componentes principales, permiten comunicar de manera intuitiva los patrones detectados y facilitan la interpretación clínica de los resultados. Este enfoque integral es fundamental para identificar perfiles de riesgo y posibles subgrupos de pacientes que podrían beneficiarse de intervenciones personalizadas.

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

> **Acceso online:**
> Puedes probar la app directamente desde tu navegador en el siguiente enlace:
> [https://heart-attack-ds-rforest.streamlit.app/](https://heart-attack-ds-rforest.streamlit.app/)

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

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Consulta el archivo [LICENSE](./LICENSE) para más detalles.
