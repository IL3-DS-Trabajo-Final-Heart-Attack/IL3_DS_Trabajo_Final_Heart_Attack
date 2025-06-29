# Proyecto de Ciencia de Datos: Predicción de Ataques Cardíacos

Este proyecto tiene como objetivo analizar y predecir la ocurrencia de ataques cardíacos utilizando técnicas de ciencia de datos. El flujo de trabajo se divide en tres etapas principales:

## 1. Análisis Exploratorio

En esta fase se realiza una exploración inicial de los datos para comprender su estructura, distribución y posibles relaciones entre variables. Se identifican patrones, valores atípicos y se visualizan las características más relevantes del dataset.

## 2. Análisis Multivariante

Aquí se profundiza en el estudio de las relaciones entre múltiples variables, utilizando técnicas estadísticas y visualizaciones avanzadas. El objetivo es identificar combinaciones de factores que puedan estar asociadas con un mayor riesgo de ataque cardíaco.

## 3. Entrenamiento de Random Forest

Finalmente, se emplea el algoritmo de Random Forest para construir un modelo predictivo capaz de estimar la probabilidad de un ataque cardíaco en función de las variables analizadas. Se evalúa el desempeño del modelo y se interpretan los resultados obtenidos.

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
