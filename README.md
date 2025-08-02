# Perceptrón Simple y Clasificación

Este proyecto explora el uso de un perceptrón simple para resolver problemas de clasificación binaria utilizando distintos datasets clásicos. El código principal de la neurona está implementado en [`libs/neuronas.py`](libs/neuronas.py).

## Estructura de Notebooks

- **notebooks/main.py**  
  Script Python para pruebas rápidas usando la clase de perceptrón.

- **notebooks/01-comprar-casas.ipynb**  
  Clasificación binaria para decidir si comprar o no una casa, usando características como precio y área. Incluye visualización, normalización y entrenamiento del perceptrón.

- **notebooks/02-mnist.ipynb**  
  Clasificación de dígitos manuscritos (MNIST), enfocándose en distinguir entre dos dígitos (por ejemplo, 0 y 1). Incluye carga, preprocesamiento y entrenamiento.

- **notebooks/03-mnist_7_8.ipynb**  
  Similar al anterior, pero enfocado en la clasificación de los dígitos 7 y 8 del dataset MNIST.

- **notebooks/04-iris.ipynb**  
  Clasificación de especies de flores usando el dataset Iris, filtrando para dos clases. Incluye análisis exploratorio, normalización y entrenamiento.

## Contenido de [`libs/neuronas.py`](libs/neuronas.py)

Este archivo contiene la implementación de la clase [`NeuronaPerceptron`](libs/neuronas.py), que permite:

- Inicializar pesos y bias aleatoriamente.
- Entrenar el perceptrón usando el algoritmo de aprendizaje supervisado.
- Realizar predicciones (feedforward) para una muestra o un lote de datos.
- Métodos auxiliares para la suma ponderada, función de activación y ajuste de parámetros.

## Datasets

Los datasets utilizados se encuentran en la carpeta `data/`:
- `houses1_dataset.csv`
- `Iris.csv`
- `mnist-original.mat`

---

**Requisitos:**  
Python 3, numpy, pandas, matplotlib, seaborn, scikit-learn, scipy

**Uso:**  
Ejecuta los notebooks en Jupyter para explorar y entrenar el perceptrón en cada caso de estudio.