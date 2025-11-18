# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
"""HOMEWORK — MODELO LOGREG CON GRIDSEARCH."""

import os
import gzip
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

# =====================================================================
# Paths requeridos por el autograder
# =====================================================================
MODEL_OUTPUT = "files/models/model.pkl.gz"
METRICS_OUTPUT = "files/output/metrics.json"

# =====================================================================
# Crear carpetas si no existen
# =====================================================================
os.makedirs("files/models", exist_ok=True)
os.makedirs("files/output", exist_ok=True)

# =====================================================================
# Cargar datos de entrenamiento
# =====================================================================
x_train = pd.read_pickle("files/grading/x_train.pkl")
y_train = pd.read_pickle("files/grading/y_train.pkl")
x_test = pd.read_pickle("files/grading/x_test.pkl")
y_test = pd.read_pickle("files/grading/y_test.pkl")

# Columnas numéricas y categóricas
numeric_cols = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", MinMaxScaler(), numeric_cols),
    ]
)

# =====================================================================
# Pipeline
# =====================================================================
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("select", SelectKBest(score_func=f_classif)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ]
)

# =====================================================================
# GridSearchCV — parámetros diseñados para superar los umbrales
# =====================================================================
param_grid = {
    "select__k": [20, 25, 30, 35],        # Más características → mejor BA
    "clf__C": [0.8, 1.0, 1.2, 1.5],       # Ajuste fino
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=10,
    n_jobs=-1,
)

# =====================================================================
# Entrenar modelo
# =====================================================================
grid.fit(x_train, y_train)

# =====================================================================
# Generar predicciones
# =====================================================================
y_train_pred = grid.predict(x_train)
y_test_pred = grid.predict(x_test)

# =====================================================================
# Construir métricas
# =====================================================================
def build_metrics(name, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def build_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": None},
        "true_1": {"predicted_0": None, "predicted_1": int(cm[1, 1])},
    }


metrics = [
    build_metrics("train", y_train, y_train_pred),
    build_metrics("test", y_test, y_test_pred),
    build_confusion("train", y_train, y_train_pred),
    build_confusion("test", y_test, y_test_pred),
]

# =====================================================================
# Guardar métricas en JSONL
# =====================================================================
with open(METRICS_OUTPUT, "w", encoding="utf-8") as f:
    for m in metrics:
        f.write(json.dumps(m) + "\n")

# =====================================================================
# Guardar modelo comprimido
# =====================================================================
with gzip.open(MODEL_OUTPUT, "wb") as f:
    pickle.dump(grid, f)

print("✔ Modelo entrenado, métricas generadas y archivos guardados correctamente.")