"""Pelatihan model untuk MLflow Project (dijalankan oleh `mlflow run`).

Pemakaian: python modelling.py <n_estimators> <max_depth> <folder_dataset>

Folder dataset berisi train.csv dan test.csv hasil prapemrosesan. Skrip ini
sengaja tidak mengatur tracking URI maupun nama eksperimen: `mlflow run`
sudah menyiapkan run aktif di penyimpanan lokal ./mlruns (eksperimen 0).
"""
import os
import sys
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

TARGET = "Machine_failure"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def baca_argumen(argv):
    n_estimators = int(argv[1]) if len(argv) > 1 else 100
    max_depth = int(argv[2]) if len(argv) > 2 else 10
    folder = argv[3] if len(argv) > 3 else "ai4i2020_preprocessing"
    if not os.path.isabs(folder):
        folder = os.path.join(BASE_DIR, folder)
    return n_estimators, max_depth, folder


def muat_data(folder):
    train = pd.read_csv(os.path.join(folder, "train.csv"))
    test = pd.read_csv(os.path.join(folder, "test.csv"))
    return (
        train.drop(columns=[TARGET]), train[TARGET],
        test.drop(columns=[TARGET]), test[TARGET],
    )


def main(argv):
    warnings.filterwarnings("ignore")
    n_estimators, max_depth, folder = baca_argumen(argv)
    X_train, y_train, X_test, y_test = muat_data(folder)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, pred))
        mlflow.log_metric("test_recall", recall_score(y_test, pred, zero_division=0))
        mlflow.log_metric("test_f1", f1_score(y_test, pred, zero_division=0))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5),
        )


if __name__ == "__main__":
    main(sys.argv)
