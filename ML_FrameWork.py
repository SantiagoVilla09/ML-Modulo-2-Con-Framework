"""
Portfolio – Module 2 (Frameworks): Classification with categorical datasets

Autor: (Santiago Villazón Ponce de León - A01746396)

Dependencias:
  pip install scikit-learn pandas numpy matplotlib
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============
# USER CONFIG
# ============
CSV_FILE = "mushrooms.csv"   # <-- Change Data Set
MODEL = "rf"                 # Options: "tree", "rf", "logistic", "nb", "svm"
TEST_SIZE = 0.2
VALID_SIZE = 0.2             # >>> ADD: proporción de validación dentro de train
RANDOM_STATE = 42

# Output filenames
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
PRED_CSV = "predicciones.csv"
CM_CSV   = "matriz_confusion.csv"
METRICS_CSV = "metricas.csv"


# =======================
# CLASSIFIERS
# =======================
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

def build_estimator(name: str, random_state: int):
    name = name.lower().strip()
    if name == "tree":
        return "DecisionTreeClassifier", DecisionTreeClassifier(random_state=random_state)
    if name == "rf":
        return "RandomForestClassifier", RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    if name == "logistic":
        return "LogisticRegression", LogisticRegression(max_iter=2000, solver="liblinear", multi_class="ovr")
    if name == "nb":
        return "MultinomialNB", MultinomialNB()
    if name == "svm":
        return "LinearSVC", LinearSVC()
    raise ValueError(f"Unsupported model: {name}")


# =======================
# DATA I/O AND PIPELINE
# =======================
def load_categorical_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).apply(lambda c: c.str.strip() if c.dtype == "object" else c)
    return df

def make_pipeline(feature_columns, estimator) -> Pipeline:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    pre = ColumnTransformer(
        transformers=[("cat", ohe, feature_columns)],
        remainder="drop"
    )
    return Pipeline(steps=[("preprocess", pre), ("clf", estimator)])


# =======================
# EXPORT UTILITIES
# =======================
def export_predictions_csv(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": (y_true == y_pred).astype(int)
    })
    df.to_csv(out_path, index=False)

def export_confusion_matrix_csv(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    labels = sorted(pd.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=pd.Index(labels, name="true\\pred"), columns=labels)
    cm_df.to_csv(out_path)

def export_metrics_csv(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    rows = []
    rows.append(("global", "-", "accuracy", round(float(accuracy_score(y_true, y_pred)), 6)))
    rows.append(("global", "-", "precision_macro", round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
    rows.append(("global", "-", "recall_macro", round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
    rows.append(("global", "-", "f1_macro", round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
    out_df = pd.DataFrame(rows, columns=["level", "label", "metric", "value"])
    out_df.to_csv(out_path, index=False)


# =======================
# MAIN
# =======================
def main():
    # 1) Load dataset
    full_df = load_categorical_csv(CSV_FILE)
    feature_cols = full_df.columns[:-1].tolist()
    target_col = full_df.columns[-1]

    X = full_df[feature_cols].astype(str)
    y = full_df[target_col].astype(str)

    # 2) Split Train/Test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3) Split Train/Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VALID_SIZE, random_state=RANDOM_STATE, stratify=y_train_full
    )

    # 4) Build pipeline
    model_name, estimator = build_estimator(MODEL, RANDOM_STATE)
    pipe = make_pipeline(feature_cols, estimator)

    # 5) Fit
    pipe.fit(X_train, y_train)

    # 6) Predict
    y_pred_test = pipe.predict(X_test)
    y_pred_val = pipe.predict(X_val)

    # 7) Metrics
    acc = accuracy_score(y_test, y_pred_test)
    prec_macro = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred_test, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    # 8) Console output
    print("\n=== Module 2 – Frameworks (Categorical Classification) ===")
    print(f"Dataset ........: {os.path.basename(CSV_FILE)}")
    print(f"Model ..........: {model_name}")
    print("\nMetrics (test):")
    print(f"  - accuracy        : {acc:.4f}")
    print(f"  - precision_macro : {prec_macro:.4f}")
    print(f"  - recall_macro    : {rec_macro:.4f}")
    print(f"  - f1_macro        : {f1_macro:.4f}")

    # >>> ADD: Diagnóstico Bias/Varianza/Ajuste
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    val_acc = accuracy_score(y_val, y_pred_val)
    gap = train_acc - val_acc
    if train_acc < 0.85 and val_acc < 0.85:
        bias, variance, fit = "alto", "bajo", "underfit"
    elif gap > 0.05 and val_acc < 0.98:
        bias, variance, fit = "bajo", "alto", "overfit"
    elif 0 <= gap <= 0.05 and val_acc >= 0.98:
        bias, variance, fit = "bajo", "bajo", "fit"
    else:
        bias, variance, fit = "medio", "medio", "fit"

    print("\n=== Diagnóstico del modelo ===")
    print(f"Bias (sesgo): {bias}")
    print(f"Varianza   : {variance}")
    print(f"Ajuste     : {fit}")

    # >>> ADD: Curva de aprendizaje
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train_full, y_train_full, cv=3,
        train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy", n_jobs=1
    )
    plt.figure(figsize=(7,5))
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train")
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="Validation")
    plt.xlabel("Training samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig("learning_curve.png")
    plt.close()
    print("[OK] Learning curve saved to learning_curve.png")

    # 9) Export CSVs
    export_predictions_csv(y_test.to_numpy(), y_pred_test, PRED_CSV)
    export_confusion_matrix_csv(y_test.to_numpy(), y_pred_test, CM_CSV)
    export_metrics_csv(y_test.to_numpy(), y_pred_test, METRICS_CSV)


if __name__ == "__main__":
    main()
