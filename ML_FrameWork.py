"""
Portfolio – Module 2 (Frameworks): Classification with categorical datasets

Autor: (Santiago Villazón Ponce de León - A01746396)

Dependencias:
  pip install scikit-learn pandas numpy
"""
import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
CSV_FILE = "mushrooms.csv"   # <-- Change Data Set:
# CSV_FILE = "cars.csv"   # 
# CSV_FILE = "tennis.csv"   #
MODEL = "rf"                 # Options: "tree", "rf", "logistic", "nb", "svm"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Output filenames
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
PRED_CSV = "predicciones.csv"
CM_CSV   = "matriz_confusion.csv"
METRICS_CSV = "metricas.csv"


# =======================
# CLASSIFIERS (scikit-learn)
# =======================
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

def build_estimator(name: str, random_state: int):
    """Return (pretty_model_name, estimator) for the selected algorithm."""
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
    """Load the CSV as strings (categorical) and validate basic shape."""
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path, dtype=str).apply(lambda c: c.str.strip() if c.dtype == "object" else c)
    if df.shape[1] < 2:
        print("[ERROR] CSV must have at least 2 columns (features + last column as target).")
        sys.exit(1)
    return df


def split_and_save_train_test(df: pd.DataFrame, test_size: float, random_state: int,
                              train_path: str, test_path: str) -> None:
    """Split the original dataframe into stratified train/test and save as CSV files."""
    # Identify features and target from original dataframe
    feature_cols = df.columns[:-1].tolist()
    target_col = df.columns[-1]

    # Ensure all as strings (categorical)
    X = df[feature_cols].astype(str)
    y = df[target_col].astype(str)

    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
        # If you need reproducibility across runs, keep random_state fixed.
    )

    # Reconstruct train/test dataframes (features + target as last column)
    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df  = pd.concat([X_test.reset_index(drop=True),  y_test.reset_index(drop=True)],  axis=1)

    # Save to CSV
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


def load_train_test(train_path: str, test_path: str):
    """Load previously saved train/test CSVs and return (X_train, X_test, y_train, y_test, feature_cols, target_col)."""
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        print(f"[ERROR] Train/Test CSVs not found. Expected files: {train_path}, {test_path}")
        sys.exit(1)

    train_df = pd.read_csv(train_path, dtype=str).apply(lambda c: c.str.strip() if c.dtype == "object" else c)
    test_df  = pd.read_csv(test_path,  dtype=str).apply(lambda c: c.str.strip() if c.dtype == "object" else c)

    if train_df.shape[1] < 2 or test_df.shape[1] < 2:
        print("[ERROR] Train/Test CSVs must have at least 2 columns (features + last column as target).")
        sys.exit(1)

    feature_cols = train_df.columns[:-1].tolist()
    target_col = train_df.columns[-1]

    X_train = train_df[feature_cols].astype(str)
    y_train = train_df[target_col].astype(str)
    X_test  = test_df[feature_cols].astype(str)
    y_test  = test_df[target_col].astype(str)

    return X_train, X_test, y_train, y_test, feature_cols, target_col


def make_pipeline(feature_columns, estimator) -> Pipeline:
    """Build a pipeline: OneHotEncoder -> Estimator."""
    # OneHotEncoder compatibility for different sklearn versions
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
    """Export y_true, y_pred, and correctness flag (1/0) to CSV."""
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": (y_true == y_pred).astype(int)
    })
    df.to_csv(out_path, index=False)


def export_confusion_matrix_csv(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    """Export the raw confusion matrix with labeled headers and row index."""
    labels = sorted(pd.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=pd.Index(labels, name="true\\pred"), columns=labels)
    cm_df.to_csv(out_path)


def export_metrics_csv(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    """
    Export global and per-class metrics in a single CSV (long format):
      columns = level, label, metric, value
      - level = "global" for aggregated metrics; label = "-"
      - level = "class"  for per-class metrics; label = class name
    """
    rows = []

    # Global metrics
    rows.append(("global", "-", "accuracy", round(float(accuracy_score(y_true, y_pred)), 6)))
    rows.append(("global", "-", "precision_macro", round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
    rows.append(("global", "-", "recall_macro", round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
    rows.append(("global", "-", "f1_macro", round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6)))

    # Per-class metrics via classification_report dict
    cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for label, stats in cr.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append(("class", label, "precision", round(float(stats.get("precision", 0.0)), 6)))
        rows.append(("class", label, "recall",    round(float(stats.get("recall", 0.0)), 6)))
        rows.append(("class", label, "f1_score",  round(float(stats.get("f1-score", 0.0)), 6)))
        rows.append(("class", label, "support",   int(stats.get("support", 0))))

    out_df = pd.DataFrame(rows, columns=["level", "label", "metric", "value"])
    out_df.to_csv(out_path, index=False)


# =======================
# MAIN
# =======================
def main():
    # 1) Load the full original dataset
    full_df = load_categorical_csv(CSV_FILE)

    # 2) Split and SAVE to train/test CSVs BEFORE any modeling
    split_and_save_train_test(
        full_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        train_path=TRAIN_CSV,
        test_path=TEST_CSV
    )
    print(f"[OK] Saved stratified splits: {TRAIN_CSV}, {TEST_CSV}")

    # 3) Load the saved train/test CSVs and prepare X/y
    X_train, X_test, y_train, y_test, feature_cols, target_col = load_train_test(TRAIN_CSV, TEST_CSV)

    # 4) Build the model pipeline
    model_name, estimator = build_estimator(MODEL, RANDOM_STATE)
    pipe = make_pipeline(feature_cols, estimator)

    # 5) Fit the model
    pipe.fit(X_train, y_train)

    # 6) Predict on test
    y_pred = pipe.predict(X_test)

    # 7) Compute metrics (for console output)
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cr_text = classification_report(y_test, y_pred, zero_division=0)

    # 8) Console output
    print("\n=== Module 2 – Frameworks (Categorical Classification) ===")
    print(f"Original dataset : {os.path.basename(CSV_FILE)}")
    print(f"Train CSV .......: {TRAIN_CSV}")
    print(f"Test CSV ........: {TEST_CSV}")
    print(f"Model ...........: {model_name}")
    print(f"Test size .......: {TEST_SIZE}")
    print("\nMetrics (test):")
    print(f"  - accuracy        : {acc:.4f}")
    print(f"  - precision_macro : {prec_macro:.4f}")
    print(f"  - recall_macro    : {rec_macro:.4f}")
    print(f"  - f1_macro        : {f1_macro:.4f}")
    print("\nClassification Report:\n")
    print(cr_text)

    # 9) Export the three required CSV outputs
    export_predictions_csv(y_test.to_numpy(), y_pred, PRED_CSV)
    export_confusion_matrix_csv(y_test.to_numpy(), y_pred, CM_CSV)
    export_metrics_csv(y_test.to_numpy(), y_pred, METRICS_CSV)

    print(f"\n[OK] Predictions CSV saved to: {PRED_CSV}")
    print(f"[OK] Confusion matrix CSV saved to: {CM_CSV}")
    print(f"[OK] Metrics CSV saved to: {METRICS_CSV}\n")


if __name__ == "__main__":
    main()
