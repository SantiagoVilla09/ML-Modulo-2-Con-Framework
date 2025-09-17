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

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    learning_curve, validation_curve
)
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
# CSV_FILE = "cars.csv"
# CSV_FILE = "tennis.csv"
MODEL = "rf"                 # Options: "tree", "rf", "logistic", "nb", "svm"
TEST_SIZE = 0.2
VALID_SIZE = 0.2             # <- porcentaje de validación dentro de TRAIN
RANDOM_STATE = 42

# Output filenames (requeridos por la entrega anterior)
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
    feature_cols = df.columns[:-1].tolist()
    target_col = df.columns[-1]

    X = df[feature_cols].astype(str)
    y = df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df  = pd.concat([X_test.reset_index(drop=True),  y_test.reset_index(drop=True)],  axis=1)

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
# EXTRA: Evidencias cuantitativas de bias/varianza
# =======================
def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def savefig_both(fname_root: str, fig):
    """Guarda en ./ y en ./resultados/ para que siempre lo veas en el Explorer."""
    fig.tight_layout()
    fig.savefig(fname_root)
    ensure_dir("resultados")
    fig.savefig(os.path.join("resultados", os.path.basename(fname_root)))

def evidencia_bias_varianza(model_name: str,
                            pipe: Pipeline,
                            X_train, y_train,
                            X_val, y_val,
                            X_train_full, y_train_full):
    """
    Genera y guarda:
      - resultados/diagnostico_bias_varianza.csv
      - resultados/cv_scores.csv
      - train_val_gap.png (+ copia en resultados/)
      - learning_curve_shaded.png (+ copia en resultados/)
      - validation_curve_*.png (+ copia en resultados/)
    """
    ensure_dir("resultados")

    # 1) Gap Train vs Validation + diagnóstico
    acc_train = accuracy_score(y_train, pipe.predict(X_train))
    acc_val   = accuracy_score(y_val,   pipe.predict(X_val))
    gap = float(acc_train - acc_val)

    if acc_train < 0.85 and acc_val < 0.85:
        bias, var, fit = "alto", "bajo", "underfit"
    elif gap > 0.05 and acc_val < 0.98:
        bias, var, fit = "bajo", "alto", "overfit"
    elif 0.0 <= gap <= 0.05 and acc_val >= 0.98:
        bias, var, fit = "bajo", "bajo", "fit"
    else:
        bias, var, fit = "medio", "medio", "fit"

    diag_df = pd.DataFrame([
        {"metric":"accuracy_train", "value":acc_train},
        {"metric":"accuracy_validation", "value":acc_val},
        {"metric":"gap_train_minus_val", "value":gap},
        {"metric":"bias", "value":bias},
        {"metric":"variance", "value":var},
        {"metric":"fit_level", "value":fit},
    ])
    diag_df.to_csv("resultados/diagnostico_bias_varianza.csv", index=False)

    # Gráfico Train vs Validation
    fig = plt.figure(figsize=(6,4))
    plt.bar(["Train","Validation"], [acc_train, acc_val])
    plt.ylim(max(0.95, min(acc_val, acc_train)-0.01), 1.005)
    for i, v in enumerate([acc_train, acc_val]):
        plt.text(i, v+0.0003, f"{v:.4f}", ha="center", fontsize=9)
    plt.title(f"Accuracy Train vs Validation (gap={gap:.4f})")
    plt.ylabel("Accuracy")
    savefig_both("train_val_gap.png", fig)
    plt.close(fig)

    # 2) Validación cruzada (k=5) en Train_Full
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train_full, y_train_full, cv=cv, scoring="accuracy", n_jobs=1)
    cv_df = pd.DataFrame({"fold": np.arange(1, len(cv_scores)+1), "accuracy": cv_scores})
    cv_df["mean"] = cv_scores.mean()
    cv_df["std"]  = cv_scores.std(ddof=1)
    cv_df.to_csv("resultados/cv_scores.csv", index=False)

    # 3) Learning curve con bandas ±1σ
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train_full, y_train_full, cv=cv,
        train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy", n_jobs=1,
        shuffle=True, random_state=RANDOM_STATE
    )
    tr_m, tr_s = train_scores.mean(axis=1), train_scores.std(axis=1)
    va_m, va_s = val_scores.mean(axis=1), val_scores.std(axis=1)

    fig = plt.figure(figsize=(7,5))
    plt.plot(train_sizes, tr_m, marker="o", label="Train")
    plt.fill_between(train_sizes, tr_m-tr_s, tr_m+tr_s, alpha=0.2)
    plt.plot(train_sizes, va_m, marker="o", label="Validation")
    plt.fill_between(train_sizes, va_m-va_s, va_m+va_s, alpha=0.2)
    plt.xlabel("Training samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve (mean ± 1σ)")
    plt.legend()
    savefig_both("learning_curve_shaded.png", fig)
    # extra: compatibilidad con tu nombre anterior
    fig2 = plt.figure(figsize=(7,5))
    plt.plot(train_sizes, tr_m, marker="o", label="Train")
    plt.plot(train_sizes, va_m, marker="o", label="Validation")
    plt.xlabel("Training samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    savefig_both("learning_curve.png", fig2)
    plt.close(fig); plt.close(fig2)

    # 4) Validation curve (hiperparámetro principal según modelo)
    param_name = None
    param_range = None
    m = model_name.lower()
    if m in ("rf", "tree"):
        param_name = "clf__max_depth"
        param_range = [None, 4, 6, 8, 12, 16]
    elif m in ("svm",):
        param_name = "clf__C"
        param_range = [0.1, 0.5, 1.0, 5.0, 10.0]
    elif m in ("logistic",):
        param_name = "clf__C"
        param_range = [0.1, 0.5, 1.0, 5.0, 10.0]
    # MultinomialNB no tiene un parámetro análogo útil para curva simple

    if param_name is not None:
        try:
            tr_sc, va_sc = validation_curve(
                pipe, X_train_full, y_train_full,
                param_name=param_name, param_range=param_range,
                cv=cv, scoring="accuracy", n_jobs=1
            )
            tr_m, tr_s = tr_sc.mean(axis=1), tr_sc.std(axis=1)
            va_m, va_s = va_sc.mean(axis=1), va_sc.std(axis=1)

            x = np.arange(len(param_range))
            labels = [str(p) for p in param_range]
            fig = plt.figure(figsize=(7,5))
            plt.plot(x, tr_m, marker="o", label="Train")
            plt.fill_between(x, tr_m-tr_s, tr_m+tr_s, alpha=0.2)
            plt.plot(x, va_m, marker="o", label="Validation")
            plt.fill_between(x, va_m-va_s, va_m+va_s, alpha=0.2)
            plt.xticks(x, labels)
            plt.xlabel(param_name.replace("clf__", ""))
            plt.ylabel("Accuracy")
            plt.title(f"Validation Curve - {param_name.replace('clf__','')} (mean ± 1σ)")
            plt.legend()
            savefig_both("validation_curve_param.png", fig)
            plt.close(fig)
        except Exception as e:
            with open("resultados/validation_curve_info.txt","w") as f:
                f.write(f"No se pudo generar validation_curve para {param_name}: {e}\n")

    # Mensaje resumen
    print("\n[Diagnóstico cuantitativo]")
    print(f"  Train acc: {acc_train:.6f}  |  Val acc: {acc_val:.6f}  |  Gap: {gap:.6f}")
    print(f"  CV (k=5)  : mean={cv_scores.mean():.6f}  ±  std={cv_scores.std(ddof=1):.6f}")
    print(f"  Bias={bias}  Varianza={var}  Ajuste={fit}")
    print("  Archivos guardados: diagnostico_bias_varianza.csv, cv_scores.csv,")
    print("    train_val_gap.png, learning_curve.png, learning_curve_shaded.png,")
    print("    validation_curve_param.png (si aplica).")


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
    X_train_full, X_test, y_train_full, y_test, feature_cols, target_col = load_train_test(TRAIN_CSV, TEST_CSV)

    # 3b) Split TRAIN into TRAIN/VALIDATION
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VALID_SIZE,
        random_state=RANDOM_STATE, stratify=y_train_full
    )

    # 4) Build the model pipeline
    model_name, estimator = build_estimator(MODEL, RANDOM_STATE)
    pipe = make_pipeline(feature_cols, estimator)

    # 5) Fit the model (solo con TRAIN; VALID sirve para diagnóstico)
    pipe.fit(X_train, y_train)

    # 6) Predict on TEST (entrega original) y también en VALID para diagnóstico
    y_pred_test = pipe.predict(X_test)
    y_pred_val  = pipe.predict(X_val)  # (no se exporta; solo diagnóstico)

    # 7) Compute metrics (for console output)
    acc = accuracy_score(y_test, y_pred_test)
    prec_macro = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred_test, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    cr_text = classification_report(y_test, y_pred_test, zero_division=0)

    # 8) Console output
    print("\n=== Module 2 – Frameworks (Categorical Classification) ===")
    print(f"Original dataset : {os.path.basename(CSV_FILE)}")
    print(f"Train CSV .......: {TRAIN_CSV}")
    print(f"Test CSV ........: {TEST_CSV}")
    print(f"Model ...........: {model_name}")
    print(f"Test size .......: {TEST_SIZE}   |   Valid size (within train) : {VALID_SIZE}")
    print("\nMetrics (test):")
    print(f"  - accuracy        : {acc:.4f}")
    print(f"  - precision_macro : {prec_macro:.4f}")
    print(f"  - recall_macro    : {rec_macro:.4f}")
    print(f"  - f1_macro        : {f1_macro:.4f}")
    print("\nClassification Report:\n")
    print(cr_text)

    # 9) Export the three required CSV outputs (como en la entrega pasada)
    export_predictions_csv(y_test.to_numpy(), y_pred_test, PRED_CSV)
    export_confusion_matrix_csv(y_test.to_numpy(), y_pred_test, CM_CSV)
    export_metrics_csv(y_test.to_numpy(), y_pred_test, METRICS_CSV)

    print(f"\n[OK] Predictions CSV saved to: {PRED_CSV}")
    print(f"[OK] Confusion matrix CSV saved to: {CM_CSV}")
    print(f"[OK] Metrics CSV saved to: {METRICS_CSV}")

    # 10) Evidencias de bias/varianza + curvas con bandas ±1σ (para respuesta al profe)
    evidencia_bias_varianza(model_name, pipe, X_train, y_train, X_val, y_val, X_train_full, y_train_full)


if __name__ == "__main__":
    main()
