from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from brownbear.models import SVM
from brownbear.preprocessing import split_training_data
from brownbear.svm_preprocessing import make_svm_preprocessor
from brownbear.svm_plot import plot_kernels

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
SEED = 42
SVM_BEST = {"kernel": "gaussian", "C": 3, "gamma": 0.1, "class_weight": {1: 1.3}}


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name:<12} macro F1 = {macro_f1:.4f}   accuracy = {accuracy_score(y_true, y_pred):.4f}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"id": str})

    train_idx, val_idx = split_training_data(train, seed=SEED)
    train_set, val_set = train.iloc[train_idx], train.iloc[val_idx]
    prep = make_svm_preprocessor().fit(train_set)
    X_train, X_val = prep.transform(train_set), prep.transform(val_set)
    y_train, y_val = train_set["is_canceled"].to_numpy(), val_set["is_canceled"].to_numpy()

    print("Validation")
    model = SVM(**SVM_BEST).fit(X_train, y_train)
    scratch_pred = model.predict(X_val)
    report("from scratch", y_val, scratch_pred)

    sklearn_params = {key: value for key, value in SVM_BEST.items() if key != "kernel"}
    sklearn_pred = SVC(kernel="rbf", **sklearn_params).fit(X_train, y_train).predict(X_val)
    report("scikit-learn", y_val, sklearn_pred)
    print(f"Prediction agreement = {np.mean(scratch_pred == sklearn_pred):.4f}")

    plot_kernels(X_train, y_train, OUTPUT_DIR / "svm_kernels.png", seed=SEED)

    final_prep = make_svm_preprocessor().fit(train)
    final_model = SVM(**SVM_BEST).fit(final_prep.transform(train), train["is_canceled"])
    final_model.save(OUTPUT_DIR / "svm_model.pkl")

    loaded = SVM.load(OUTPUT_DIR / "svm_model.pkl")
    submission = pd.DataFrame({
        "id": test["id"],
        "is_canceled": loaded.predict(final_prep.transform(test)).astype(int),
    })
    submission.to_csv(OUTPUT_DIR / "submission_svm.csv", index=False)
    print(f"Saved {len(submission)} predictions to {OUTPUT_DIR / 'submission_svm.csv'}")


if __name__ == "__main__":
    main()
