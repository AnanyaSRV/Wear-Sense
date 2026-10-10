
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import load
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path("data") / "UCI HAR Dataset"
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("report")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "RBF SVM": "rbf_svm.joblib",
}


def main():
    train_dir = ROOT / "train"

    X = pd.read_csv(
        train_dir / "X_train.txt",
        sep=r"\s+",
        header=None,
    )

    y = pd.read_csv(
        train_dir / "y_train.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0].to_numpy()

    groups = pd.read_csv(
        train_dir / "subject_train.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0].to_numpy()

    print("Training samples:", len(X))
    print("Training participants:", len(np.unique(groups)))

    # Five folds; each participant belongs to one validation fold.
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for model_name, filename in MODEL_FILES.items():
        model = load(MODEL_DIR / filename)

        print(f"\nEvaluating {model_name}...")

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups=groups),
            start=1,
        ):
            # Check that participants do not overlap.
            train_subjects = set(groups[train_idx])
            val_subjects = set(groups[val_idx])

            assert train_subjects.isdisjoint(val_subjects), (
                "Participant leakage detected!"
            )

            # Reuse the saved model's estimator configuration,
            # but fit a fresh copy using this fold's training data.
            fold_model = clone(model)
            fold_model.fit(X.iloc[train_idx], y[train_idx])

            predictions = fold_model.predict(X.iloc[val_idx])

            results.append({
                "model": model_name,
                "fold": fold,
                "validation_subjects": len(val_subjects),
                "accuracy": accuracy_score(
                    y[val_idx], predictions
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y[val_idx], predictions
                ),
                "macro_f1": f1_score(
                    y[val_idx],
                    predictions,
                    average="macro",
                ),
            })

            print(
                f"  Fold {fold}: "
                f"accuracy={results[-1]['accuracy']:.4f}, "
                f"balanced_accuracy="
                f"{results[-1]['balanced_accuracy']:.4f}, "
                f"macro_f1={results[-1]['macro_f1']:.4f}, "
                f"validation_subjects={len(val_subjects)}"
            )

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df.to_csv(
        OUTPUT_DIR / "grouped_validation_folds.csv",
        index=False,
    )

    summary = (
        results_df.groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .sort_values("macro_f1_mean", ascending=False)
    )

    summary.to_csv(
        OUTPUT_DIR / "grouped_validation_summary.csv"
    )

    print("\nGROUPED CROSS-VALIDATION SUMMARY")
    print(
        summary.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nSaved:")
    print("report/grouped_validation_folds.csv")
    print("report/grouped_validation_summary.csv")


if __name__ == "__main__":
    main()
