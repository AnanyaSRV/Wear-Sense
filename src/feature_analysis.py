
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import load
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path("data") / "UCI HAR Dataset"
MODEL_PATH = Path("models") / "logistic_regression.joblib"
OUTPUT_DIR = Path("report")

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

    subjects = pd.read_csv(
        train_dir / "subject_train.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0].to_numpy()

    feature_lines = (
        (ROOT / "features.txt")
        .read_text()
        .splitlines()
    )
    feature_names = [
        " ".join(line.split()[1:])
        for line in feature_lines
    ]

    model_template = load(MODEL_PATH)

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    importance_values = []
    fold_scores = []

    # Analyze each validation fold separately.
    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups=subjects),
        start=1,
    ):
        model = clone(model_template)
        model.fit(X.iloc[train_idx], y[train_idx])

        X_val = X.iloc[val_idx]
        y_val = y[val_idx]

        baseline_predictions = model.predict(X_val)
        baseline_macro_f1 = f1_score(
            y_val,
            baseline_predictions,
            average="macro",
        )

        # Permutation importance for overall macro-F1.
        overall = permutation_importance(
            model,
            X_val,
            y_val,
            scoring="f1_macro",
            n_repeats=3,
            random_state=42,
            n_jobs=-1,
        )

        # Focus on distinguishing sitting (4) from standing (5).
        posture_mask = np.isin(y_val, [4, 5])
        X_posture = X_val.iloc[np.flatnonzero(posture_mask)]
        y_posture = y_val[posture_mask]

        posture = permutation_importance(
            model,
            X_posture,
            y_posture,
            scoring="f1_macro",
            n_repeats=3,
            random_state=42,
            n_jobs=-1,
        )

        for i, name in enumerate(feature_names):
            importance_values.append({
                "fold": fold,
                "feature": name,
                "overall_importance": overall.importances_mean[i],
                "posture_importance": posture.importances_mean[i],
            })

        fold_scores.append({
            "fold": fold,
            "baseline_macro_f1": baseline_macro_f1,
            "validation_subjects": len(
                np.unique(subjects[val_idx])
            ),
        })

        print(
            f"Fold {fold}/5 complete | "
            f"macro-F1={baseline_macro_f1:.4f}"
        )

    importance_df = pd.DataFrame(importance_values)

    summary = (
        importance_df.groupby("feature")
        .agg(
            overall_importance=("overall_importance", "mean"),
            posture_importance=("posture_importance", "mean"),
        )
        .sort_values("overall_importance", ascending=False)
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    summary.to_csv(OUTPUT_DIR / "feature_importance.csv")
    pd.DataFrame(fold_scores).to_csv(
        OUTPUT_DIR / "feature_analysis_folds.csv",
        index=False,
    )

    print("\nTOP 20 FEATURES FOR OVERALL CLASSIFICATION")
    print(summary.head(20).to_string())

    print("\nTOP 20 FEATURES FOR SITTING/STANDING")
    print(
        summary.sort_values(
            "posture_importance",
            ascending=False,
        ).head(20).to_string()
    )

    print("\nSaved:")
    print("report/feature_importance.csv")
    print("report/feature_analysis_folds.csv")


if __name__ == "__main__":
    main()
