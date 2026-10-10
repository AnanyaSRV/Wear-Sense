
from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from joblib import load

ROOT = Path("data") / "UCI HAR Dataset"
MODEL_PATH = Path("models") / "logistic_regression.joblib"
OUTPUT = Path("report") / "feature_selection_results.csv"


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

    model_template = load(MODEL_PATH)
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    feature_counts = [561, 100, 50, 25]
    rows = []

    for k in feature_counts:
        print(f"\nTesting top {k} features...")

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups=groups),
            start=1,
        ):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            # Feature selection is fitted only on the training fold.
            if k < X.shape[1]:
                selector = SelectKBest(
                    score_func=f_classif,
                    k=k,
                )
                X_train_selected = selector.fit_transform(
                    X_train, y_train
                )
                X_val_selected = selector.transform(X_val)
            else:
                X_train_selected = X_train
                X_val_selected = X_val

            model = clone(model_template)
            model.fit(X_train_selected, y_train)
            predictions = model.predict(X_val_selected)

            rows.append({
                "features": k,
                "fold": fold,
                "macro_f1": f1_score(
                    y_val,
                    predictions,
                    average="macro",
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_val,
                    predictions,
                ),
            })

            print(
                f"  Fold {fold}: "
                f"macro-F1="
                f"{rows[-1]['macro_f1']:.4f}"
            )

    results = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    summary = results.groupby("features").agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
    )

    print("\nFEATURE SELECTION SUMMARY")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
