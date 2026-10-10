
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.base import clone
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path("data") / "UCI HAR Dataset"
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("report")

ACTIVITIES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}

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

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Store each held-out prediction so metrics can be calculated
    # across all out-of-fold predictions for each model.
    all_predictions = {}
    all_actual = []
    all_subjects = []

    for model_name, filename in MODEL_FILES.items():
        print(f"\nEvaluating {model_name}...")

        model = load(MODEL_DIR / filename)
        predictions = pd.Series(index=range(len(y)), dtype="float64")

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups=groups),
            start=1,
        ):
            fold_model = clone(model)
            fold_model.fit(X.iloc[train_idx], y[train_idx])

            fold_predictions = fold_model.predict(X.iloc[val_idx])
            predictions.iloc[val_idx] = fold_predictions

            print(f"  Completed fold {fold}/5")

        all_predictions[model_name] = predictions.to_numpy()

    all_actual = y
    all_subjects = groups

    rows = []

    for model_name, predictions in all_predictions.items():
        precision, recall, f1, support = (
            precision_recall_fscore_support(
                all_actual,
                predictions,
                labels=[1, 2, 3, 4, 5, 6],
                zero_division=0,
            )
        )

        for i, activity_id in enumerate([1, 2, 3, 4, 5, 6]):
            rows.append({
                "model": model_name,
                "activity": ACTIVITIES[activity_id],
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": support[i],
            })

    results = pd.DataFrame(rows)

    print("\nGROUPED OUT-OF-FOLD RECALL")
    print(
        results.pivot(
            index="activity",
            columns="model",
            values="recall",
        ).to_string(float_format=lambda x: f"{x:.3f}")
    )

    print("\nGROUPED OUT-OF-FOLD F1")
    print(
        results.pivot(
            index="activity",
            columns="model",
            values="f1",
        ).to_string(float_format=lambda x: f"{x:.3f}")
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    output = OUTPUT_DIR / "grouped_activity_metrics.csv"
    results.to_csv(output, index=False)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
