
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_DIR = Path("data") / "UCI HAR Dataset"
OUTPUT_DIR = Path("report")

MODELS = {
    "RBF SVM (C=10, gamma=0.001)": "svm",
    "Extra Trees": "extra_trees",
}

ACTIVITIES = [
    "Walking",
    "Walking upstairs",
    "Walking downstairs",
    "Sitting",
    "Standing",
    "Laying",
]


def load_data():
    train_dir = DATA_DIR / "train"

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

    return X, y, groups


def build_model(model_name):
    if model_name == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=10.0,
                gamma=0.001,
            ),
        )

    return ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight=None,
    )


def main():
    X, y, groups = load_data()

    cv = GroupKFold(n_splits=5)
    results = []
    class_results = []

    print("WearSense: SVM vs Extra Trees")
    print(f"Training samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Participants: {len(np.unique(groups))}")
    print("Validation: 5-fold GroupKFold")
    print("Official test set: not used")

    for display_name, model_key in MODELS.items():
        print(f"\nEvaluating {display_name}")

        oof_actual = np.full(len(y), -1, dtype=int)
        oof_predicted = np.full(len(y), -1, dtype=int)

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups),
            start=1,
        ):
            model = build_model(model_key)

            model.fit(X.iloc[train_idx], y[train_idx])
            predictions = model.predict(X.iloc[val_idx])

            # Every participant must be absent from the fold's
            # training data if they are in its validation data.
            assert set(groups[train_idx]).isdisjoint(
                set(groups[val_idx])
            )

            oof_actual[val_idx] = y[val_idx]
            oof_predicted[val_idx] = predictions

            fold_f1 = f1_score(
                y[val_idx],
                predictions,
                average="macro",
            )

            print(f"  Fold {fold}: macro-F1={fold_f1:.4f}")

        assert np.all(oof_predicted != -1)

        results.append({
            "model": display_name,
            "accuracy": accuracy_score(y, oof_predicted),
            "balanced_accuracy": balanced_accuracy_score(
                y, oof_predicted
            ),
            "macro_f1": f1_score(
                y, oof_predicted, average="macro"
            ),
        })

        recalls = recall_score(
            y,
            oof_predicted,
            labels=[1, 2, 3, 4, 5, 6],
            average=None,
            zero_division=0,
        )

        for i, activity in enumerate(ACTIVITIES):
            class_results.append({
                "model": display_name,
                "activity": activity,
                "recall": recalls[i],
                "f1": f1_score(
                    y,
                    oof_predicted,
                    labels=[i + 1],
                    average="macro",
                    zero_division=0,
                ),
            })

    summary = pd.DataFrame(results).sort_values(
        "macro_f1",
        ascending=False,
    )
    activity_summary = pd.DataFrame(class_results)

    OUTPUT_DIR.mkdir(exist_ok=True)

    summary.to_csv(
        OUTPUT_DIR / "svm_vs_extra_trees.csv",
        index=False,
    )
    activity_summary.to_csv(
        OUTPUT_DIR / "svm_vs_extra_trees_by_activity.csv",
        index=False,
    )

    print("\nOVERALL COMPARISON")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nPER-ACTIVITY RECALL")
    print(
        activity_summary.pivot(
            index="activity",
            columns="model",
            values="recall",
        ).to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nSaved:")
    print("report/svm_vs_extra_trees.csv")
    print("report/svm_vs_extra_trees_by_activity.csv")


if __name__ == "__main__":
    main()
