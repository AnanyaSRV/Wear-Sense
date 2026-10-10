
from pathlib import Path

import numpy as np
import pandas as pd

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
OUTPUT_DIR.mkdir(exist_ok=True)

ACTIVITY_NAMES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}


CONFIGURATIONS = [
    {
        "name": "Baseline (C=3, gamma=0.001)",
        "C": 3.0,
        "gamma": 0.001,
        "class_weight": None,
    },
    {
        "name": "Class-weighted (C=3, gamma=0.001)",
        "C": 3.0,
        "gamma": 0.001,
        "class_weight": "balanced",
    },
]



def load_training_data():
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


def main():
    X, y, groups = load_training_data()

    splitter = GroupKFold(n_splits=5)
    all_results = []

    print("WearSense — Grouped Posture Experiment")
    print(f"Training samples: {len(X)}")
    print(f"Unique training subjects: {len(np.unique(groups))}")
    print("Validation: 5-fold GroupKFold")
    print("Official test set: not used")
    print()

    for config in CONFIGURATIONS:
        print(f"Evaluating: {config['name']}")

        oof_actual = []
        oof_predicted = []

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(X, y, groups),
            start=1,
        ):
            model = make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    C=config["C"],
                    gamma=config["gamma"],
                    class_weight=config["class_weight"],
                ),
            )

            model.fit(X.iloc[train_idx], y[train_idx])
            predictions = model.predict(X.iloc[val_idx])

            oof_actual.extend(y[val_idx])
            oof_predicted.extend(predictions)

            print(
                f"  Fold {fold}: "
                f"macro-F1={f1_score(y[val_idx], predictions, average='macro'):.4f}"
            )

        actual = np.asarray(oof_actual)
        predicted = np.asarray(oof_predicted)

        recalls = recall_score(
            actual,
            predicted,
            labels=[1, 2, 3, 4, 5, 6],
            average=None,
            zero_division=0,
        )

        metrics = {
            "model": config["name"],
            "accuracy": accuracy_score(actual, predicted),
            "balanced_accuracy": balanced_accuracy_score(
                actual, predicted
            ),
            "macro_f1": f1_score(
                actual, predicted, average="macro"
            ),
            "sitting_recall": recalls[3],
            "standing_recall": recalls[4],
        }

        all_results.append(metrics)
        print(
            f"  Overall grouped macro-F1: {metrics['macro_f1']:.4f}"
        )
        print()

    results = pd.DataFrame(all_results).sort_values(
        "macro_f1",
        ascending=False,
    )

    output_path = OUTPUT_DIR / "posture_experiment_results.csv"
    results.to_csv(output_path, index=False)

    print("FINAL COMPARISON")
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
