
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path("data") / "UCI HAR Dataset"
MODEL_DIR = Path("models")

ACTIVITIES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}


def main():
    test_dir = ROOT / "test"

    X_test = pd.read_csv(
        test_dir / "X_test.txt",
        sep=r"\s+",
        header=None,
    )

    y_test = pd.read_csv(
        test_dir / "y_test.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0]

    models = {
        "Logistic Regression": "logistic_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "RBF SVM": "rbf_svm.joblib",
    }

    rows = []

    for model_name, filename in models.items():
        model = load(MODEL_DIR / filename)
        predictions = model.predict(X_test)

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_test,
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

    print("\nRECALL BY MODEL AND ACTIVITY")
    print(
        results.pivot(
            index="activity",
            columns="model",
            values="recall",
        ).to_string(
            float_format=lambda value: f"{value:.3f}"
        )
    )


    print("\nF1 SCORE BY MODEL AND ACTIVITY")
    print(
        results.pivot(
            index="activity",
            columns="model",
            values="f1",
        ).to_string(
            float_format=lambda value: f"{value:.3f}"
        )
    )

    # NEW: Calculate overall metrics
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
    )

    print("\nOVERALL MODEL METRICS")

    for model_name, filename in models.items():
        model = load(MODEL_DIR / filename)
        predictions = model.predict(X_test)

        print(
            f"{model_name:22s} | "
            f"Accuracy: {accuracy_score(y_test, predictions):.4f} | "
            f"Balanced accuracy: "
            f"{balanced_accuracy_score(y_test, predictions):.4f} | "
            f"Macro-F1: "
            f"{f1_score(y_test, predictions, average='macro'):.4f}"
        )

    # EXISTING: Keep the CSV-saving code
    output = Path("report") / "model_activity_comparison.csv"
    output.parent.mkdir(exist_ok=True)
    results.to_csv(output, index=False)

    print(f"\nSaved: {output}")
if __name__ == "__main__":
    main()
