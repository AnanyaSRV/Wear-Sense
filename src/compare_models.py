
from pathlib import Path
import json
import time

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path("data") / "UCI HAR Dataset"
MODEL_DIR = Path("models")
REPORT_DIR = Path("report")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def load_data(split):
    folder = ROOT / split

    X = pd.read_csv(
        folder / f"X_{split}.txt",
        sep=r"\s+",
        header=None
    )

    y = pd.read_csv(
        folder / f"y_{split}.txt",
        sep=r"\s+",
        header=None
    ).iloc[:, 0]

    return X, y


def main():
    X_train, y_train = load_data("train")
    X_test, y_test = load_data("test")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=10,
                gamma="scale"
            ))
        ])
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(X_test)
        predict_seconds = time.perf_counter() - start

        metrics = {
            "model": name,
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "macro_precision": float(
                precision_score(
                    y_test, predictions,
                    average="macro", zero_division=0
                )
            ),
            "macro_recall": float(
                recall_score(
                    y_test, predictions,
                    average="macro", zero_division=0
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_test, predictions,
                    average="macro", zero_division=0
                )
            ),
            "train_seconds": train_seconds,
            "test_batch_seconds": predict_seconds
        }

        results.append(metrics)

        print(f"\n{name} results")
        for key, value in metrics.items():
            if key != "model":
                print(f"{key}: {value:.4f}")

        print("\nClassification report:")
        print(classification_report(
            y_test, predictions, zero_division=0
        ))

        filename = name.lower().replace(" ", "_")
        dump(model, MODEL_DIR / f"{filename}.joblib")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    print("\nFinal comparison:")
    print(
        results_df[
            ["model", "accuracy", "macro_f1",
             "train_seconds", "test_batch_seconds"]
        ].to_string(index=False)
    )

    print("\nSaved comparison to report/model_comparison.csv")


if __name__ == "__main__":
    main()
