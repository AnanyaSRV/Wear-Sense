
from pathlib import Path
import json
import time

import pandas as pd
from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def evaluate(model, X, y):
    start = time.perf_counter()
    predictions = model.predict(X)
    elapsed = time.perf_counter() - start

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(
            f1_score(y, predictions, average="macro")
        ),
        "predictions": predictions,
        "inference_seconds": elapsed,
    }


def main():
    X_train, y_train = load_data("train")
    X_test, y_test = load_data("test")

    # Reference model: most frequent training class
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    # First learned model
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    logistic.fit(X_train, y_train)

    results = {}

    for name, model in [
        ("dummy", dummy),
        ("logistic_regression", logistic),
    ]:
        metrics = evaluate(model, X_test, y_test)

        print(f"\n{'=' * 50}")
        print(name.upper())
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(
            f"Inference time: "
            f"{metrics['inference_seconds']:.4f} seconds"
        )

        print("\nClassification report:")
        print(classification_report(
            y_test,
            metrics["predictions"],
            zero_division=0
        ))

        results[name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "inference_seconds": metrics["inference_seconds"],
        }

        if name == "logistic_regression":
            dump(model, MODEL_DIR / "logistic_regression.joblib")

    with open(
        REPORT_DIR / "baseline_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print("\nSaved Logistic Regression model to models/")
    print("Saved metrics to report/baseline_results.json")


if __name__ == "__main__":
    main()
