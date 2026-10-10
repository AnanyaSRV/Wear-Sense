
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

ROOT = Path("data") / "UCI HAR Dataset"
MODEL_DIR = Path("models")
OUTPUT = Path("report") / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

ACTIVITY_NAMES = [
    "Walking",
    "Walking upstairs",
    "Walking downstairs",
    "Sitting",
    "Standing",
    "Laying",
]


def load_test_data():
    folder = ROOT / "test"

    X = pd.read_csv(
        folder / "X_test.txt",
        sep=r"\s+",
        header=None
    )
    y = pd.read_csv(
        folder / "y_test.txt",
        sep=r"\s+",
        header=None
    ).iloc[:, 0]

    return X, y


def main():
    X_test, y_test = load_test_data()

    models = {
        "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
        "Random Forest": MODEL_DIR / "random_forest.joblib",
        "RBF SVM": MODEL_DIR / "rbf_svm.joblib",
    }

    rows = []
    predictions = pd.DataFrame({
        "true_activity": y_test
    })

    for name, model_path in models.items():
        if not model_path.exists():
            print(f"Skipping missing model: {model_path}")
            continue

        model = load(model_path)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )

        rows.append({
            "model": name,
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
        })

        predictions[name] = y_pred

        cm = confusion_matrix(
            y_test, y_pred, labels=[1, 2, 3, 4, 5, 6]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=ACTIVITY_NAMES
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        display.plot(
            ax=ax,
            cmap="Blues",
            xticks_rotation=35,
            values_format="d",
            colorbar=False
        )
        ax.set_title(f"WearSense - {name} Confusion Matrix")
        fig.tight_layout()

        filename = name.lower().replace(" ", "_")
        fig.savefig(
            OUTPUT / f"{filename}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

        print(f"\n{name}")
        print(classification_report(
            y_test,
            y_pred,
            labels=[1, 2, 3, 4, 5, 6],
            target_names=ACTIVITY_NAMES,
            zero_division=0
        ))

    results = pd.DataFrame(rows)
    results.to_csv(
        "report/evaluation_metrics.csv",
        index=False
    )
    predictions.to_csv(
        "report/test_predictions.csv",
        index=False
    )

    if not results.empty:
        metrics = ["accuracy", "macro_precision",
                   "macro_recall", "macro_f1"]

        ax = results.set_index("model")[metrics].plot(
            kind="bar",
            figsize=(11, 6),
            ylim=(0, 1)
        )
        ax.set_title("WearSense Model Performance Comparison")
        ax.set_ylabel("Score")
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(
            OUTPUT / "model_performance_comparison.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    print("\nEvaluation files saved in report/ and report/figures/.")


if __name__ == "__main__":
    main()
