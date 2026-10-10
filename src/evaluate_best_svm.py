
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

ROOT = Path("data") / "UCI HAR Dataset"
MODEL_PATH = Path("models") / "rbf_svm_grouped_best.joblib"
OUT = Path("report") / "best_svm"

ACTIVITIES = [
    "Walking",
    "Walking upstairs",
    "Walking downstairs",
    "Sitting",
    "Standing",
    "Laying",
]


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

    model = load(MODEL_PATH)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )
    balanced_acc = balanced_accuracy_score(
        y_test, predictions
    )

    print("\nBEST SVM — HELD-OUT TEST RESULTS")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        labels=[1, 2, 3, 4, 5, 6],
        target_names=ACTIVITIES,
        digits=4,
        zero_division=0,
    ))

    OUT.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame([{
        "model": "RBF SVM (grouped-CV selected)",
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "test_samples": len(y_test),
    }])
    metrics.to_csv(OUT / "metrics.csv", index=False)

    pd.DataFrame({
        "actual": y_test,
        "predicted": predictions,
    }).to_csv(OUT / "predictions.csv", index=False)

    cm = confusion_matrix(
        y_test, predictions, labels=[1, 2, 3, 4, 5, 6]
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=ACTIVITIES,
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=35)
    ax.set_title("WearSense — Selected RBF SVM Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("\nSaved evaluation files in:", OUT)


if __name__ == "__main__":
    main()
