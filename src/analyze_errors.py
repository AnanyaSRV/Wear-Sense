
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix

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
    folder = ROOT / "test"

    X_test = pd.read_csv(
        folder / "X_test.txt",
        sep=r"\s+",
        header=None,
    )

    y_test = pd.read_csv(
        folder / "y_test.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0]

    model = load(MODEL_DIR / "rbf_svm.joblib")
    predictions = model.predict(X_test)

    labels = [1, 2, 3, 4, 5, 6]
    cm = confusion_matrix(y_test, predictions, labels=labels)

    matrix = pd.DataFrame(
        cm,
        index=[f"Actual: {ACTIVITIES[i]}" for i in labels],
        columns=[f"Predicted: {ACTIVITIES[i]}" for i in labels],
    )

    print("\nCONFUSION MATRIX")
    print(matrix.to_string())

    print("\nSITTING / STANDING ERRORS")

    mask = (
        ((y_test == 4) & (predictions == 5))
        | ((y_test == 5) & (predictions == 4))
    )

    errors = pd.DataFrame({
        "actual": y_test[mask].map(ACTIVITIES),
        "predicted": pd.Series(
            predictions, index=y_test.index
        )[mask].map(ACTIVITIES),
    })

    print(errors.value_counts().to_string())
    print(f"\nTotal sitting/standing confusions: {mask.sum()}")

    output = Path("report") / "sitting_standing_errors.csv"
    errors.to_csv(output, index_label="test_row")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
