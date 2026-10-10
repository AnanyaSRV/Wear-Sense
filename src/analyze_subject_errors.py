
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score

ROOT = Path("data") / "UCI HAR Dataset"
MODEL_PATH = Path("models") / "rbf_svm.joblib"

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

    subjects = pd.read_csv(
        test_dir / "subject_test.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0]

    model = load(MODEL_PATH)
    predictions = pd.Series(model.predict(X_test))

    results = pd.DataFrame({
        "subject": subjects,
        "actual": y_test,
        "predicted": predictions,
    })

    results["actual_name"] = results["actual"].map(ACTIVITIES)
    results["predicted_name"] = results["predicted"].map(ACTIVITIES)
    results["correct"] = results["actual"] == results["predicted"]

    # Evaluate each participant separately.
    summary = (
        results.groupby("subject")
        .agg(
            samples=("correct", "size"),
            accuracy=("correct", "mean"),
            errors=("correct", lambda values: (~values).sum()),
        )
        .sort_values("accuracy")
    )

    print("\nPER-SUBJECT MODEL PERFORMANCE")
    print(summary.to_string(
        formatters={"accuracy": lambda value: f"{value:.2%}"}
    ))

    # Focus on sitting and standing.
    mask = results["actual"].isin([4, 5])
    sitting_standing = results[mask]

    subject_confusions = (
        sitting_standing.assign(
            confusion=(
                sitting_standing["actual"].astype(str)
                + " -> "
                + sitting_standing["predicted"].astype(str)
            )
        )
        .groupby(["subject", "confusion"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nSITTING/STANDING ERRORS BY SUBJECT")
    print(subject_confusions.to_string())

    output_dir = Path("report")
    output_dir.mkdir(exist_ok=True)

    summary.to_csv(output_dir / "subject_performance.csv")
    subject_confusions.to_csv(
        output_dir / "sitting_standing_by_subject.csv"
    )

    print("\nSaved:")
    print("report/subject_performance.csv")
    print("report/sitting_standing_by_subject.csv")


if __name__ == "__main__":
    main()
