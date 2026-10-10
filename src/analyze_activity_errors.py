
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix

ROOT = Path("data") / "UCI HAR Dataset"
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

    X = pd.read_csv(
        test_dir / "X_test.txt",
        sep=r"\s+",
        header=None,
    )
    y = pd.read_csv(
        test_dir / "y_test.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0]
    subjects = pd.read_csv(
        test_dir / "subject_test.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0]

    model = load(Path("models") / "rbf_svm.joblib")
    predictions = pd.Series(model.predict(X))

    results = pd.DataFrame({
        "subject": subjects,
        "actual": y,
        "predicted": predictions,
    })

    results["correct"] = results["actual"] == results["predicted"]
    results["actual_activity"] = results["actual"].map(ACTIVITIES)

    # Error rate for each subject and actual activity.
    activity_report = (
        results.groupby(["subject", "actual_activity"])
        .agg(
            samples=("correct", "size"),
            correct=("correct", "sum"),
        )
        .reset_index()
    )
    activity_report["errors"] = (
        activity_report["samples"] - activity_report["correct"]
    )
    activity_report["accuracy"] = (
        activity_report["correct"] / activity_report["samples"]
    )

    # Show the weakest activity groups first.
    activity_report = activity_report.sort_values(
        ["accuracy", "errors"],
        ascending=[True, False],
    )

    print("\nLOWEST-ACCURACY SUBJECT/ACTIVITY GROUPS")
    print(
        activity_report.head(20).to_string(
            index=False,
            formatters={"accuracy": lambda x: f"{x:.2%}"},
        )
    )

    # Exact mistakes for sitting/standing.
    errors = results[
        (results["actual"] != results["predicted"])
        & results["actual"].isin([4, 5])
    ].copy()

    errors["actual_activity"] = errors["actual"].map(ACTIVITIES)
    errors["predicted_activity"] = errors["predicted"].map(ACTIVITIES)

    print("\nSITTING/STANDING MISCLASSIFICATIONS BY SUBJECT")
    print(
        errors.groupby(
            ["subject", "actual_activity", "predicted_activity"]
        ).size().to_string()
    )

    out = Path("report")
    out.mkdir(exist_ok=True)
    activity_report.to_csv(
        out / "subject_activity_performance.csv",
        index=False,
    )
    errors.to_csv(
        out / "sitting_standing_misclassifications.csv",
        index=False,
    )

    print("\nSaved:")
    print("report/subject_activity_performance.csv")
    print("report/sitting_standing_misclassifications.csv")


if __name__ == "__main__":
    main()
