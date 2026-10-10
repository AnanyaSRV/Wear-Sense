
from pathlib import Path
import pandas as pd

REPORT_DIR = Path("report")

FILES_TO_CHECK = [
    REPORT_DIR / "svm_vs_extra_trees.csv",
    REPORT_DIR / "svm_vs_extra_trees_by_activity.csv",
    REPORT_DIR / "best_svm" / "metrics.csv",
    REPORT_DIR / "best_svm" / "predictions.csv",
    REPORT_DIR / "best_svm" / "confusion_matrix.png",
    REPORT_DIR / "feature_importance.csv",
    REPORT_DIR / "evaluation_metrics.csv",
]

def main():
    print("=" * 60)
    print("WEARSENSE — FINAL EVALUATION FILE CHECK")
    print("=" * 60)

    missing = []

    for path in FILES_TO_CHECK:
        if path.exists():
            print(f"[OK]      {path}")
        else:
            print(f"[MISSING] {path}")
            missing.append(path)

    if missing:
        print("\nSome expected files are missing.")
        print("We will resolve missing files before finalizing the report.")
        return

    print("\n" + "=" * 60)
    print("SELECTED MODEL CANDIDATE")
    print("=" * 60)

    comparison = pd.read_csv(
        REPORT_DIR / "svm_vs_extra_trees.csv"
    )
    print(comparison.to_string(index=False))

    print("\n" + "=" * 60)
    print("OFFICIAL TEST-SET RESULTS")
    print("=" * 60)

    test_metrics = pd.read_csv(
        REPORT_DIR / "best_svm" / "metrics.csv"
    )
    print(test_metrics.to_string(index=False))

    print("\n" + "=" * 60)
    print("EVALUATION CHECK COMPLETE")
    print("=" * 60)
    print(
        "Note: the official test set has been inspected during "
        "development. Report its results as exploratory, not as "
        "an untouched final evaluation."
    )

if __name__ == "__main__":
    main()
