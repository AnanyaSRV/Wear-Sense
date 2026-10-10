
from pathlib import Path
import pandas as pd

ROOT = Path("data") / "UCI HAR Dataset"


def load_split(split):
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

    subjects = pd.read_csv(
        folder / f"subject_{split}.txt",
        sep=r"\s+",
        header=None
    ).iloc[:, 0]

    return X, y, subjects


def main():
    X_train, y_train, subjects_train = load_split("train")
    X_test, y_test, subjects_test = load_split("test")

    activity_file = ROOT / "activity_labels.txt"
    activities = pd.read_csv(
        activity_file,
        sep=r"\s+",
        header=None,
        names=["activity_id", "activity_name"]
    )

    print("=" * 55)
    print("WEARSENSE - DATASET VALIDATION")
    print("=" * 55)

    print("\nTraining feature shape:", X_train.shape)
    print("Testing feature shape: ", X_test.shape)
    print("Number of features:    ", X_train.shape[1])

    print("\nTraining labels:", len(y_train))
    print("Testing labels: ", len(y_test))

    print("\nMissing values in training:", X_train.isna().sum().sum())
    print("Missing values in testing: ", X_test.isna().sum().sum())

    print("\nTraining subjects:", subjects_train.nunique())
    print("Testing subjects: ", subjects_test.nunique())

    overlap = set(subjects_train) & set(subjects_test)
    print("Overlapping subjects:", sorted(overlap))

    print("\nTraining activity counts:")
    print(y_train.value_counts().sort_index())

    print("\nTesting activity counts:")
    print(y_test.value_counts().sort_index())

    print("\nActivity label mapping:")
    print(activities.to_string(index=False))

    assert X_train.shape[1] == X_test.shape[1], (
        "Training and testing feature counts differ!"
    )
    assert len(X_train) == len(y_train) == len(subjects_train)
    assert len(X_test) == len(y_test) == len(subjects_test)
    assert not overlap, "Participant leakage detected!"
    assert set(y_train.unique()) == set(activities["activity_id"])
    assert set(y_test.unique()) == set(activities["activity_id"])

    print("\nSUCCESS: Dataset validation passed!")


if __name__ == "__main__":
    main()
