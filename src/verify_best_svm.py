
from pathlib import Path

import pandas as pd
from joblib import load

MODEL_PATH = Path("models") / "rbf_svm_grouped_best.joblib"
DATA_DIR = Path("data") / "UCI HAR Dataset" / "train"

ACTIVITIES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}


def main():
    print("Loading saved model...")
    model = load(MODEL_PATH)

    X = pd.read_csv(
        DATA_DIR / "X_train.txt",
        sep=r"\s+",
        header=None,
    )

    y = pd.read_csv(
        DATA_DIR / "y_train.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0]

    # Use a few samples to verify model behavior.
    X_sample = X.iloc[:5]
    y_sample = y.iloc[:5]

    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)

    print("\nMODEL VERIFICATION")
    print("Model file exists:", MODEL_PATH.exists())
    print("Feature count:", X.shape[1])
    print("Probability output shape:", probabilities.shape)
    print("Probability rows sum to 1:",
          (abs(probabilities.sum(axis=1) - 1) < 1e-6).all())

    print("\nSAMPLE PREDICTIONS")
    for i, (actual, predicted) in enumerate(
        zip(y_sample, predictions), start=1
    ):
        print(
            f"Sample {i}: "
            f"Actual={ACTIVITIES[int(actual)]}, "
            f"Predicted={ACTIVITIES[int(predicted)]}, "
            f"Top probability={probabilities[i-1].max():.4f}"
        )

    print("\nVerification completed.")


if __name__ == "__main__":
    main()
