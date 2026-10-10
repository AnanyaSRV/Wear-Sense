
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_DIR = Path("data") / "UCI HAR Dataset" / "train"
MODEL_DIR = Path("models")


def main():
    X_train = pd.read_csv(
        DATA_DIR / "X_train.txt",
        sep=r"\s+",
        header=None,
    )

    y_train = pd.read_csv(
        DATA_DIR / "y_train.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=3.0,
            gamma=0.001,
            probability=True,
            random_state=42,
        )),
    ])

    print("Training selected RBF SVM on the full training set...")
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(exist_ok=True)
    output_path = MODEL_DIR / "rbf_svm_grouped_best.joblib"
    dump(model, output_path)

    print("Model saved:", output_path)
    print("Training samples:", len(X_train))
    print("Features:", X_train.shape[1])
    print("Configuration: RBF SVM, C=3.0, gamma=0.001")


if __name__ == "__main__":
    main()
