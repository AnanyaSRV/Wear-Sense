
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ROOT = Path("data") / "UCI HAR Dataset"
MODEL_DIR = Path("models")
REPORT_DIR = Path("report")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data():
    folder = ROOT / "train"

    X = pd.read_csv(
        folder / "X_train.txt",
        sep=r"\s+",
        header=None
    )

    y = pd.read_csv(
        folder / "y_train.txt",
        sep=r"\s+",
        header=None
    ).iloc[:, 0]

    return X, y


def main():
    X_train, y_train = load_training_data()

    # All model selection happens using training data only.
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    candidates = {
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=4000,
                    random_state=42
                ))
            ]),
            "parameters": {
                "classifier__C": [0.1, 1.0, 3.0, 10.0]
            },
            "path": MODEL_DIR / "logistic_regression.joblib"
        },

        "RBF SVM": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(
                    kernel="rbf",
                    probability=False,
                    cache_size=1000
                ))
            ]),
            "parameters": {
                "classifier__C": [1, 3, 10],
                "classifier__gamma": [
                    "scale", 0.001, 0.01
                ]
            },
            "path": MODEL_DIR / "rbf_svm.joblib"
        }
    }

    all_results = []

    for name, config in candidates.items():
        print("\n" + "=" * 65)
        print(f"TUNING: {name}")
        print("Using stratified 3-fold CV on training data only.")

        search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["parameters"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=2,
            return_train_score=True
        )

        search.fit(X_train, y_train)

        # Refit best estimator is trained on all training data.
        dump(search.best_estimator_, config["path"])

        print(f"\nBest CV macro-F1: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        print(f"Saved model: {config['path']}")

        for i, params in enumerate(search.cv_results_["params"]):
            all_results.append({
                "model": name,
                "parameters": str(params),
                "mean_cv_macro_f1":
                    search.cv_results_["mean_test_score"][i],
                "std_cv_macro_f1":
                    search.cv_results_["std_test_score"][i],
                "mean_train_macro_f1":
                    search.cv_results_["mean_train_score"][i]
            })

    results = pd.DataFrame(all_results)
    results = results.sort_values(
        "mean_cv_macro_f1",
        ascending=False
    )

    results.to_csv(
        REPORT_DIR / "hyperparameter_tuning.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("TOP 10 CONFIGURATIONS")
    print(results.head(10).to_string(index=False))
    print("\nSaved: report/hyperparameter_tuning.csv")
    print("Tuning complete. The held-out test set was not loaded.")


if __name__ == "__main__":
    main()
