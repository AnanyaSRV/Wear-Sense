
from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score

ROOT = Path("data") / "UCI HAR Dataset"
OUTPUT = Path("report") / "grouped_tuning_results.csv"


def main():
    train_dir = ROOT / "train"

    X = pd.read_csv(
        train_dir / "X_train.txt",
        sep=r"\s+",
        header=None,
    )

    y = pd.read_csv(
        train_dir / "y_train.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0].to_numpy()

    groups = pd.read_csv(
        train_dir / "subject_train.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0].to_numpy()

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    models = {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=4000)),
            ]),
            {
                "model__C": [0.1, 1.0, 3.0, 10.0],
            },
        ),
        "RBF SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf")),
            ]),
            {
                "model__C": [1.0, 3.0, 10.0],
                "model__gamma": ["scale", 0.001, 0.01],
            },
        ),
    }

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scorer = make_scorer(
        f1_score,
        average="macro",
    )

    rows = []

    for name, (pipeline, params) in models.items():
        print(f"\nTuning {name}...")

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring=scorer,
            cv=cv,
            refit=False,
            n_jobs=-1,
            return_train_score=False,
        )

        search.fit(X, y, groups=groups)

        results = pd.DataFrame(search.cv_results_)

        for _, result in results.iterrows():
            rows.append({
                "model": name,
                "parameters": str(result["params"]),
                "mean_macro_f1": result["mean_test_score"],
                "std_macro_f1": result["std_test_score"],
            })

        best_index = results["mean_test_score"].idxmax()
        best = results.loc[best_index]

        print("Best parameters:", best["params"])
        print(f"Mean grouped CV macro-F1: {best['mean_test_score']:.4f}")
        print(f"Fold standard deviation: {best['std_test_score']:.4f}")

    output_df = pd.DataFrame(rows).sort_values(
        ["model", "mean_macro_f1"],
        ascending=[True, False],
    )

    OUTPUT.parent.mkdir(exist_ok=True)
    output_df.to_csv(OUTPUT, index=False)

    print("\nTOP CONFIGURATIONS")
    print(
        output_df.groupby("model", group_keys=False)
        .head(5)
        .to_string(index=False)
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
