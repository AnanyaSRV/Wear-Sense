
from pathlib import Path
import joblib
import traceback

MODEL_DIR = Path(__file__).parent / "models"

def inspect_object(obj, indent=0):
    prefix = " " * indent

    print(f"{prefix}Class: {type(obj).__name__}")
    print(f"{prefix}Module: {type(obj).__module__}")

    if hasattr(obj, "n_features_in_"):
        print(f"{prefix}Input features: {obj.n_features_in_}")

    if hasattr(obj, "feature_names_in_"):
        print(f"{prefix}Feature names: {len(obj.feature_names_in_)}")
        print(f"{prefix}First feature names:")
        for name in list(obj.feature_names_in_)[:10]:
            print(f"{prefix}  - {name}")

    if hasattr(obj, "classes_"):
        print(f"{prefix}Classes: {obj.classes_}")

    if hasattr(obj, "steps"):
        print(f"{prefix}Pipeline steps:")
        for name, step in obj.steps:
            print(f"{prefix}  Step: {name}")
            inspect_object(step, indent + 4)

    if hasattr(obj, "named_steps"):
        print(f"{prefix}Named steps: {list(obj.named_steps.keys())}")

    if hasattr(obj, "get_params"):
        try:
            params = obj.get_params(deep=False)
            print(f"{prefix}Parameters:")
            for key, value in params.items():
                print(f"{prefix}  {key}: {value}")
        except Exception:
            pass


def main():
    if not MODEL_DIR.exists():
        print("Models directory not found:", MODEL_DIR)
        return

    model_files = sorted(MODEL_DIR.glob("*.joblib"))

    if not model_files:
        print("No .joblib models found.")
        return

    print("=" * 70)
    print("WEARSENSE MODEL INSPECTION")
    print("=" * 70)

    for model_path in model_files:
        print("\n" + "=" * 70)
        print("MODEL:", model_path.name)
        print("=" * 70)

        try:
            model = joblib.load(model_path)
            inspect_object(model)

        except Exception as error:
            print("Could not load model:", error)
            traceback.print_exc()

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
