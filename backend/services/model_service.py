
import joblib
import numpy as np

from backend.config import (
    MODEL_PATH,
    EXPECTED_FEATURE_COUNT,
    ACTIVITY_LABELS,
)


class ModelService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

        if not hasattr(self.model, "n_features_in_"):
            raise ValueError(
                "Loaded model does not expose n_features_in_."
            )

        if self.model.n_features_in_ != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_FEATURE_COUNT} features, "
                f"but model expects {self.model.n_features_in_}."
            )

    def get_model_info(self):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        classes = self.model.classes_

        return {
            "model_name": MODEL_PATH.name,
            "model_type": type(self.model).__name__,
            "feature_count": int(self.model.n_features_in_),
            "classes": [
                {
                    "id": int(class_id),
                    "name": ACTIVITY_LABELS.get(
                        int(class_id),
                        f"Unknown activity {class_id}",
                    ),
                }
                for class_id in classes
            ],
            "probability_available": hasattr(
                self.model,
                "predict_proba",
            ),
        }

    def predict(self, features):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        feature_array = np.asarray(
            features,
            dtype=np.float64,
        )

        if feature_array.ndim != 1:
            raise ValueError(
                "Features must be a one-dimensional list."
            )

        if feature_array.shape[0] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_FEATURE_COUNT} features, "
                f"received {feature_array.shape[0]}."
            )

        if not np.isfinite(feature_array).all():
            raise ValueError(
                "Features contain NaN or infinite values."
            )

        input_data = feature_array.reshape(1, -1)

        predicted_class = int(
            self.model.predict(input_data)[0]
        )

        probabilities = None

        if hasattr(self.model, "predict_proba"):
            probability_values = self.model.predict_proba(
                input_data
            )[0]

            probabilities = [
                {
                    "activity_id": int(class_id),
                    "activity": ACTIVITY_LABELS.get(
                        int(class_id),
                        f"Unknown activity {class_id}",
                    ),
                    "probability": round(float(probability), 6),
                }
                for class_id, probability in zip(
                    self.model.classes_,
                    probability_values,
                )
            ]

            probabilities.sort(
                key=lambda item: item["probability"],
                reverse=True,
            )

        return {
            "predicted_activity_id": predicted_class,
            "predicted_activity": ACTIVITY_LABELS.get(
                predicted_class,
                f"Unknown activity {predicted_class}",
            ),
            "probabilities": probabilities,
            "probability_note": (
                "Model probabilities are not calibrated "
                "certainty estimates."
            ),
        }


model_service = ModelService()
