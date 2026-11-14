
import numpy as np

from backend.config import (
    X_TEST_PATH,
    Y_TEST_PATH,
    SUBJECT_TEST_PATH,
    EXPECTED_FEATURE_COUNT,
    ACTIVITY_LABELS,
)


class DatasetService:
    def __init__(self):
        self.features = None
        self.labels = None
        self.subjects = None

        self.load_dataset()

    def load_dataset(self):
        required_files = [
            X_TEST_PATH,
            Y_TEST_PATH,
            SUBJECT_TEST_PATH,
        ]

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Dataset file not found: {file_path}"
                )

        self.features = np.loadtxt(
            X_TEST_PATH,
            dtype=np.float64,
        )

        self.labels = np.loadtxt(
            Y_TEST_PATH,
            dtype=np.int64,
        ).reshape(-1)

        self.subjects = np.loadtxt(
            SUBJECT_TEST_PATH,
            dtype=np.int64,
        ).reshape(-1)

        if self.features.ndim != 2:
            raise ValueError(
                "The test feature dataset must be two-dimensional."
            )

        if self.features.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_FEATURE_COUNT} dataset features, "
                f"found {self.features.shape[1]}."
            )

        if not (
            len(self.features)
            == len(self.labels)
            == len(self.subjects)
        ):
            raise ValueError(
                "Feature, label, and subject counts do not match."
            )

    def get_dataset_info(self):
        return {
            "dataset_name": "UCI Human Activity Recognition Using Smartphones",
            "split": "test",
            "sample_count": int(len(self.features)),
            "feature_count": int(self.features.shape[1]),
            "activity_count": int(len(np.unique(self.labels))),
            "subject_count": int(len(np.unique(self.subjects))),
            "data_source": "UCI HAR test dataset",
            "is_live_sensor_data": False,
        }

    def get_samples(self, start=0, limit=20):
        total = len(self.features)

        if start < 0:
            raise ValueError("Start index cannot be negative.")

        if limit < 1 or limit > 100:
            raise ValueError(
                "Limit must be between 1 and 100."
            )

        if start >= total:
            return {
                "total_samples": total,
                "start": start,
                "returned": 0,
                "samples": [],
            }

        end = min(start + limit, total)

        samples = []

        for index in range(start, end):
            actual_label = int(self.labels[index])

            samples.append({
                "sample_index": index,
                "actual_activity_id": actual_label,
                "actual_activity": ACTIVITY_LABELS.get(
                    actual_label,
                    f"Unknown activity {actual_label}",
                ),
                "subject_id": int(self.subjects[index]),
            })

        return {
            "total_samples": total,
            "start": start,
            "returned": len(samples),
            "samples": samples,
        }

    def get_sample(self, index):
        if index < 0 or index >= len(self.features):
            raise IndexError(
                f"Sample index must be between 0 and "
                f"{len(self.features) - 1}."
            )

        actual_label = int(self.labels[index])

        return {
            "sample_index": index,
            "features": self.features[index].tolist(),
            "actual_activity_id": actual_label,
            "actual_activity": ACTIVITY_LABELS.get(
                actual_label,
                f"Unknown activity {actual_label}",
            ),
            "subject_id": int(self.subjects[index]),
        }


dataset_service = DatasetService()
