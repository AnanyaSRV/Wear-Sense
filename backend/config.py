
from pathlib import Path

# Project root: C:\Users\srvan\WearSense
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "rbf_svm_grouped_best.joblib"

DATASET_PATH = PROJECT_ROOT / "data" / "UCI HAR Dataset"

X_TEST_PATH = DATASET_PATH / "test" / "X_test.txt"
Y_TEST_PATH = DATASET_PATH / "test" / "y_test.txt"
SUBJECT_TEST_PATH = DATASET_PATH / "test" / "subject_test.txt"

ACTIVITY_LABELS_PATH = DATASET_PATH / "activity_labels.txt"

EXPECTED_FEATURE_COUNT = 561

ACTIVITY_LABELS = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}

FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
