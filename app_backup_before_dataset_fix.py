
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path("data") / "UCI HAR Dataset"
MODEL_PATH = Path("models") /"rbf_svm_grouped_best.joblib"
REPORT_PATH = Path("report") / "evaluation_metrics.csv"
FIGURE_PATH = (
    Path("report")
    / "figures"
    / "logistic_regression_confusion_matrix.png"
)

ACTIVITIES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}

ACTIVITY_ICONS = {
    1: "🚶",
    2: "🪜",
    3: "⬇️",
    4: "🪑",
    5: "🧍",
    6: "🛌",
}

st.set_page_config(
    page_title="WearSense | Activity Intelligence",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# Dashboard styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(115deg, #142b4a, #245b76);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.3rem;
        font-size: 2.5rem;
    }

    .hero p {
        color: #e2edf5;
        margin-bottom: 0;
        font-size: 1.05rem;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    .section-note {
        color: #788397;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load model and test dataset
# --------------------------------------------------
@st.cache_resource
def load_model():
    return load(MODEL_PATH)


@st.cache_data
def load_test_data():
    folder = ROOT / "test"

    X = pd.read_csv(
        folder / "X_test.txt",
        sep=r"\s+",
        header=None,
    )

    y = pd.read_csv(
        folder / "y_test.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0]

    return X, y


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>WearSense</h1>
        <p>
            AI-powered human activity recognition
            for intelligent smart activewear.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python src/train.py")
    st.stop()

try:
    X_test, y_test = load_test_data()
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model or test dataset: {exc}")
    st.stop()

if len(X_test) != len(y_test):
    st.error("The test features and labels have different lengths.")
    st.stop()

st.caption(
    "Research prototype using the preprocessed UCI Human Activity "
    "Recognition dataset. Predictions use wearable inertial-sensor "
    "features, not live garment sensor readings."
)

tab1, tab2, tab3 = st.tabs(
    [
        "🏃 Activity Explorer",
        "📊 Model Evaluation",
        "👕 Wearable System Design",
    ]
)


# --------------------------------------------------
# Tab 1: Activity Explorer
# --------------------------------------------------
with tab1:
    st.header("Activity Prediction")
    st.write(
        "Explore individual held-out test samples and inspect "
        "the model's predicted activity and class probabilities."
    )

    sample_index = st.slider(
        "Select a test sample",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        help="Each index represents one sample in the original test set.",
    )

    sample = X_test.iloc[[sample_index]]
    true_label = int(y_test.iloc[sample_index])

    prediction = int(model.predict(sample)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        class_ids = [int(c) for c in model.classes_]
    else:
        st.error("This model does not support probability estimates.")
        st.stop()

    predicted_probability = float(max(probabilities))
    sorted_probabilities = sorted(probabilities, reverse=True)
    probability_margin = (
        float(sorted_probabilities[0] - sorted_probabilities[1])
        if len(sorted_probabilities) > 1
        else predicted_probability
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Activity",
            f"{ACTIVITY_ICONS[prediction]} {ACTIVITIES[prediction]}",
        )

    with col2:
        st.metric(
            "Actual Activity",
            f"{ACTIVITY_ICONS[true_label]} {ACTIVITIES[true_label]}",
        )

    with col3:
        st.metric(
            "Top-Class Probability",
            f"{predicted_probability:.2%}",
            help=(
                "The largest probability returned by the model. "
                "It is not a guarantee that the prediction is correct."
            ),
        )

    if prediction == true_label:
        st.success("Correct prediction: predicted and actual activities match.")
    else:
        st.warning(
            "Incorrect prediction: the predicted activity differs "
            "from the dataset label."
        )

    st.subheader("Class Probability Distribution")

    # Include every activity, even if its probability is very small.
    probability_df = pd.DataFrame(
        {
            "Activity ID": class_ids,
            "Activity": [ACTIVITIES[c] for c in class_ids],
            "Probability": probabilities,
        }
    )

    # Use the same activity order for every selected sample.
    probability_df["Activity"] = pd.Categorical(
        probability_df["Activity"],
        categories=list(ACTIVITIES.values()),
        ordered=True,
    )

    probability_df = probability_df.sort_values("Activity")

    # Horizontal bars make all six class labels easy to read.
    # Highlight the predicted class and label each bar in percent.
    fig, ax = plt.subplots(figsize=(10, 4.8))

    bar_colors = [
        "#ef8354" if activity_id == prediction else "#3977a8"
        for activity_id in probability_df["Activity ID"]
    ]

    bars = ax.barh(
        probability_df["Activity"].astype(str),
        probability_df["Probability"],
        color=bar_colors,
        height=0.65,
    )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Model output probability")
    ax.set_ylabel("")
    ax.set_title("Probability by Activity Class", pad=14)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, probability in zip(bars, probability_df["Probability"]):
        ax.text(
            min(probability + 0.015, 1.025),
            bar.get_y() + bar.get_height() / 2,
            f"{probability:.2%}",
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Orange identifies the predicted class. Blue identifies the "
        "other classes. Percentages show the model's raw probability "
        "outputs; they have not been independently calibrated."
    )

    with st.expander("Inspect probability values"):
        display_df = probability_df[
            ["Activity", "Probability"]
        ].copy()

        display_df["Activity"] = display_df["Activity"].astype(str)
        display_df["Probability"] = display_df["Probability"].map(
            lambda value: f"{value:.6%}"
        )

        st.dataframe(display_df, hide_index=True, use_container_width=True)

    with st.expander("Inspect sample features"):
        st.write(
            f"Sample index: {sample_index} | "
            f"Number of input features: {sample.shape[1]}"
        )
        st.dataframe(sample, hide_index=True, use_container_width=True)



# --------------------------------------------------
# Tab 2: Model Evaluation
# --------------------------------------------------
with tab2:
    st.header("Held-Out Test Performance — Selected RBF SVM")

    # Load evaluation results generated by evaluate_best_svm.py.
    BEST_SVM_REPORT = Path("report") / "best_svm"
    METRICS_PATH = BEST_SVM_REPORT / "metrics.csv"
    FIGURE_PATH_BEST = BEST_SVM_REPORT / "confusion_matrix.png"
    PREDICTIONS_PATH = BEST_SVM_REPORT / "predictions.csv"

    if METRICS_PATH.exists():
        best_metrics = pd.read_csv(METRICS_PATH).iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Test Samples",
            f"{int(best_metrics['test_samples']):,}",
        )
        col2.metric(
            "Accuracy",
            f"{best_metrics['accuracy']:.2%}",
        )
        col3.metric(
            "Balanced Accuracy",
            f"{best_metrics['balanced_accuracy']:.2%}",
        )
        col4.metric(
            "Macro-F1",
            f"{best_metrics['macro_f1']:.2%}",
        )

        st.info(
            "These metrics correspond to the selected RBF SVM "
            "(C=3.0, gamma=0.001) evaluated on the original UCI HAR "
            "test split. This split contains participants excluded "
            "from training. The test set has been used during "
            "development, so these results are exploratory."
        )
    else:
        st.warning(
            "Selected SVM metrics not found. Run "
            "`python src/evaluate_best_svm.py` first."
        )

    # Preserve the original comparison table.
    if REPORT_PATH.exists():
        results = pd.read_csv(REPORT_PATH)

        st.subheader("Original Model Comparison")
        st.dataframe(
            results,
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "This table contains the previously saved baseline "
            "evaluation. It is separate from the selected SVM "
            "metrics displayed above."
        )
    else:
        st.warning(
            "Original comparison report not found at "
            "`report/evaluation_metrics.csv`."
        )

    # Display the selected SVM confusion matrix.
    if FIGURE_PATH_BEST.exists():
        st.subheader("Selected RBF SVM — Confusion Matrix")
        st.image(
            str(FIGURE_PATH_BEST),
            caption=(
                "Selected RBF SVM (C=3.0, gamma=0.001) — "
                "exploratory held-out test evaluation"
            ),
            use_container_width=True,
        )
    else:
        st.warning(
            "Selected SVM confusion matrix not found. Run "
            "`python src/evaluate_best_svm.py` first."
        )

    # Misclassification summary and prediction inspection.
    if PREDICTIONS_PATH.exists():
        predictions_df = pd.read_csv(PREDICTIONS_PATH)

        st.subheader("Misclassification Summary")

        # Check that the saved report contains the required columns.
        required_columns = {"actual", "predicted"}

        if required_columns.issubset(predictions_df.columns):
            # Convert numeric activity labels to readable names.
            activity_names = {
                1: "Walking",
                2: "Walking upstairs",
                3: "Walking downstairs",
                4: "Sitting",
                5: "Standing",
                6: "Laying",
            }

            predictions_df["Actual Activity"] = (
                pd.to_numeric(predictions_df["actual"], errors="coerce")
                .map(activity_names)
            )

            predictions_df["Predicted Activity"] = (
                pd.to_numeric(
                    predictions_df["predicted"], errors="coerce"
                ).map(activity_names)
            )

            predictions_df["Correct"] = (
                predictions_df["Actual Activity"]
                == predictions_df["Predicted Activity"]
            )

            total_samples = len(predictions_df)
            incorrect_df = predictions_df[
                ~predictions_df["Correct"]
            ].copy()

            total_errors = len(incorrect_df)
            error_rate = (
                total_errors / total_samples
                if total_samples > 0
                else 0
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Samples", f"{total_samples:,}")
            col2.metric("Incorrect Predictions", f"{total_errors:,}")
            col3.metric("Error Rate", f"{error_rate:.2%}")

            if total_errors > 0:
                st.markdown("#### Errors by Actual Activity")

                errors_by_activity = (
                    incorrect_df["Actual Activity"]
                    .value_counts()
                    .rename_axis("Actual Activity")
                    .reset_index(name="Incorrect Predictions")
                )

                st.dataframe(
                    errors_by_activity,
                    hide_index=True,
                    use_container_width=True,
                )

                st.markdown("#### Most Common Misclassifications")

                confusion_pairs = (
                    incorrect_df.groupby(
                        ["Actual Activity", "Predicted Activity"]
                    )
                    .size()
                    .reset_index(name="Error Count")
                    .sort_values("Error Count", ascending=False)
                )

                confusion_pairs["Share of All Errors"] = (
                    confusion_pairs["Error Count"] / total_errors
                ).map(lambda value: f"{value:.2%}")

                st.dataframe(
                    confusion_pairs,
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.success("No incorrect predictions were found.")

            st.divider()

            with st.expander(
                "Inspect test predictions and analyze errors"
            ):
                activity_options = [
                    "All activities",
                    "Walking",
                    "Walking upstairs",
                    "Walking downstairs",
                    "Sitting",
                    "Standing",
                    "Laying",
                ]

                filter_col1, filter_col2 = st.columns(2)

                with filter_col1:
                    selected_activity = st.selectbox(
                        "Filter by actual activity",
                        activity_options,
                    )

                with filter_col2:
                    selected_status = st.selectbox(
                        "Prediction status",
                        ["All predictions", "Correct only", "Incorrect only"],
                    )

                filtered_df = predictions_df.copy()

                if selected_activity != "All activities":
                    filtered_df = filtered_df[
                        filtered_df["Actual Activity"]
                        == selected_activity
                    ]

                if selected_status == "Correct only":
                    filtered_df = filtered_df[filtered_df["Correct"]]
                elif selected_status == "Incorrect only":
                    filtered_df = filtered_df[~filtered_df["Correct"]]

                st.write(
                    f"Showing {len(filtered_df):,} of "
                    f"{total_samples:,} samples."
                )

                display_columns = [
                    "Actual Activity",
                    "Predicted Activity",
                    "Correct",
                ]

                st.dataframe(
                    filtered_df[display_columns],
                    hide_index=True,
                    use_container_width=True,
                )
        else:
            st.error(
                "The prediction report must contain 'actual' and "
                "'predicted' columns. Check the CSV at "
                f"`{PREDICTIONS_PATH}`."
            )
    else:
        st.warning(
            "Prediction report not found. Run "
            "`python src/evaluate_best_svm.py` first."
        )

# --------------------------------------------------
# Tab 3: Wearable System Design
# --------------------------------------------------
with tab3:
    st.header("Proposed Smart-Activewear Architecture")

    st.write(
        "WearSense outlines how wearable sensing, signal processing, "
        "machine learning, and user feedback could work together."
    )

    layers = [
        (
            "01",
            "Sensing layer",
            "Accelerometer and gyroscope embedded in or attached to a garment.",
        ),
        (
            "02",
            "Processing layer",
            "Sensor-data preparation and feature extraction.",
        ),
        (
            "03",
            "AI layer",
            "Activity classification using a trained machine-learning model.",
        ),
        (
            "04",
            "Feedback layer",
            "A dashboard displaying activity predictions and model outputs.",
        ),
    ]

    for number, title, description in layers:
        with st.container(border=True):
            col1, col2 = st.columns([1, 12])

            with col1:
                st.markdown(f"### {number}")

            with col2:
                st.markdown(f"**{title}**")
                st.write(description)

    st.subheader("Current Limitations")

    st.markdown(
        """
        - The prototype uses the preprocessed UCI HAR dataset.
        - Live garment sensors have not yet been integrated.
        - Physical textile and electronics integration has not yet
          been implemented.
        - Model probabilities should not be interpreted as calibrated
          measures of real-world certainty.
        """
    )

st.divider()

st.caption(
    "WearSense | Human Activity Recognition Research Prototype | "
    "Not intended for clinical diagnosis or injury prevention."
)
