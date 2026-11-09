
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="AI Health Insight",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.joblib")
    with open("metrics.json", "r") as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_model()

st.sidebar.title("🩺 AI Health Insight")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset Explorer",
        "AI Prediction",
        "Model Evaluation",
        "About Project"
    ]
)

st.sidebar.caption("Diabetes Risk Classification")

if page == "Dashboard":

    st.title("Health Insight Dashboard")
    st.write("AI-powered diabetes risk classification project.")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Training Data",
        metadata["train_rows"]
    )

    col2.metric(
        "Testing Data",
        metadata["test_rows"]
    )

    col3.metric(
        "Algorithm",
        metadata["best_model"]
    )

    st.subheader("Model Performance")

    metrics = metadata["metrics"]

    results = pd.DataFrame(metrics).T
    st.dataframe(results, use_container_width=True)

elif page == "Dataset Explorer":

    st.title("Dataset Explorer")

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(data.head(20), use_container_width=True)

        c1, c2 = st.columns(2)

        c1.metric("Total Rows", len(data))
        c2.metric("Total Columns", len(data.columns))

        st.subheader("Missing Values")
        st.dataframe(
            data.isnull().sum().to_frame("Missing Values")
        )

elif page == "AI Prediction":

    st.title("AI Diabetes Risk Prediction")

    st.write(
        "Enter the feature values based on the training dataset."
    )

    feature_columns = metadata["feature_columns"]
    numeric_columns = metadata["numeric_columns"]

    values = {}

    with st.form("prediction_form"):

        cols = st.columns(2)

        for i, feature in enumerate(feature_columns):

            with cols[i % 2]:

                if feature in numeric_columns:

                    values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        format="%.3f"
                    )

                else:

                    values[feature] = st.text_input(
                        feature
                    )

        submit = st.form_submit_button(
            "Analyze Risk"
        )

    if submit:

        input_data = pd.DataFrame(
            [values],
            columns=feature_columns
        )

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        st.success(
            f"Predicted class: {prediction}"
        )

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            probability_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probabilities
            })

            st.dataframe(
                probability_df,
                use_container_width=True
            )

        st.caption(
            "Educational prediction only. Not a medical diagnosis."
        )

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    results = pd.DataFrame(
        metadata["metrics"]
    ).T

    st.subheader("Performance Metrics")

    st.dataframe(
        results.style.format("{:.4f}"),
        use_container_width=True
    )

    st.subheader("Confusion Matrix")

    cm_path = Path("confusion_matrix.png")

    if cm_path.exists():
        st.image(str(cm_path))

elif page == "About Project":

    st.title("About AI Health Insight")

    st.write("""
    AI Health Insight is a machine learning project
    designed to explore diabetes risk classification
    using tabular healthcare data.

    **Algorithms:**
    - Random Forest
    - Logistic Regression

    **Technology:**
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Git and GitHub
    """)

    st.info(
        "This project is intended for educational purposes."
    )

