import json
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Health Insight",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --navy: #142d4e;
        --blue: #2563eb;
        --teal: #0f9f9a;
        --ink: #172b4d;
        --muted: #64748b;
        --surface: #ffffff;
        --canvas: #f4f7fb;
        --line: #e5ebf3;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244,247,251,.88); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102744 0%, #173c63 100%);
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: #eef5ff !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #b9cbe0 !important;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16); }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 9px 11px;
        border-radius: 9px;
        margin: 3px 0;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,.10);
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--navy) !important;
        letter-spacing: -0.035em;
    }
    h1 { font-size: clamp(1.8rem, 3vw, 2.45rem) !important; font-weight: 800 !important; }
    h2 { font-size: 1.35rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.08rem !important; font-weight: 700 !important; }
    .page-eyebrow {
        color: var(--teal);
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: .72rem;
        font-weight: 700;
        margin-bottom: .35rem;
    }
    .page-subtitle { color: var(--muted); margin-top: -.55rem; margin-bottom: 1.5rem; }
    .hero {
        background: linear-gradient(120deg, #102744 0%, #1d5680 72%, #168b91 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(20,45,78,.12);
    }
    .hero h1 { color: white !important; margin: 0 0 8px 0; }
    .hero p { color: #d9e9f7; margin: 0; }
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 18px 20px;
        min-height: 112px;
        box-shadow: 0 5px 18px rgba(25,48,78,.045);
    }
    .metric-label { color: var(--muted); font-size: .82rem; font-weight: 600; }
    .metric-value {
        color: var(--navy);
        font-family: 'Manrope',sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 7px;
    }
    .metric-note { color: #8492a6; font-size: .75rem; margin-top: 2px; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 15px;
        background: white;
    }
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 20px;
    }
    .stButton > button, .stFormSubmitButton > button {
        background: var(--blue);
        color: white;
        border: 0;
        border-radius: 9px;
        padding: .62rem 1.2rem;
        font-weight: 700;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #1d4ed8;
        color: white;
        border: 0;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
    }
    input, textarea, [data-baseweb="select"] > div {
        border-radius: 8px !important;
    }
    .info-note {
        background: #edf6ff;
        border-left: 4px solid #3b82f6;
        padding: 13px 16px;
        border-radius: 8px;
        color: #25476b;
        font-size: .9rem;
    }
    .footer {
        color: #94a3b8;
        font-size: .78rem;
        text-align: center;
        padding: 2.5rem 0 1rem;
        border-top: 1px solid var(--line);
        margin-top: 2.5rem;
    }
    .section-kicker {
        color: var(--teal);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: 0 0 .25rem;
    }
    [data-testid="stHorizontalBlock"] { gap: 1rem; }
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 14px 16px;
    }
    .feature-intro {
        background: linear-gradient(120deg,#eef6ff,#effaf9);
        border: 1px solid #dceaf5;
        border-radius: 13px;
        padding: 15px 18px;
        margin: 0 0 18px;
        color: #35516f;
        font-size: .9rem;
    }
    .result-panel {
        background: linear-gradient(135deg,#102744,#1c557d);
        color: white;
        padding: 22px 24px;
        border-radius: 16px;
        min-height: 150px;
    }
    .result-panel small {
        color: #c6d9eb;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .result-panel strong {
        display: block;
        color: white;
        font-family: 'Manrope',sans-serif;
        font-size: 1.8rem;
        margin-top: 12px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px 9px 0 0;
        padding: 10px 15px;
    }
    @media (max-width: 800px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        .hero { padding: 22px 20px; }
        .metric-card { min-height: 100px; padding: 14px; }
        .metric-value { font-size: 1.3rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Load existing model and metadata
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    project_dir = Path(__file__).resolve().parent

    model_path = project_dir / "Streamlit/diabetes_model.joblib"
    metrics_path = project_dir / "Streamlit/metrics.json"

    # Check model file
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di: {model_path}"
        )

    # Check metadata file
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics tidak ditemukan di: {metrics_path}"
        )

    model = joblib.load(model_path)

    with open(metrics_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


try:
    model, metadata = load_model()
except Exception as error:
    st.error(
        "Aplikasi belum dapat memuat model atau metadata. "
        "Pastikan diabetes_model.joblib dan metrics.json tersedia "
        "di folder aplikasi."
    )
    st.exception(error)
    st.stop()


# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 4px 22px 4px">
            <div style="font-size:2rem">🩺</div>
            <div style="font-family:Manrope,sans-serif;font-size:1.25rem;
                        font-weight:800;color:white">AI Health Insight</div>
            <div style="font-size:.78rem;color:#b9cbe0;margin-top:4px">
                Diabetes Risk Classification
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**MAIN MENU**")
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Dataset Explorer",
            "AI Prediction",
            "Model Evaluation",
            "About Project",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Machine Learning • Healthcare Analytics")
    st.caption("Educational project")


def page_heading(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="page-eyebrow">{eyebrow}</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(
        f'<p class="page-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def readable_label(feature):
    """Turn technical column names into friendlier form labels."""
    return str(feature).replace("_", " ").replace("-", " ").strip().title()


def normalize_feature(name):
    """Normalize feature/metric names for safe matching across pages."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


# -------------------------------------------------------------------
# Dashboard
# -------------------------------------------------------------------
if page == "Dashboard":
    metrics = metadata.get("metrics", {})
    results = (
        pd.DataFrame(metrics).T
        if isinstance(metrics, dict) and metrics
        else pd.DataFrame()
    )
    numeric_results = (
        results.apply(pd.to_numeric, errors="coerce")
        if not results.empty
        else pd.DataFrame()
    )

    best_model = metadata.get("best_model", "—")
    train_rows = metadata.get("train_rows", "—")
    test_rows = metadata.get("test_rows", "—")

    def find_metric(metric_names):
        if results.empty:
            return None
        for column in results.columns:
            normalized = normalize_feature(column)
            if normalized in metric_names:
                try:
                    value = pd.to_numeric(
                        results.loc[best_model, column],
                        errors="coerce",
                    )
                    if pd.notna(value):
                        return float(value)
                except (KeyError, TypeError, ValueError):
                    continue
        return None

    accuracy = find_metric({"accuracy"})
    precision = find_metric({"precision"})
    recall = find_metric({"recall"})
    f1_score = find_metric({"f1", "f1_score", "f1score"})

    st.markdown(
        f"""
        <div class="hero" style="padding:30px 34px;position:relative;overflow:hidden">
            <div style="max-width:760px;position:relative;z-index:1">
                <div style="font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;
                            font-weight:800;color:#9ee7e2;margin-bottom:10px">
                    AI HEALTH INSIGHT · OVERVIEW
                </div>
                <h1 style="font-size:2.25rem!important;margin-bottom:9px">
                    Understand health data.<br>Explore model insights.
                </h1>
                <p style="font-size:1rem;max-width:620px;line-height:1.65">
                    A machine learning dashboard for exploring diabetes risk classification,
                    dataset information, and model performance.
                </p>
            </div>
            <div style="position:absolute;right:30px;bottom:-38px;font-size:10rem;
                        opacity:.10;line-height:1">✚</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-kicker">AT A GLANCE</div>',
        unsafe_allow_html=True,
    )
    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        metric_card(
            "Training records",
            f"{train_rows:,}" if isinstance(train_rows, int) else str(train_rows),
            "Records used during training",
        )
    with stat2:
        metric_card(
            "Testing records",
            f"{test_rows:,}" if isinstance(test_rows, int) else str(test_rows),
            "Records used for evaluation",
        )
    with stat3:
        metric_card(
            "Selected model",
            str(best_model),
            "Best model in saved metadata",
        )
    with stat4:
        metric_card(
            "Accuracy",
            f"{accuracy:.1%}" if accuracy is not None else "—",
            "Saved evaluation result",
        )

    st.markdown("")
    st.markdown(
        '<div class="section-kicker">MODEL QUALITY</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Performance snapshot")

    if not numeric_results.empty:
        metric_specs = [
            ("Accuracy", accuracy),
            ("Precision", precision),
            ("Recall", recall),
            ("F1-score", f1_score),
        ]
        metric_columns = st.columns(4)

        for column, (label, value) in zip(metric_columns, metric_specs):
            with column:
                display_value = f"{value:.1%}" if value is not None else "—"
                progress_value = (
                    max(0, min(100, value * 100))
                    if value is not None
                    else 0
                )
                st.markdown(
                    f"""
                    <div class="metric-card" style="min-height:128px">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{display_value}</div>
                        <div style="height:6px;background:#edf2f7;border-radius:99px;
                                    margin-top:12px;overflow:hidden">
                            <div style="height:6px;width:{progress_value:.1f}%;
                                        background:linear-gradient(90deg,#2563eb,#0f9f9a);
                                        border-radius:99px"></div>
                        </div>
                        <div class="metric-note">Selected model · {best_model}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("")
        chart_col, table_col = st.columns([1.15, 1])

        with chart_col:
            with st.container(border=True):
                st.subheader("Compare model metrics")
                st.caption("Scores loaded from metrics.json")
                st.bar_chart(
                    numeric_results,
                    y_label="Score",
                    height=300,
                )

        with table_col:
            with st.container(border=True):
                st.subheader("Detailed results")
                st.caption("Exact values saved with this project")
                st.dataframe(
                    numeric_results.style.format("{:.4f}", na_rep="—"),
                    use_container_width=True,
                    height=300,
                )
    else:
        with st.container(border=True):
            st.info(
                "Model metrics are not available yet. Check that metrics.json "
                "contains the evaluation results."
            )

    # Dataset split only — confusion matrix has been removed from Dashboard.
    st.markdown("")
    st.markdown(
        '<div class="section-kicker">DATA OVERVIEW</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Dataset split")
        st.caption("Training and testing records")

        if isinstance(train_rows, (int, float)) and isinstance(test_rows, (int, float)):
            total_rows = train_rows + test_rows

            if total_rows > 0:
                split_df = pd.DataFrame(
                    {"Records": [train_rows, test_rows]},
                    index=["Training", "Testing"],
                )
                st.bar_chart(
                    split_df,
                    y="Records",
                    height=210,
                )
                st.markdown(
                    f"**Total records:** {total_rows:,}  \n"
                    f"Training: {train_rows / total_rows:.1%} · "
                    f"Testing: {test_rows / total_rows:.1%}"
                )
            else:
                st.info("Dataset split counts are zero.")
        else:
            st.info("Training/testing record counts are not available.")

    st.markdown("")
    st.markdown(
        '<div class="info-note"><b>Educational use only.</b> This project demonstrates '
        'machine learning classification using tabular healthcare data. Model scores '
        'describe performance on the saved evaluation data and are not a clinical '
        'diagnosis or a substitute for professional medical advice.</div>',
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Dataset Explorer
# -------------------------------------------------------------------
elif page == "Dataset Explorer":
    page_heading(
        "DATA MANAGEMENT",
        "Dataset Explorer",
        "Explore the diabetes dataset bundled with this website. No upload is needed.",
    )

    project_dir = Path(__file__).resolve().parent
    dataset_dirs = [
        project_dir,
        project_dir / "data",
        project_dir / "datasets",
    ]
    preferred_names = [
        "diabetes.csv",
        "diabetes_dataset.csv",
        "diabetes_prediction_dataset.csv",
        "diabetes_data.csv",
    ]

    dataset_candidates = []
    for folder in dataset_dirs:
        if folder.exists():
            for filename in preferred_names:
                candidate = folder / filename
                if candidate.is_file() and candidate not in dataset_candidates:
                    dataset_candidates.append(candidate)

    if not dataset_candidates:
        for folder in dataset_dirs:
            if folder.exists():
                dataset_candidates.extend(
                    p for p in sorted(folder.glob("*.csv"))
                    if p.is_file() and p not in dataset_candidates
                )

    if dataset_candidates:
        selected_dataset = dataset_candidates[0]
        try:
            data = pd.read_csv(selected_dataset)
            st.markdown(
                f'<div class="info-note"><b>Dataset aktif:</b> '
                f'{selected_dataset.name} &nbsp;·&nbsp; '
                f'{len(data):,} records loaded automatically from the project.</div>',
                unsafe_allow_html=True,
            )

            st.markdown("")
            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card(
                    "Total rows",
                    f"{len(data):,}",
                    "Records in project dataset",
                )
            with c2:
                metric_card(
                    "Total columns",
                    f"{len(data.columns):,}",
                    "Available features",
                )
            with c3:
                missing_total = int(data.isna().sum().sum())
                metric_card(
                    "Missing values",
                    f"{missing_total:,}",
                    "Empty cells detected",
                )

            st.markdown("")
            tab_preview, tab_quality, tab_stats = st.tabs(
                ["Data preview", "Data quality", "Summary statistics"]
            )

            with tab_preview:
                st.subheader("First 20 records")
                st.dataframe(
                    data.head(20),
                    use_container_width=True,
                    height=420,
                )

            with tab_quality:
                st.subheader("Missing values by column")
                missing = data.isna().sum().sort_values(ascending=False)
                missing_df = missing.rename("Missing values").to_frame()
                st.dataframe(
                    missing_df,
                    use_container_width=True,
                )
                if missing.sum() == 0:
                    st.success("No missing values were found in this dataset.")

            with tab_stats:
                st.subheader("Numeric feature summary")
                numeric_data = data.select_dtypes(include="number")
                if not numeric_data.empty:
                    st.dataframe(
                        numeric_data.describe().T,
                        use_container_width=True,
                    )
                else:
                    st.info("No numeric columns found in the bundled dataset.")

        except Exception as error:
            st.error(f"Dataset could not be read: {error}")
    else:
        st.error(
            "Dataset CSV belum ditemukan di project. Tambahkan file dataset "
            "ke repository GitHub, sejajar dengan app.py atau di folder data/ "
            "atau datasets/. Setelah itu website akan membacanya otomatis—"
            "pengunjung tidak perlu mengunggah file."
        )


# -------------------------------------------------------------------
# AI Prediction
# -------------------------------------------------------------------
elif page == "AI Prediction":
    page_heading(
        "PREDICTIVE ANALYTICS",
        "Diabetes Risk Prediction",
        "Complete the health profile below, then run the trained model to see its classification output.",
    )

    feature_columns = metadata["feature_columns"]
    numeric_columns = metadata["numeric_columns"]

    st.markdown(
        """
        <div class="feature-intro">
            <b>How to use this form</b><br>
            Enter values using the same units and category labels as the training dataset.
            Fields are grouped to make the assessment easier to complete.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Illustrative profiles are for software testing only.
    demo_profiles = {
        "Lower-value test profile": {
            "age": 25, "gender": "Male", "country": "Indonesia",
            "height_cm": 170, "weight_kg": 62, "bmi": 21.45,
            "waist_circumference_cm": 75, "blood_glucose": 90,
            "hba1c": 5.1, "fasting_blood_sugar": 85, "insulin_level": 8,
            "blood_pressure_systolic": 115, "blood_pressure_diastolic": 75,
            "total_cholesterol": 165, "hdl": 55, "ldl": 95,
            "triglycerides": 100, "heart_rate": 72,
        },
        "Higher-value test profile": {
            "age": 55, "gender": "Male", "country": "Indonesia",
            "height_cm": 170, "weight_kg": 90, "bmi": 31.14,
            "waist_circumference_cm": 108, "blood_glucose": 175,
            "hba1c": 8.0, "fasting_blood_sugar": 155, "insulin_level": 25,
            "blood_pressure_systolic": 150, "blood_pressure_diastolic": 95,
            "total_cholesterol": 245, "hdl": 35, "ldl": 165,
            "triglycerides": 240, "heart_rate": 90,
        },
    }

    def apply_demo_profile(profile_name):
        profile = demo_profiles[profile_name]
        for feature in feature_columns:
            key = f"prediction_{feature}"
            normalized = normalize_feature(feature)
            if normalized in profile:
                st.session_state[key] = profile[normalized]

    demo_col, note_col = st.columns([1, 2])

    with demo_col:
        selected_demo = st.selectbox(
            "Optional test profile",
            ["Manual entry", *demo_profiles.keys()],
            help="These sample values are for testing the application interface only.",
        )
        if selected_demo != "Manual entry":
            if st.button("Load sample values", use_container_width=True):
                apply_demo_profile(selected_demo)
                st.rerun()

    with note_col:
        st.caption(
            "Sample profiles are illustrative test inputs. The model output depends "
            "on the trained model and dataset; no result is guaranteed."
        )

    groups = {
        "Personal information": [],
        "Body measurements": [],
        "Blood glucose indicators": [],
        "Blood pressure and vital signs": [],
        "Cholesterol and lipid profile": [],
        "Other indicators": [],
    }

    for feature in feature_columns:
        name = normalize_feature(feature)

        if any(token in name for token in ["age", "gender", "sex", "country", "region"]):
            groups["Personal information"].append(feature)
        elif any(token in name for token in ["height", "weight", "bmi", "waist", "hip"]):
            groups["Body measurements"].append(feature)
        elif any(token in name for token in ["glucose", "sugar", "hba1c", "insulin"]):
            groups["Blood glucose indicators"].append(feature)
        elif any(token in name for token in ["pressure", "heart_rate", "pulse"]):
            groups["Blood pressure and vital signs"].append(feature)
        elif any(token in name for token in ["cholesterol", "hdl", "ldl", "triglyceride"]):
            groups["Cholesterol and lipid profile"].append(feature)
        else:
            groups["Other indicators"].append(feature)

    values = {}

    with st.form("prediction_form"):
        for group_name, group_features in groups.items():
            if not group_features:
                continue

            st.markdown(
                f'<div class="section-kicker">{group_name}</div>',
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                cols = st.columns(2)

                for index, feature in enumerate(group_features):
                    key = f"prediction_{feature}"
                    label = readable_label(feature)

                    with cols[index % 2]:
                        if feature in numeric_columns:
                            default_value = st.session_state.get(key, 0.0)
                            values[feature] = st.number_input(
                                label,
                                value=float(default_value),
                                format="%.3f",
                                key=key,
                                help=f"Expected model feature: {feature}",
                            )
                        else:
                            default_value = st.session_state.get(key, "")
                            values[feature] = st.text_input(
                                label,
                                value=str(default_value),
                                key=key,
                                help=f"Expected category/label: {feature}",
                            )

        st.markdown("")
        submitted = st.form_submit_button(
            "🧠  Analyze diabetes risk",
            use_container_width=True,
        )

    if submitted:
        try:
            input_data = pd.DataFrame(
                [values],
                columns=feature_columns,
            )
            prediction = model.predict(input_data)[0]

            st.markdown("")
            st.markdown(
                '<div class="section-kicker">MODEL OUTPUT</div>',
                unsafe_allow_html=True,
            )

            result_col, probability_col = st.columns([1, 1.5])

            with result_col:
                st.markdown(
                    f"""
                    <div class="result-panel">
                        <small>Predicted class</small>
                        <strong>{prediction}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with probability_col:
                with st.container(border=True):
                    st.subheader("Prediction probability")

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_data)[0]
                        probability_df = pd.DataFrame(
                            {
                                "Class": model.classes_,
                                "Probability": probabilities,
                            }
                        ).sort_values(
                            "Probability",
                            ascending=False,
                        )

                        st.dataframe(
                            probability_df.style.format(
                                {"Probability": "{:.1%}"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.bar_chart(
                            probability_df.set_index("Class")["Probability"],
                            y_label="Probability",
                            height=190,
                        )
                    else:
                        st.info(
                            "Model ini tidak menyediakan predict_proba(), "
                            "jadi probabilitas kelas tidak dapat ditampilkan."
                        )

            st.markdown(
                '<div class="info-note">This output is generated by a machine learning '
                'model for educational testing only. It is not a medical diagnosis. '
                'Consult a qualified healthcare professional for medical assessment.</div>',
                unsafe_allow_html=True,
            )

        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")


# -------------------------------------------------------------------
# Model Evaluation — confusion matrix removed
# -------------------------------------------------------------------
elif page == "Model Evaluation":
    page_heading(
        "MODEL VALIDATION",
        "Model Evaluation",
        "Review model performance and compare the evaluation results.",
    )

    results = pd.DataFrame(metadata.get("metrics", {})).T

    if not results.empty:
        numeric_results = results.apply(
            pd.to_numeric,
            errors="coerce",
        )

        metric_cols = st.columns(
            max(1, min(len(numeric_results.columns), 4))
        )

        for index, metric_name in enumerate(numeric_results.columns[:4]):
            values_for_metric = numeric_results[metric_name].dropna()

            if not values_for_metric.empty:
                with metric_cols[index]:
                    metric_card(
                        readable_label(metric_name),
                        f"{values_for_metric.iloc[0]:.3f}",
                        str(numeric_results.index[0]),
                    )

        with st.container(border=True):
            st.subheader("Performance metrics by model")
            st.caption("Comparison of saved model evaluation scores")

            st.dataframe(
                numeric_results.style.format(
                    "{:.4f}",
                    na_rep="—",
                ),
                use_container_width=True,
            )

            st.bar_chart(
                numeric_results,
                height=300,
                y_label="Score",
            )
    else:
        st.info(
            "No evaluation metrics were found in metrics.json."
        )


# -------------------------------------------------------------------
# About Project
# -------------------------------------------------------------------
elif page == "About Project":
    page_heading(
        "PROJECT INFORMATION",
        "About AI Health Insight",
        "An educational machine learning project exploring diabetes risk classification.",
    )

    with st.container(border=True):
        st.subheader("Project overview")
        st.write(
            "AI Health Insight explores diabetes risk classification using "
            "tabular healthcare data and machine learning models."
        )

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("🤖 Algorithms")
            st.markdown("- Random Forest\n- Logistic Regression")

    with col2:
        with st.container(border=True):
            st.subheader("⚙️ Technology stack")
            st.markdown(
                "- Python\n- Streamlit\n- Pandas\n- Scikit-learn\n- Git and GitHub"
            )

    with st.container(border=True):
        st.subheader("How the application works")
        steps = st.columns(3)

        steps[0].markdown(
            "### 01\n**Input data**\n\nProvide feature values based on the dataset."
        )
        steps[1].markdown(
            "### 02\n**Model analysis**\n\nThe trained model processes the input."
        )
        steps[2].markdown(
            "### 03\n**Prediction**\n\nThe application displays its predicted class."
        )

    st.markdown("")
    info1, info2, info3 = st.columns(3)

    with info1:
        metric_card(
            "Application",
            "Streamlit",
            "Interactive web interface",
        )
    with info2:
        metric_card(
            "ML approach",
            "Classification",
            "Tabular healthcare features",
        )
    with info3:
        metric_card(
            "Purpose",
            "Education",
            "Demonstration and model exploration",
        )

    st.markdown(
        '<div class="info-note">For educational and research purposes only. '
        'The application is not a medical diagnostic tool.</div>',
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown(
    '<div class="footer">AI Health Insight · Diabetes Risk Classification · Educational Project</div>',
    unsafe_allow_html=True,
)
