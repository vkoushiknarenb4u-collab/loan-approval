"""Dark LoanIQ style Streamlit dashboard for the loan approval demo."""

from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "loan_model.pkl"
DATASET_PATH = PROJECT_DIR / "dataset" / "loan_data.csv"

FORM_KEYS = [
    "gender",
    "married",
    "dependents",
    "education",
    "self_employed",
    "applicant_income",
    "coapplicant_income",
    "loan_amount",
    "loan_term",
    "credit_history",
    "property_area",
]


@st.cache_resource
def load_model_bundle() -> dict:
    """Load the trained pipeline and test metrics once per app session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def apply_dashboard_styles() -> None:
    """Style Streamlit's standard widgets to match the supplied dark dashboard."""
    st.markdown(
        """
        <style>
        #MainMenu, footer, [data-testid="stHeader"] {visibility: hidden;}
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse at 75% 0%, rgba(38, 85, 160, .14), transparent 42%),
                linear-gradient(135deg, #080f1b 0%, #0a1322 48%, #080e18 100%);
            color: #edf4ff;
        }
        [data-testid="stMain"] {background: transparent;}
        .block-container {max-width: 1480px; padding: 1.5rem 2rem 2.5rem;}
        h1, h2, h3, h4, p, label {color: #edf4ff;}
        h1 {font-size: clamp(2rem, 3vw, 2.65rem); letter-spacing: -.04em; margin: 0;}
        h2 {font-size: 1.2rem; letter-spacing: -.02em;}
        [data-testid="stCaptionContainer"] p {color: #9eb3d1;}

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0c1626 0%, #09111d 100%);
            border-right: 1px solid rgba(137, 166, 207, .12);
        }
        [data-testid="stSidebar"] > div:first-child {padding: 1.4rem .9rem;}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color: #a9bddb;}
        .brand {display:flex; align-items:center; gap:.55rem; margin:.1rem .4rem 2.2rem;
                color:#f6f9ff; font-size:1.35rem; font-weight:800; letter-spacing:-.04em;}
        .brand-icon {display:grid; place-items:center; width:1.9rem; height:1.9rem;
                     border-radius:.55rem; color:#69a2ff; background:#112a4d; font-size:1.2rem;}
        .brand span {color:#69a2ff;}
        [data-testid="stSidebar"] [data-testid="stRadio"] > label {display:none;}
        [data-testid="stSidebar"] div[role="radiogroup"] {gap:.45rem;}
        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            padding:.7rem .8rem; border-radius:.75rem; color:#a8bddc;
            border:1px solid transparent; background:transparent; transition:all .15s ease;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            color:#f4f8ff; background:rgba(41, 92, 173, .18);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            color:#fff; border-color:rgba(74, 133, 255, .28);
            background:linear-gradient(100deg, #164eaa, #163878);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {display:none;}
        .sidebar-note {margin-top:3rem; padding:.9rem; border:1px solid rgba(135,160,198,.16);
                       border-radius:1rem; background:rgba(19,34,54,.7); color:#a8bddc;
                       font-size:.82rem; line-height:1.5;}

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color:rgba(126, 157, 202, .16) !important;
            border-radius:1.05rem !important;
            background:linear-gradient(145deg, rgba(18,31,49,.82), rgba(12,22,36,.88));
        }
        [data-testid="stForm"] {border:0; padding:0; background:transparent;}
        [data-testid="stSelectbox"] label, [data-testid="stNumberInput"] label {
            color:#bed0e9 !important; font-size:.86rem !important;
        }
        div[data-baseweb="select"] > div, [data-testid="stNumberInput"] input {
            color:#edf4ff !important; background:#19273a !important;
            border-color:rgba(137,166,207,.2) !important; border-radius:.65rem !important;
        }
        [data-testid="stNumberInput"] button {color:#bdd0eb !important;}
        .stButton button, .stFormSubmitButton button {
            min-height:2.75rem; border-radius:.7rem; font-weight:700;
            border:1px solid rgba(112,149,202,.28); color:#dceaff;
            background:rgba(19,34,54,.8); transition:all .15s ease;
        }
        .stButton button:hover, .stFormSubmitButton button:hover {
            color:white; border-color:#4b88fa; transform:translateY(-1px);
        }
        .stFormSubmitButton button[kind="primaryFormSubmit"],
        .stFormSubmitButton button[kind="primary"] {
            color:white; border-color:#2b70f7;
            background:linear-gradient(105deg,#1763e8,#2d78ff);
            box-shadow:0 8px 24px rgba(25,99,232,.2);
        }
        div[data-testid="stProgress"] > div > div {background:#35d692;}
        [data-testid="stMetric"] {background:rgba(9,17,29,.38); border:1px solid rgba(137,166,207,.12);
            border-radius:.8rem; padding:.85rem;}
        [data-testid="stMetricLabel"] p {color:#a8bddc !important;}
        [data-testid="stMetricValue"] {color:#f4f8ff !important;}

        .eyebrow {color:#82a4d2; font-size:.72rem; font-weight:700; letter-spacing:.16em;
                  text-transform:uppercase; margin:0 0 .4rem;}
        .page-title {color:#f2f6ff; font-size:clamp(2rem,3vw,2.6rem); font-weight:800;
                     line-height:1.13; letter-spacing:-.045em; margin:0;}
        .page-title span {color:#4b8cff;}
        .page-subtitle {color:#a8bddc; margin:.45rem 0 0; font-size:1rem;}
        .insight-card {display:flex; align-items:center; gap:.8rem; padding:1rem 1.1rem;
            margin-top:.7rem; border:1px solid rgba(126,157,202,.16); border-radius:1rem;
            background:linear-gradient(110deg,rgba(23,40,63,.88),rgba(16,29,47,.88));}
        .insight-icon {display:grid; place-items:center; flex:0 0 2.7rem; height:2.7rem;
            border-radius:.8rem; color:#31dfa0; background:rgba(24,152,115,.2); font-size:1.35rem;}
        .insight-card strong {display:block; color:#f2f7ff; font-size:.94rem;}
        .insight-card small {color:#9fb4d3; font-size:.8rem;}
        .panel-head {display:flex; align-items:center; gap:.8rem; margin-bottom:.9rem;}
        .panel-icon {display:grid; place-items:center; flex:0 0 2.65rem; height:2.65rem;
            border-radius:.8rem; color:#75aaff; background:rgba(34,102,221,.2); font-size:1.25rem;}
        .panel-title {color:#f3f7ff; font-size:1.15rem; font-weight:750; line-height:1.2;}
        .panel-subtitle {color:#9db2d2; font-size:.82rem; margin-top:.18rem;}
        .section-head {display:flex; align-items:center; gap:.7rem; margin:.5rem 0 .45rem;}
        .section-number {display:grid; place-items:center; flex:0 0 2rem; height:2rem;
            border-radius:50%; color:#092146; background:linear-gradient(140deg,#9bd0ff,#4d9eff);
            font-weight:800;}
        .section-title {font-size:.99rem; font-weight:750; color:#f0f6ff;}
        .section-note {color:#9db2d2; font-size:.78rem; margin-top:.1rem;}
        .result-card {display:flex; align-items:center; justify-content:space-between; gap:.75rem;
            margin:.35rem 0 .85rem; padding:1.1rem; border-radius:.9rem; border:1px solid;}
        .result-card.approved {background:linear-gradient(110deg,rgba(7,70,54,.44),rgba(10,48,44,.54));
            border-color:rgba(47,212,149,.2);}
        .result-card.rejected {background:linear-gradient(110deg,rgba(94,33,48,.38),rgba(55,25,39,.48));
            border-color:rgba(246,105,125,.22);}
        .result-copy {display:flex; align-items:center; gap:.75rem;}
        .result-check {display:grid; place-items:center; width:3rem; height:3rem; flex:0 0 3rem;
            border-radius:50%; font-size:1.45rem; font-weight:800;}
        .approved .result-check {background:rgba(47,212,149,.2); color:#38e09b;}
        .rejected .result-check {background:rgba(246,105,125,.18); color:#ff8796;}
        .result-title {font-size:1.15rem; font-weight:800;}
        .approved .result-title {color:#45e5a5;}
        .rejected .result-title {color:#ff91a0;}
        .result-description {color:#b3c5df; font-size:.78rem; margin-top:.2rem; max-width:15rem;}
        .confidence {display:flex; flex-direction:column; align-items:center; gap:.2rem;
            color:#9eb4d3; font-size:.72rem;}
        .confidence-ring {display:grid; place-items:center; width:4.1rem; height:4.1rem;
            border-radius:50%;}
        .confidence-inner {display:grid; place-items:center; width:3.35rem; height:3.35rem;
            border-radius:50%; background:#10253a; color:white; font-size:.95rem; font-weight:800;}
        .reason-title {font-size:.9rem; font-weight:750; color:#dceaff; margin:.8rem 0 .45rem;}
        .factor-row {display:flex; align-items:center; gap:.65rem; padding:.6rem .65rem;
            margin:.38rem 0; border:1px solid rgba(120,151,192,.11); border-radius:.75rem;
            background:rgba(11,21,35,.55);}
        .factor-icon {display:grid; place-items:center; width:2.35rem; height:2.35rem; flex:0 0 2.35rem;
            border-radius:.7rem; background:rgba(47,99,170,.2); color:#9bc5ff; font-weight:800;}
        .factor-copy {min-width:0; flex:1;}
        .factor-copy strong {display:block; color:#edf4ff; font-size:.8rem; line-height:1.3;}
        .factor-copy small {display:block; color:#9db2d2; font-size:.7rem; margin-top:.1rem;}
        .factor-tag {white-space:nowrap; border-radius:1rem; padding:.27rem .55rem; font-size:.66rem; font-weight:700;}
        .factor-tag.approve {background:rgba(25,166,117,.17); color:#55dda8;}
        .factor-tag.reject {background:rgba(203,71,91,.17); color:#ff9aa6;}
        .empty-result {padding:1.1rem; border:1px dashed rgba(137,166,207,.2); border-radius:.9rem;
            background:rgba(9,17,29,.3); color:#9eb4d3; font-size:.86rem; line-height:1.55;}
        .performance-note {color:#9eb4d3; font-size:.8rem; line-height:1.5;}
        .top-gap {margin-top:1.15rem;}
        @media (max-width: 900px) {
            .block-container {padding:1rem .85rem 2rem;}
            .page-subtitle {font-size:.9rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_application() -> None:
    """Reset all form fields and clear the displayed prediction."""
    for key in FORM_KEYS + ["prediction_result"]:
        st.session_state.pop(key, None)


def render_sidebar() -> str:
    """Render the compact dashboard brand and page navigation."""
    st.sidebar.markdown(
        '<div class="brand"><div class="brand-icon">▥</div><div>Loan<span>IQ</span></div></div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        ["Prediction", "Model Info", "Dataset", "About"],
        label_visibility="collapsed",
        key="page_navigation",
    )
    st.sidebar.markdown(
        '<div class="sidebar-note"><b>Smarter Decisions</b><br>Understand the model\'s predictions with clear, simple factors.</div>',
        unsafe_allow_html=True,
    )
    return page


def render_page_header(title: str, subtitle: str, with_prediction_badge: bool = False) -> None:
    """Render the dashboard page heading and optional insight badge."""
    title_html = (
        "Loan Approval <span>Prediction</span>"
        if with_prediction_badge
        else escape(title)
    )
    subtitle_html = escape(subtitle)
    left, right = st.columns([2.4, 1.0], gap="large")
    with left:
        st.markdown(
            f'<div class="eyebrow">PREDICT &nbsp;•&nbsp; ANALYZE &nbsp;•&nbsp; MAKE BETTER DECISIONS</div>'
            f'<div class="page-title">{title_html}</div>'
            f'<div class="page-subtitle">{subtitle_html}</div>',
            unsafe_allow_html=True,
        )
    if with_prediction_badge:
        with right:
            st.markdown(
                '<div class="insight-card"><div class="insight-icon">◈</div>'
                '<div><strong>Data-driven insights</strong>'
                '<small>Powered by machine learning</small></div></div>',
                unsafe_allow_html=True,
            )


def render_panel_heading(icon: str, title: str, subtitle: str = "") -> None:
    """Render a reusable icon and heading for a dashboard panel."""
    st.markdown(
        f'<div class="panel-head"><div class="panel-icon">{escape(icon)}</div>'
        f'<div><div class="panel-title">{escape(title)}</div>'
        f'<div class="panel-subtitle">{escape(subtitle)}</div></div></div>',
        unsafe_allow_html=True,
    )


def render_section_heading(number: int, title: str, subtitle: str) -> None:
    """Render a numbered section heading inside the applicant form."""
    st.markdown(
        f'<div class="section-head"><div class="section-number">{number}</div>'
        f'<div><div class="section-title">{escape(title)}</div>'
        f'<div class="section-note">{escape(subtitle)}</div></div></div>',
        unsafe_allow_html=True,
    )


def format_factor_name(transformed_name: str, applicant: pd.DataFrame) -> str:
    """Show a readable feature label with the value entered in the form."""
    name = transformed_name.split("__", maxsplit=1)[-1]
    feature_labels = [
        ("ApplicantIncome", "Applicant income"),
        ("CoapplicantIncome", "Co-applicant income"),
        ("LoanAmount", "Loan amount"),
        ("Loan_Amount_Term", "Loan term"),
        ("Credit_History", "Credit history"),
        ("Gender", "Gender"),
        ("Married", "Married"),
        ("Dependents", "Dependents"),
        ("Education", "Education"),
        ("Self_Employed", "Self-employed"),
        ("Property_Area", "Property area"),
    ]
    for feature, label in feature_labels:
        if name == feature:
            value = applicant.iloc[0][feature]
            if feature == "Credit_History":
                value = "Yes" if int(value) == 1 else "No"
            elif feature == "Loan_Amount_Term":
                value = f"{int(value)} months"
            elif feature == "LoanAmount":
                value = f"{int(value):,} (thousand dataset units)"
            elif feature in {"ApplicantIncome", "CoapplicantIncome"}:
                value = f"{int(value):,}"
            return f"{label} ({value})"
        if name.startswith(feature + "_"):
            category = name[len(feature) + 1 :]
            return f"{label} ({category})"
    return name


def local_factors(model, applicant: pd.DataFrame) -> list[tuple[str, float]]:
    """Calculate feature contributions to this application's approval score.

    Logistic regression is linear after preprocessing. Each transformed input
    multiplied by its learned coefficient contributes to the approval log-odds.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(applicant)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)[0]

    contributions = transformed * classifier.coef_[0]
    names = preprocessor.get_feature_names_out()

    # Name encoded values for people, then order the strongest contributions first.
    grouped: dict[str, float] = {}
    for feature_name, contribution in zip(names, contributions):
        readable_name = format_factor_name(feature_name, applicant)
        grouped[readable_name] = grouped.get(readable_name, 0.0) + float(contribution)
    return sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)


def render_factor_section(
    title: str,
    factors: list[tuple[str, float]],
    direction: str,
    tag: str,
    tag_class: str,
    limit: int = 2,
) -> None:
    """Display the strongest factors in one direction as compact dashboard rows."""
    st.markdown(f'<div class="reason-title">{escape(title)}</div>', unsafe_allow_html=True)
    if not factors:
        st.caption("No individual input fields moved the model score in this direction.")
        return

    for factor, _contribution in factors[:limit]:
        factor_html = escape(factor)
        st.markdown(
            f'<div class="factor-row"><div class="factor-icon">✦</div>'
            f'<div class="factor-copy"><strong>{factor_html}</strong>'
            f'<small>Pushes the score {escape(direction)}</small></div>'
            f'<div class="factor-tag {tag_class}">{escape(tag)}</div></div>',
            unsafe_allow_html=True,
        )


def render_prediction_result(model) -> None:
    """Render the latest prediction, confidence ring, and both sides of its reasons."""
    result = st.session_state.get("prediction_result")
    render_panel_heading("▥", "Prediction Result", "Model prediction based on the given details")
    if result is None:
        st.markdown(
            '<div class="empty-result">Fill in the applicant details and select '
            '<b>Predict Approval</b> to see the result, probability, and factors that '
            'influenced the model.</div>',
            unsafe_allow_html=True,
        )
        return

    approved = result["approved"]
    approval_probability = result["approval_probability"]
    outcome_probability = approval_probability if approved else 1 - approval_probability
    applicant = pd.DataFrame([result["applicant"]])
    state_class = "approved" if approved else "rejected"
    outcome = "Loan Approved" if approved else "Loan Rejected"
    symbol = "✓" if approved else "!"
    description = (
        "The model estimates a higher chance of approval for these details."
        if approved
        else "The model estimates a higher chance of rejection for these details."
    )
    angle = outcome_probability * 360
    st.markdown(
        f'<div class="result-card {state_class}"><div class="result-copy">'
        f'<div class="result-check">{symbol}</div><div><div class="result-title">{outcome}</div>'
        f'<div class="result-description">{description}</div></div></div>'
        f'<div class="confidence"><span>Confidence</span>'
        f'<div class="confidence-ring" style="background:conic-gradient(#35d995 {angle:.1f}deg, #26384e {angle:.1f}deg)">'
        f'<div class="confidence-inner">{outcome_probability:.0%}</div></div></div></div>',
        unsafe_allow_html=True,
    )
    st.progress(
        approval_probability,
        text=f"Estimated probability of approval: {approval_probability:.1%}",
    )

    factors = local_factors(model, applicant)
    supporting_approval = [item for item in factors if item[1] > 0]
    supporting_rejection = [item for item in factors if item[1] < 0]
    st.markdown('<div class="reason-title">Key Factors Influencing Prediction</div>', unsafe_allow_html=True)
    if approved:
        render_factor_section(
            "Reasons supporting approval", supporting_approval, "toward approval",
            "Supports approval", "approve",
        )
        render_factor_section(
            "Factors weighing against approval", supporting_rejection, "toward rejection",
            "Against approval", "reject",
        )
    else:
        render_factor_section(
            "Reasons supporting rejection", supporting_rejection, "toward rejection",
            "Supports rejection", "reject",
        )
        render_factor_section(
            "Factors weighing in favor of approval", supporting_approval, "toward approval",
            "Supports approval", "approve",
        )
    st.caption("These show how the model weighed each value; they are not causal reasons or a guarantee.")


def render_performance(metrics: dict) -> None:
    """Render headline model performance in a dashboard card."""
    render_panel_heading("▥", "Model Performance", "Evaluation on the held-out test data")
    accuracy_col, model_col = st.columns([1, 1.65])
    accuracy_col.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    with model_col:
        st.markdown(
            '<div class="reason-title">Model used</div>'
            '<div class="panel-title">Logistic Regression</div>'
            '<div class="performance-note">A simple classification model trained on historical application data.</div>',
            unsafe_allow_html=True,
        )
    small_metrics = st.columns(3)
    for column, label, key in zip(
        small_metrics,
        ["Precision", "Recall", "F1-score"],
        ["precision", "recall", "f1_score"],
    ):
        column.metric(label, f"{metrics[key]:.1%}")


def render_prediction_page(bundle: dict) -> None:
    """Render the applicant form and prediction/performance dashboard."""
    render_page_header(
        "Loan Approval Prediction",
        "Enter applicant details to see the likelihood of loan approval using a logistic regression model.",
        with_prediction_badge=True,
    )
    left_column, right_column = st.columns([1.62, 1.0], gap="medium")

    with left_column:
        with st.container(border=True):
            render_panel_heading("▣", "Applicant Details", "Provide the applicant's information to predict loan approval")
            with st.form("applicant_form"):
                with st.container(border=True):
                    render_section_heading(1, "Personal Information", "Basic applicant details")
                    personal_left, personal_right = st.columns(2, gap="medium")
                    with personal_left:
                        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
                        married = st.selectbox("Married", ["Yes", "No"], key="married")
                        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], key="dependents")
                    with personal_right:
                        education = st.selectbox(
                            "Education", ["Graduate", "Not Graduate"], key="education"
                        )
                        self_employed = st.selectbox(
                            "Self employed", ["No", "Yes"], key="self_employed"
                        )

                with st.container(border=True):
                    render_section_heading(2, "Financial Information", "Income and loan details")
                    financial_left, financial_right = st.columns(2, gap="medium")
                    with financial_left:
                        applicant_income = st.number_input(
                            "Applicant income (₹)", min_value=0, value=5000, step=500,
                            key="applicant_income",
                        )
                        coapplicant_income = st.number_input(
                            "Co-applicant income (₹)", min_value=0, value=0, step=500,
                            key="coapplicant_income",
                        )
                    with financial_right:
                        loan_amount = st.number_input(
                            "Loan amount (₹, in thousands)", min_value=1, value=120, step=10,
                            key="loan_amount",
                        )
                        loan_term = st.selectbox(
                            "Loan term (months)", [360, 300, 240, 180, 120, 84, 60, 36, 12],
                            index=0, key="loan_term",
                        )

                with st.container(border=True):
                    render_section_heading(3, "Other Information", "Credit and property details")
                    other_left, other_right = st.columns(2, gap="medium")
                    with other_left:
                        credit_history = st.selectbox(
                            "Credit history meets guidelines", ["Yes", "No"], key="credit_history"
                        )
                    with other_right:
                        property_area = st.selectbox(
                            "Property area", ["Urban", "Semiurban", "Rural"], key="property_area"
                        )

                reset_col, submit_col = st.columns([.75, 1.8], gap="small")
                with reset_col:
                    st.form_submit_button("↺  Reset", on_click=reset_application, key="reset_button")
                with submit_col:
                    submitted = st.form_submit_button(
                        "✦  Predict Approval", type="primary", key="predict_button"
                    )

    if submitted:
        applicant = pd.DataFrame(
            [
                {
                    "Gender": gender,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self_Employed": self_employed,
                    "ApplicantIncome": applicant_income,
                    "CoapplicantIncome": coapplicant_income,
                    "LoanAmount": loan_amount,
                    "Loan_Amount_Term": loan_term,
                    "Credit_History": 1 if credit_history == "Yes" else 0,
                    "Property_Area": property_area,
                }
            ]
        )
        approval_probability = float(bundle["model"].predict_proba(applicant)[0, 1])
        st.session_state["prediction_result"] = {
            "approved": approval_probability >= 0.5,
            "approval_probability": approval_probability,
            "applicant": applicant.iloc[0].to_dict(),
        }

    with right_column:
        with st.container(border=True):
            render_prediction_result(bundle["model"])
        with st.container(border=True):
            render_performance(bundle["metrics"])


def render_model_info_page(bundle: dict) -> None:
    """Show model details, evaluation scores, and confusion matrix."""
    render_page_header("Model Information", "How the classifier is trained and evaluated.")
    metrics = bundle["metrics"]
    with st.container(border=True):
        render_panel_heading("▥", "Logistic Regression", "A binary classifier for the approval label")
        st.write(
            "The training pipeline fills missing numeric values with the median, "
            "fills missing categories with the most frequent value, scales numeric "
            "features, one-hot encodes categories, then fits logistic regression. "
            "The saved pipeline applies the same preprocessing to app inputs."
        )
        score_cols = st.columns(4)
        for column, label, key in zip(
            score_cols,
            ["Accuracy", "Precision", "Recall", "F1-score"],
            ["accuracy", "precision", "recall", "f1_score"],
        ):
            column.metric(label, f"{metrics[key]:.1%}")
        st.write("Confusion matrix: rows are actual outcomes; columns are predictions.")
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"],
                index=["Actual rejected", "Actual approved"],
                columns=["Predicted rejected", "Predicted approved"],
            )
        )


def render_dataset_page() -> None:
    """Show a small overview of the CSV used to train the model."""
    render_page_header("Dataset", "Historical loan applications used by this learning demo.")
    if not DATASET_PATH.exists():
        st.error(f"Dataset file not found: {DATASET_PATH}")
        return
    data = pd.read_csv(DATASET_PATH)
    with st.container(border=True):
        render_panel_heading("▤", "Loan application data", "The target is Loan_Status (Y = approved, N = rejected)")
        total, approved, rejected = st.columns(3)
        total.metric("Applications", f"{len(data):,}")
        total_labels = data["Loan_Status"].astype(str).str.strip()
        approved.metric("Recorded approvals", f"{int((total_labels == 'Y').sum()):,}")
        rejected.metric("Recorded rejections", f"{int((total_labels == 'N').sum()):,}")
        st.dataframe(data.head(12))
        st.caption("Loan_ID is included in the CSV for identification but is not used as a model feature.")


def render_about_page() -> None:
    """Explain the purpose and limitations of the example."""
    render_page_header("About LoanIQ", "A beginner project for learning a complete classification workflow.")
    with st.container(border=True):
        render_panel_heading("◈", "What this project demonstrates", "CSV data, model training, evaluation, and a prediction interface")
        st.write(
            "This demo predicts a recorded approval label from applicant details. "
            "It demonstrates a standard machine learning workflow with logistic regression, "
            "missing-value handling, categorical encoding, and test-set metrics."
        )
        st.warning(
            "This is a learning demo, not a lending tool. The data is small and historical, "
            "and includes sensitive fields such as gender and marital status. Do not use its "
            "predictions to make real loan decisions."
        )


def main() -> None:
    st.set_page_config(
        page_title="LoanIQ | Loan Approval Prediction",
        page_icon="▥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_dashboard_styles()
    page = render_sidebar()

    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(str(error))
        st.info("Open a terminal in this project folder and run `python train_model.py`.")
        return

    if page == "Prediction":
        render_prediction_page(bundle)
    elif page == "Model Info":
        render_model_info_page(bundle)
    elif page == "Dataset":
        render_dataset_page()
    else:
        render_about_page()


if __name__ == "__main__":
    main()
