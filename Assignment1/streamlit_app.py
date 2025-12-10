# streamlit_app.py
# Streamlit app for Attrition demo — robust to missing fields and provides SHAP-based local explanation
# Run: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# -------- Configuration / paths (adjust if needed) ----------
MODEL_PATH = "models/xgb_best.joblib"        # your saved pipeline (Pipeline with preprocessor + clf)
PREPROCESSOR_PATH = "models/preprocessor.joblib"  # optional: if you saved separately
FEATURE_LIST_PATH = "models/feature_columns.joblib"  # optional fallback if pipeline metadata isn't available

# -------- Helpers to get expected input columns from trained pipeline ----------
def get_expected_input_columns(model):
    """
    Tries multiple ways to discover the original feature names expected by the fitted preprocessor.
    Returns a list of column names.
    """
    # 1) If pipeline has a named preprocessor step with feature_names_in_
    try:
        preproc = model.named_steps.get('preproc', None)
        if preproc is not None and hasattr(preproc, 'feature_names_in_'):
            return list(preproc.feature_names_in_)
    except Exception:
        pass

    # 2) If pipeline has preproc and its transformers_ list contains column lists
    try:
        preproc = model.named_steps.get('preproc', None)
        if preproc is not None and hasattr(preproc, 'transformers_'):
            cols = []
            for name, trans, col_list in preproc.transformers_:
                # In some cases remainder='drop' or 'passthrough' appear; col_list might be a slice or array
                if isinstance(col_list, (list, tuple, np.ndarray)):
                    cols.extend(list(col_list))
            if cols:
                return cols
    except Exception:
        pass

    # 3) Try pipeline.feature_names_in_ (if pipeline itself records it)
    try:
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
    except Exception:
        pass

    # 4) Fallback to a saved feature list file
    if os.path.exists(FEATURE_LIST_PATH):
        try:
            return joblib.load(FEATURE_LIST_PATH)
        except Exception:
            pass

    # 5) Last resort: ask user to provide a CSV sample or abort
    raise RuntimeError(
        "Could not infer expected feature columns from the saved model. "
        "Save the feature list during training (joblib.dump(feature_cols,...)) or re-train with preprocessor.feature_names_in_ available."
    )

# -------- Safe default values for features (set sensible defaults for missing UI inputs) ----------
# You MUST adapt these defaults to your dataset semantics. They are conservative placeholders.
DEFAULTS = {
    'age': 30,
    'gender': 'Male',
    'education': 'Bachelor',
    'department': 'Sales',
    'job_role': 'Sales Executive',
    'years_at_company': 2,
    'promotions': 0,
    'overtime': 'No',
    'performance_rating': 3,
    'monthly_income': 30000
}

# Optional: map ordinals used during training (if you used ordinal encoding)
ORDINAL_MAPS = {
    # Example: 'education': ['High School', 'Bachelor', 'Master', 'PhD']
    # Fill if you used OrdinalEncoder for some columns in training.
}

# -------- Load model (pipeline) ----------
@st.cache_resource(show_spinner=False)
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Train & save the pipeline first.")
        st.stop()
    model = joblib.load(path)
    return model

model = load_model()

# Discover expected input columns
try:
    expected_columns = get_expected_input_columns(model)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# For transparency: show columns in sidebar (collapsible)
with st.sidebar.expander("Model expected features (preview)"):
    st.write(f"Total columns expected: {len(expected_columns)}")
    st.write(expected_columns[:40])  # show up to 40 to avoid overload

# -------- UI layout ----------
st.set_page_config(page_title="Attrition Predictor", layout="wide")
st.title("Employee Attrition — Interactive Demo")
st.markdown("Provide employee details in the left panel. The model predicts probability of leaving (attrition).")

left, right = st.columns([1, 1.2])

with left:
    st.header("Employee input")
    # Provide explicit inputs for the main features (these are the ones users typically care about).
    age = st.number_input("Age", min_value=16, max_value=100, value=DEFAULTS['age'])
    gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'], index=0 if DEFAULTS['gender']=='Male' else 1)
    education = st.selectbox("Education", options=['High School', 'Bachelor', 'Master', 'PhD'], index=1)
    department = st.selectbox("Department", options=['Sales', 'Research & Dev', 'HR', 'Finance', 'IT', 'Operations'], index=0)
    job_role = st.text_input("Job role (free text)", value=DEFAULTS['job_role'])
    years_at_company = st.number_input("Years at company", min_value=0.0, max_value=50.0, value=float(DEFAULTS['years_at_company']))
    promotions = st.number_input("Promotions", min_value=0, max_value=20, value=int(DEFAULTS['promotions']))
    overtime = st.selectbox("Overtime", options=['Yes', 'No'], index=1 if DEFAULTS['overtime']=='No' else 0)
    performance_rating = st.slider("Performance rating", min_value=1, max_value=5, value=int(DEFAULTS['performance_rating']))
    monthly_income = st.number_input("Monthly income", min_value=0.0, value=float(DEFAULTS['monthly_income']))

    # Extra: allow user to upload CSV with single-row sample to preserve exact column names & extra features
    st.markdown("---")
    st.markdown("Advanced: upload CSV with a single row to provide all features exactly as in training data.")
    uploaded = st.file_uploader("Single-row CSV (optional)", type=['csv'])
    use_uploaded = uploaded is not None

    if use_uploaded:
        try:
            uploaded_df = pd.read_csv(uploaded)
            if uploaded_df.shape[0] != 1:
                st.warning("CSV must contain exactly one row. Only the first row will be used.")
            uploaded_row = uploaded_df.iloc[0].to_dict()
        except Exception as ex:
            st.error("Failed to read uploaded CSV: " + str(ex))
            uploaded_df = None

    predict_button = st.button("Predict attrition probability")

with right:
    st.header("Prediction & Explanation")
    prob_placeholder = st.empty()
    bar_placeholder = st.empty()
    shap_placeholder = st.empty()
    details_placeholder = st.expander("Details (show raw model inputs & debug info)", expanded=False)

# -------- Build a complete single-row DataFrame matching expected_columns ----------
def build_input_row(exp_cols):
    """
    Build a one-row DataFrame that contains all expected columns.
    Priority:
      1) If user uploaded CSV, use upload values for matching columns.
      2) Otherwise use explicit UI fields for core attributes.
      3) For missing columns use DEFAULTS or generic safe values (NaN otherwise).
    """
    row = {}
    # start with defaults (safe)
    for c in exp_cols:
        if c in DEFAULTS:
            row[c] = DEFAULTS[c]
        else:
            row[c] = np.nan

    # overlay explicit UI fields
    ui_map = {
        'age': age,
        'gender': gender,
        'education': education,
        'department': department,
        'job_role': job_role,
        'years_at_company': years_at_company,
        'promotions': promotions,
        'overtime': overtime,
        'performance_rating': performance_rating,
        'monthly_income': monthly_income
    }
    for k, v in ui_map.items():
        if k in exp_cols:
            row[k] = v

    # overlay uploaded CSV row if present (highest priority)
    if use_uploaded and uploaded is not None:
        for k, v in uploaded_row.items():
            if k in exp_cols:
                row[k] = v

    # final coercion: ensure types are plausible: strings -> str, numbers -> float
    for c in exp_cols:
        val = row[c]
        if pd.isna(val):
            # try fallback
            if c in DEFAULTS:
                row[c] = DEFAULTS[c]
            else:
                # numeric-ish heuristic: if column name contains keywords assume numeric
                if any(token in c.lower() for token in ['age','income','years','promot','rating','count','num']):
                    row[c] = 0
                else:
                    row[c] = 'Missing'
        else:
            # coerce types: strings as str; keep numbers numeric
            if isinstance(val, (int, float, np.integer, np.floating)):
                row[c] = val
            else:
                row[c] = str(val)
    return pd.DataFrame([row])

# -------- Prediction flow ----------
if predict_button:
    # prepare input
    try:
        input_df = build_input_row(expected_columns)
    except Exception as ex:
        st.error("Failed to construct input row: " + str(ex))
        st.stop()

    # show debug info if user wants
    with details_placeholder:
        st.write("Input row passed to model (first 50 cols):")
        st.write(input_df.iloc[:, :50].to_dict(orient='records')[0])

    # compute probability and show UI
    try:
        # Model may be a pipeline or only an estimator; handle both.
        if hasattr(model, 'predict_proba'):
            proba = float(model.predict_proba(input_df)[:, 1][0])
        else:
            # If model is only classifier without preprocessor, attempt to use a separately saved preprocessor
            if os.path.exists(PREPROCESSOR_PATH):
                pre = joblib.load(PREPROCESSOR_PATH)
                X_trans = pre.transform(input_df)
                clf = joblib.load(MODEL_PATH)
                proba = float(clf.predict_proba(X_trans)[:,1][0])
            else:
                st.error("Model does not appear to be a Pipeline. Save a pipeline (preprocessor + clf) or supply PREPROCESSOR_PATH.")
                st.stop()

    except Exception as ex:
        st.error("Model prediction failed: " + str(ex))
        st.stop()

    # show big number + progress bar
    prob_pct = proba * 100
    prob_placeholder.markdown(f"## Attrition probability: **{prob_pct:.1f}%**")
    bar_placeholder.progress(min(int(prob_pct), 100))

    # color-coded advice quick heuristics
    if proba >= 0.6:
        st.warning("High risk — investigate. Consider outreach, review overtime, check promotion history.")
    elif proba >= 0.3:
        st.info("Moderate risk — monitor employee and consider manager check-in.")
    else:
        st.success("Low predicted attrition risk.")

    # -------- Local explainability: SHAP (top features) ----------
    try:
        # We need the transformed input that the learner sees for shap TreeExplainer on the fitted classifier.
        # If model is a Pipeline, get last step (estimator) and preprocessor (to build X_trans)
        if hasattr(model, 'named_steps'):
            preproc = model.named_steps.get('preproc', None)
            clf = model.named_steps.get(list(model.named_steps.keys())[-1])
            if preproc is not None:
                # transform to the features used by the model (numerical matrix)
                X_trans = preproc.transform(input_df)
                # Try to get feature names for columns after preprocessing
                try:
                    feature_names = preproc.get_feature_names_out()
                except Exception:
                    # fallback: use list of indices
                    feature_names = [f"f{i}" for i in range(X_trans.shape[1])]
                X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
            else:
                # Pipeline without named preproc — attempt to transform directly (rare)
                X_trans_df = input_df.copy()
                clf = model
        else:
            st.info("SHAP not available for this model structure.")
            X_trans_df = None
            clf = model

        # Compute SHAP values for the single row for tree models
        if X_trans_df is not None and hasattr(clf, 'predict_proba'):
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_trans_df)
            # shap_vals shape is (1, n_features) or list if multiclass; handle binary
            if isinstance(shap_vals, list):
                shap_local = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
            else:
                shap_local = shap_vals[0]

            # Build a small DF of top contributors
            shap_df = pd.DataFrame({
                'feature': X_trans_df.columns,
                'shap_value': shap_local
            })
            shap_df['abs_shap'] = shap_df['shap_value'].abs()
            shap_df = shap_df.sort_values('abs_shap', ascending=False).head(10)

            # show bar chart
            shap_placeholder.subheader("Top local feature drivers (SHAP)")
            fig, ax = plt.subplots(figsize=(6,3))
            shap_df = shap_df.sort_values('shap_value')  # so bars align left->right
            colors = ['#d9534f' if v<0 else '#5cb85c' for v in shap_df['shap_value']]
            ax.barh(shap_df['feature'], shap_df['shap_value'], color=colors)
            ax.set_xlabel("SHAP value (positive -> increases attrition prob)")
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as ex:
        st.info("SHAP explanation skipped due to: " + str(ex))

    # finished
    st.caption("Note: This demo uses a pre-trained model. Model outputs are probabilistic estimates, not certainties.")
