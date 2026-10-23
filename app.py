import streamlit as st
import pandas as pd
import joblib
import shap
import json
import xgboost as xgb

st.set_page_config(page_title="Churn Risk Predictor", page_icon="🔮")
st.title("🔮 Customer Churn Risk Predictor")
st.markdown("Enter customer details to see churn risk **and the top reasons why**.")

# ---------- Load artifacts ----------
model = joblib.load('churn_model.pkl')
cols = joblib.load('model_columns.pkl')

def fix_base_score(m):
    """Patch XGBoost base_score if serialized as '[7.3E-1]'."""
    try:
        booster = m.get_booster() if hasattr(m, 'get_booster') else m
        config = json.loads(booster.save_config())
        bscore = config['learner']['learner_model_param']['base_score']
        if isinstance(bscore, str) and bscore.startswith('['):
            config['learner']['learner_model_param']['base_score'] = str(
                float(bscore.strip('[]'))
            )
            booster.load_config(json.dumps(config))
    except Exception:
        pass

fix_base_score(model)

# ---------- Encoders (match sklearn LabelEncoder alphabetical ordering) ----------
CONTRACT_MAP     = {"Month-to-month": 0, "One year": 1, "Two year": 2}
INTERNET_MAP     = {"DSL": 0, "Fiber optic": 1, "No": 2}
TECHSUPPORT_MAP  = {"No": 0, "No internet service": 1, "Yes": 2}
PAYMENT_MAP      = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)":   1,
    "Electronic check":          2,
    "Mailed check":              3,
}
YESNO_MAP        = {"No": 0, "Yes": 1}

# ---------- Form ----------
with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure      = st.slider("Tenure (months)", 0, 72, 12)
        monthly     = st.slider("Monthly Charges ($)", 18, 120, 70)
        total       = st.number_input("Total Charges ($)", 0, 9000, 1000)
        contract    = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless   = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        internet    = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        techsupport = st.selectbox(
            "Tech Support",
            ["No", "No internet service", "Yes"],
            help="Select 'No internet service' if the customer has no internet.",
        )
        payment     = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        senior      = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner     = st.selectbox("Partner", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Predict Churn Risk")

# ---------- Predict ----------
if submitted:
    row = {c: 0 for c in cols}

    # Numeric
    row['tenure']         = tenure
    row['MonthlyCharges'] = monthly
    row['TotalCharges']   = total

    # Encoded categoricals (single integer columns, NOT one-hot)
    row['Contract']         = CONTRACT_MAP[contract]
    row['InternetService']  = INTERNET_MAP[internet]
    row['TechSupport']      = TECHSUPPORT_MAP[techsupport]
    row['PaymentMethod']    = PAYMENT_MAP[payment]
    row['PaperlessBilling'] = YESNO_MAP[paperless]
    row['Partner']          = YESNO_MAP[partner]
    row['SeniorCitizen']    = 1 if senior == "Yes" else 0

    # Sensible defaults for fields not on the form
    row['gender']           = 1   # Male
    row['Dependents']       = 0   # No
    row['PhoneService']     = 1   # Yes (majority of customers)
    row['MultipleLines']    = 0   # No
    row['OnlineSecurity']   = 0   # No
    row['OnlineBackup']     = 0   # No
    row['DeviceProtection'] = 0   # No
    row['StreamingTV']      = 0   # No
    row['StreamingMovies']  = 0   # No

    X_input = pd.DataFrame([row])[cols]
    risk = model.predict_proba(X_input)[0, 1] * 100

    st.metric("Churn Risk", f"{risk:.1f}%")
    if risk > 60:
        st.error("🔴 HIGH RISK — send retention offer")
    elif risk > 30:
        st.warning("🟡 MEDIUM RISK — monitor")
    else:
        st.success("🟢 LOW RISK — safe for now")

    # ---------- SHAP explanation ----------
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_input)

    # Normalize return shape across SHAP versions
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    if hasattr(sv, 'ndim') and sv.ndim == 3:
        sv = sv[:, :, 1]   # newer SHAP → take positive-class slice
    sv = sv[0]             # first (only) row → shape (n_features,)

    contrib = pd.Series(sv, index=cols)
    contrib = contrib.reindex(
        contrib.abs().sort_values(ascending=False).index
    ).head(5)

    st.subheader("Top reasons for this prediction:")
    for feature, value in contrib.items():
        direction = "↑ increases risk" if value > 0 else "↓ decreases risk"
        st.write(f"- **{feature}** — {direction} (impact: {abs(value):.2f})")

    with st.expander("Assumptions (fields not on the form)"):
        st.caption(
            "Gender=Male, Dependents=No, PhoneService=Yes, "
            "MultipleLines=No, OnlineSecurity=No, OnlineBackup=No, "
            "DeviceProtection=No, StreamingTV=No, StreamingMovies=No. "
            "Expand the form later if you want to expose these."
        )
