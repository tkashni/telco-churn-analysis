import streamlit as st  # web app framework
import pandas as pd  # tables
import joblib  # loads model
import shap  # explains predictions
import json  # for the base_score patch
import xgboost as xgb  # needed for loading

st.set_page_config(page_title="Churn Risk Predictor", page_icon="🔮")  # page title
st.title("🔮 Customer Churn Risk Predictor")  # big title
st.markdown("Enter customer details to see churn risk **and the top reasons why**.")  # subtitle

model = joblib.load('churn_model.pkl')  # load model
cols = joblib.load('model_columns.pkl')  # load columns

def fix_base_score(m):  # same patch used in notebook
    try:  # wrap in try/except
        booster = m.get_booster() if hasattr(m, 'get_booster') else m  # get booster
        config = json.loads(booster.save_config())  # read config
        bscore = config['learner']['learner_model_param']['base_score']  # get base_score
        if isinstance(bscore, str) and bscore.startswith('['):  # if still stringified
            config['learner']['learner_model_param']['base_score'] = str(float(bscore.strip('[]')))  # convert
            booster.load_config(json.dumps(config))  # apply
    except Exception:  # if already patched, ignore
        pass  # do nothing

fix_base_score(model)  # apply patch

with st.form("customer_form"):  # form with inputs
    col1, col2 = st.columns(2)  # two-column layout
    with col1:  # left column
        tenure = st.slider("Tenure (months)", 0, 72, 12)  # tenure slider
        monthly = st.slider("Monthly Charges ($)", 18, 120, 70)  # monthly bill
        total = st.number_input("Total Charges ($)", 0, 9000, 1000)  # total charges
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])  # contract dropdown
    with col2:  # right column
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])  # internet dropdown
        techsupport = st.selectbox("Tech Support", ["Yes", "No"])  # tech support dropdown
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])  # payment
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])  # senior dropdown
    submitted = st.form_submit_button("🔍 Predict Churn Risk")  # submit button

if submitted:  # only if user clicked submit
    row = {c: 0 for c in cols}  # start all features at 0
    row['tenure'] = tenure  # set tenure
    row['MonthlyCharges'] = monthly  # set monthly
    row['TotalCharges'] = total  # set total
    row['Contract'] = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]  # encode contract
    row['SeniorCitizen'] = 1 if senior == "Yes" else 0  # encode senior
    if 'InternetService_Fiber optic' in row:  # if column exists
        row['InternetService_Fiber optic'] = 1 if internet == "Fiber optic" else 0  # flag
    if 'InternetService_DSL' in row:  # if column exists
        row['InternetService_DSL'] = 1 if internet == "DSL" else 0  # flag
    if 'TechSupport_Yes' in row:  # if column exists
        row['TechSupport_Yes'] = 1 if techsupport == "Yes" else 0  # flag

    X_input = pd.DataFrame([row])[cols]  # build 1-row dataframe
    risk = model.predict_proba(X_input)[0, 1] * 100  # churn probability %

    st.metric("Churn Risk", f"{risk:.1f}%")  # big number
    if risk > 60:  # high
        st.error("🔴 HIGH RISK — send retention offer")  # red banner
    elif risk > 30:  # medium
        st.warning("🟡 MEDIUM RISK — monitor")  # yellow banner
    else:  # low
        st.success("🟢 LOW RISK — safe for now")  # green banner

    explainer = shap.TreeExplainer(model)  # build explainer
    shap_vals = explainer.shap_values(X_input)  # compute contributions
    contrib = pd.Series(shap_vals[0], index=cols).abs().sort_values(ascending=False).head(5)  # top 5 reasons

    st.subheader("Top reasons for this prediction:")  # header
    for feature, value in contrib.items():  # loop through top 5
        st.write(f"- **{feature}** (impact: {value:.2f})")  # show each