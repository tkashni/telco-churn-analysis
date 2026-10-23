# 🔮 Customer Churn Prediction — Telco

## 💼 The Business Problem
We lose **~$139,000/month (~30.5% of MRR)** to customer churn. This project identifies the top drivers and ships a **live risk-scoring tool** for the retention team.

## 🎯 Key Insights
- **Month-to-month** customers churn at **42.7%** — vs. only **2.8%** for two-year customers.
- Customers in their **first 12 months** churn at **47.7%** — 5× the rate of 49–72-month customers.
- **Fiber optic** customers without tech support churn the fastest.

## ✅ 3 Actionable Recommendations
1. **Contract Conversion Campaign** — Offer 10% off for 3 months to move high-risk month-to-month customers onto 1-year contracts.
2. **90-Day Onboarding Program** — Day-30 check-in call + Day-60 tutorial to reduce early-lifecycle churn.
3. **Free Tech Support Trial** — 3 months free for Fiber Optic customers (highest churn segment).

## 🛠️ Tech Stack
Python · XGBoost · SMOTE · SHAP · Streamlit

## 📈 Model Performance
- **ROC-AUC: 0.829**
- **Recall (churners): 82%**
- **Explainability: SHAP per-customer**

## 🚀 Try It Live
👉 [Interactive Predictor](YOUR_STREAMLIT_URL_HERE)

## 📊 Screenshots
![Contract Churn](screenshots/contract.png)
![Tenure Churn](screenshots/tenure.png)
![SHAP](screenshots/shap.png)
![App](screenshots/app.png)

## 📁 Repo Structure
- `Telco Customer Churn.ipynb` — EDA + model
- `app.py` — Streamlit app
- `churn_model.pkl` — trained XGBoost model
- `requirements.txt` — packages