import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
X_test = pd.read_csv("X_test.csv")
# -------------------------------
# Load Pretrained Models
# -------------------------------
# This loads the dictionary of all fitted pipelines
models = joblib.load("all_models12.pkl")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Customer Features")

age = st.sidebar.slider("Age", 18, 70, 35)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)
balance = st.sidebar.slider("Balance", 0.0, 250000.0, 50000.0)
products = st.sidebar.slider("NumOfProducts", 1, 4, 2)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
salary = st.sidebar.slider("Estimated Salary", 0.0, 200000.0, 50000.0)
geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
is_active = st.sidebar.selectbox("Is Active Member", [0,1])
has_card = st.sidebar.selectbox("Has Credit Card", [0,1])

# Model selector
selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
clf = models[selected_model_name]   # load fitted pipeline

# -------------------------------
# Construct Input Row
# -------------------------------
input_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": products,
    "HasCrCard": has_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary,
    "BalanceSalaryRatio": balance/(salary+1),
    "ProductDensity": products/(tenure+1),
    "EngagementProduct": is_active*products,
    "AgeTenureInteraction": age*tenure,
    "Year": 2026
}])

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Churn Risk Calculator",
    "Probability Distribution",
    "Feature Importance",
    "What-If Simulator"
])

# -------------------------------
# 1. Churn Risk Calculator
# -------------------------------
with tab1:
    prob = clf.predict_proba(input_data)[0,1]
    st.metric("Predicted Churn Probability", f"{prob:.2f}")

# -------------------------------
# 2. Probability Distribution
# -------------------------------
with tab2:
    # Example: use model on test set if available
    y_prob = clf.predict_proba(X_test)[:,1]
    #y_prob = np.random.rand(1000)  # demo
    fig, ax = plt.subplots()
    sns.histplot(y_prob, bins=20, kde=True, ax=ax)
    st.pyplot(fig)

# -------------------------------
# 3. Feature Importance Dashboard
# -------------------------------
with tab3:
    if hasattr(clf.named_steps["classifier"], "feature_importances_"):
        importances = clf.named_steps["classifier"].feature_importances_
        feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
        fi = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi = fi.sort_values("Importance", ascending=False).head(15)
        st.bar_chart(fi.set_index("Feature"))
    else:
        st.write("Feature importance not available for this model.")

# -------------------------------
# 4. What-If Scenario Simulator
# -------------------------------
with tab4:
    st.write("Adjust sliders in the sidebar to simulate changes.")
    st.write(f"Current churn probability: {prob:.2f}")
