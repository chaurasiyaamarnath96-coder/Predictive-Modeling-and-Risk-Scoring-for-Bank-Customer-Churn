Predictive Modeling and Risk Scoring for Bank Customer Churn
📌 Overview
Customer churn is a critical challenge in retail banking, directly impacting Customer Lifetime Value (CLV), revenue stability, and long-term competitiveness. Traditional churn analysis explains past attrition but fails to provide proactive insights.

This project develops a predictive churn intelligence system that assigns risk probabilities to customers before they leave. Using customer-level data from a European retail bank, we apply feature engineering, stratified sampling, and machine learning models including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, LightGBM, and XGBoost.

🎯 Objectives
Predict customer churn with high accuracy.

Generate churn probability scores.

Identify key churn drivers using explainable AI.

Reduce false positives in churn detection.

Enable scenario-based churn risk analysis.

📂 Dataset
The dataset includes demographic, financial, and behavioral attributes:

Demographics: Age, Gender, Geography

Financials: CreditScore, Balance, EstimatedSalary

Behavioral: Tenure, NumOfProducts, HasCrCard, IsActiveMember

Target Variable: Exited (1 = churned, 0 = retained)

🔎 Correlation Insights
Age positively correlated with churn (0.285)

Balance moderately correlated (0.118)

IsActiveMember negatively correlated (-0.156)

NumOfProducts negatively correlated (-0.048)

Interpretation: Older, inactive customers with higher balances are more likely to churn.

⚙️ Methodology
Data Preprocessing
Removed non-informative features (CustomerId, Surname)

No missing values present

Encoded categorical variables (Geography, Gender)

Scaled numerical features

Feature Engineering
Balance-to-Salary Ratio = Balance / (EstimatedSalary + 1)

Product Density = NumOfProducts / (Tenure + 1)

Engagement-Product Interaction = IsActiveMember × NumOfProducts

Age-Tenure Interaction = Age × Tenure

Train–Test Strategy
Stratified 80/20 split

Preserved churn distribution

Optional k-fold cross-validation

🤖 Models Implemented
Logistic Regression (baseline)

Decision Tree

Random Forest

Gradient Boosting

LightGBM

XGBoost

📊 Model Evaluation
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.8060	0.5704	0.1892	0.2841	0.7735
Decision Tree	0.7845	0.4714	0.4865	0.4788	0.6736
Random Forest	0.8620	0.7673	0.4619	0.5767	0.8530
Gradient Boosting	0.8705	0.7913	0.4939	0.6082	0.8674
LightGBM	0.8190	0.5421	0.7125	0.6157	0.8578
XGBoost	0.8490	0.6756	0.4963	0.5722	0.8311


📈 Key Findings
Gradient Boosting: Best overall performance (high accuracy & precision).

LightGBM: Highest recall, effective for identifying churners.

Logistic Regression: Underperformed due to linear assumptions.

Feature Engineering significantly improved predictive accuracy.

🔍 Model Explainability
Feature Importance: Age, Balance, IsActiveMember, NumOfProducts

SHAP Values: Local interpretability for individual predictions

Partial Dependence Plots: Nonlinear relationships (e.g., older inactive customers at higher risk)

Explainability ensures regulatory compliance and builds stakeholder trust.

📌 Recommendations
Adopt Ensemble Models for Production

Gradient Boosting for precision-focused campaigns

LightGBM for recall-focused retention strategies

Implement Risk Scoring System

Assign churn probability scores

Segment customers into risk tiers (low, medium, high)

Design Proactive Retention Campaigns

High-risk: personalized offers, loyalty rewards, direct outreach

Medium-risk: cross-sell and upsell opportunities

Leverage Explainability Tools

SHAP values & feature importance for transparency

Continuous Monitoring & Model Updating

Retrain periodically with new data

Monitor drift in customer behavior

Integrate with CRM Systems

Embed churn predictions into workflows

Automate alerts for relationship managers

Scenario-Based Risk Analysis

Simulate churn under different conditions (e.g., economic downturns)

📚 References
Breiman, L. (2001). Random Forests. Machine Learning.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.

Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS.

European Central Bank regulatory guidelines on model risk management.

IBM Watson Analytics: Telco Customer Churn dataset.

🚀 Future Work
Deploy models via FastAPI for real-time churn prediction.

Integrate with SQL databases for customer segmentation queries.

Extend to multi-class churn analysis (e.g., voluntary vs. involuntary churn).
