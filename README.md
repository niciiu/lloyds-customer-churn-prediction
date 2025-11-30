# Customer Churn Prediction – Lloyds Banking Group

**machine learning project for customer churn prediction, including EDA, feature engineering, data preprocessing, model training, evaluation, and business insights.**

This project simulates the workflow of a Data Scientist working within a banking analytics team. The primary objective is to develop a predictive model capable of identifying customers at high risk of churn so the business can take proactive retention actions.


## 1. Project Overview

Customer churn is a critical business challenge in the banking industry, where retaining existing customers is significantly more cost-effective than acquiring new ones. This project focuses on:

* Understanding customer behavior through Exploratory Data Analysis (EDA)
* Engineering meaningful features such as recency metrics and interaction behavior
* Building and comparing machine learning models (Random Forest vs. XGBoost)
* Evaluating model performance using metrics suitable for imbalanced datasets
* Translating model results into actionable business recommendations

The project provides a complete machine learning workflow designed to support real-world decision-making.


## 2. Project Structure

```
lloyds-customer-churn-prediction/
│
├── data/
│   ├── raw/                 # Raw input files 
│   ├── processed/           # Cleaned dataset used for modeling
│
├── notebooks/
│   ├── CustChurnEDA.ipynb   # Exploratory Data Analysis
│   ├── CustChurnML.ipynb
│
├── models/
│   ├── best_model.pkl       # Saved best-performing model
│
├── report/
│   ├──picture/              # EDA and Machine Learning Visual
│   ├── Customer Churn Prediction (1).pdf
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 3. Dataset Description

The cleaned dataset contains **1,000 customers** with the following key features:

*Customer Attributes*
- Age
- Gender
- Marital Status
- Income Level

*Account & Transaction Behavior*
- TransactionID
- AmountSpent
- LoginFrequency
- Service usage channels
- Product category interactions

*Interaction History*
- InteractionID
- Interaction Type
- Resolution Status
- Recency features (days since last interaction, last transaction, last login)

*Target Variable*
- ChurnStatus** (0 = active, 1 = churned)

## 4. Exploratory Data Analysis (EDA)

Key insights from the EDA:
- High-value customers (high spending frequency) tend to show lower churn likelihood.
- Customers with low engagement (low login frequency, fewer interactions) show higher churn indicators.
- Unresolved service interactions correlate strongly with churn.
- Recency features (days since last login) show clear separation between churned vs. non-churned customers.
A sample visualization from the EDA is included in the report.


## 5. Data Preprocessing & Feature Engineering

### Steps completed:
- Parsing date fields and converting to numeric recency features
- One-hot encoding categorical variables
- Standard scaling for numerical features
- Oversampling minority class using SMOTE
- Train-test split using stratified sampling

### Final feature set includes:
- Demographic variables
- Behavioral variables
- Recency metrics
- Interaction intensity variables
- Encoded categories

## 6. Model Development
Two machine learning models were trained and compared:

1. Random Forest
- Class-weight balance
- SMOTE oversampling
- GridSearchCV for tuning hyperparameters

2. XGBoost Classifier
- Handles imbalance using scale_pos_weight
- Gradient boosting approach
- Hyperparameter tuning via GridSearchCV

Both models used a pipeline integrating:
- Preprocessing
- SMOTE
- Classifier

Evaluation metrics focused on imbalanced classification, such as:
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## 7. Model Performance Summary

| Metric    | Random Forest | XGBoost  |
| --------- | ------------- | -------- |
| Accuracy  | 0.78          | 0.77     |
| Precision | 0.33          | 0.31     |
| Recall    | 0.07          | 0.10     |
| F1-score  | 0.12          | 0.15     |
| ROC-AUC   | **0.51**      | 0.46     |

### Interpretation

Random Forest provided more stable performance overall.
XGBoost detected more churners (higher recall) but with lower reliability and stability.
Both models struggle due to limited signal in the dataset and noticeable class imbalance.

### Business Recommendation

Use **Random Forest** as the primary model due to its better ROC-AUC stability and interpretability.
XGBoost may be useful as a secondary model for experiments requiring higher sensitivity at the cost of accuracy.


## 8. Business Insights & Recommendations
Based on model outputs and feature importance:

### Key Drivers of Churn
- Low login frequency
- High days since last interaction
- Unresolved service issues
- Low transaction activity
- Limited product engagement

### Recommended Actions

1. **Proactive Retention Campaigns**
   Target customers with low recency scores and low activity levels.

2. **Improve Service Resolution Processes**
   Unresolved cases strongly correlate with churn.

3. **Increase Personalised Engagement**
   Provide tailored offers to customers with reduced transaction frequency.

4. **Product Education & Onboarding**
   Encourage broader usage of banking products (web/app services, online banking).

---

## 9. Files Included in This Repository

* **best_model.pkl** → the final trained model
* **Cleaned dataset** (processed)
* **Notebooks** for EDA, preprocessing, and modeling
* **PDF report** containing full narrative analysis
---

## 10. How to Run the Project

Install dependencies:

```
pip install -r requirements.txt
```

Run training:

```
python src/model_training.py
```

Open Jupyter Notebook:

```
jupyter notebook
```

---

## 11. Next Steps

Recommendation improvements:
- Add SHAP explainability for deeper driver interpretation
- Implement advanced techniques (LightGBM, CatBoost)
- Build automated ML pipeline
- Deploy model via FastAPI or Streamlit
- Integrate experiment tracking using MLflow
