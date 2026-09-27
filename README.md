\# Fraud Detection System



An end-to-end machine learning fraud detection system built to identify potentially fraudulent credit card transactions using feature engineering, model selection, threshold optimization, and deployment through Streamlit and FastAPI.



\## 🚀 Project Overview



Fraudulent transactions are highly imbalanced compared with legitimate transactions, making fraud detection a challenging machine learning problem.



This project implements a complete ML inference workflow:



\- Data preprocessing

\- Feature engineering

\- Categorical encoding

\- Feature selection

\- Model selection

\- Hyperparameter tuning

\- Cross-validation

\- Model comparison

\- Decision-threshold optimization

\- Streamlit web application

\- FastAPI REST API



The final deployed model is a \*\*Logistic Regression model with log-transformed transaction amounts\*\*.



\---



\## 🎯 Problem Statement



The objective is to classify financial transactions as:



\- \*\*Legitimate\*\*

\- \*\*Fraudulent\*\*



The major challenge is severe class imbalance, where fraudulent transactions represent only a small fraction of the available transactions.



Therefore, evaluation focuses on metrics such as:



\- Precision

\- Recall

\- F1-score

\- PR-AUC

\- ROC-AUC



rather than relying only on accuracy.



\---



\## 📊 Dataset



The project uses separate training and testing datasets.



| Dataset | Transactions | Features |

|---|---:|---:|

| Training | 1,296,675 | 23 |

| Testing | 555,719 | 23 |



\### Class Distribution



The training dataset contains:



\- Legitimate transactions: \*\*1,289,169\*\*

\- Fraudulent transactions: \*\*7,506\*\*

\- Fraud rate: \*\*0.579%\*\*



This severe imbalance makes fraud detection significantly harder than ordinary binary classification.



\### Temporal Split



The training data covers approximately:



`2019-01-01 → 2020-06-21`



The test data covers:



`2020-06-21 → 2020-12-31`



The temporal test set provides an out-of-time evaluation rather than randomly mixing future transactions into training.



\---



\# 🛠️ Machine Learning Pipeline



```text

Raw Transaction Data

&#x20;       │

&#x20;       ▼

Data Cleaning

&#x20;       │

&#x20;       ▼

Feature Engineering

&#x20;       │

&#x20;       ├── Transaction Amount

&#x20;       ├── Customer Age

&#x20;       ├── Card Transaction Count

&#x20;       ├── Geographic Distance

&#x20;       └── Cyclical Time Features

&#x20;       │

&#x20;       ▼

Categorical Encoding

&#x20;       │

&#x20;       ▼

Feature Scaling

&#x20;       │

&#x20;       ▼

Feature Selection

&#x20;       │

&#x20;       ▼

Model Selection

&#x20;       │

&#x20;       ▼

Hyperparameter Tuning

&#x20;       │

&#x20;       ▼

Threshold Optimization

&#x20;       │

&#x20;       ▼

Fraud Probability

&#x20;       │

&#x20;       ▼

Fraud / Legitimate

