# Customer Churn Prediction (Advanced)

This project focuses on predicting **customer churn** — whether a customer will leave a telecom service — using advanced **machine learning techniques** such as **XGBoost**.  
The analysis provides insights into what drives customer retention and identifies the most influential factors behind churn behavior.

---

## Project Overview

The objective of this project is to build a predictive model that identifies customers likely to cancel their subscriptions.  
By analyzing customer demographics, billing, and contract details, the project demonstrates how **data-driven insights** can reduce churn and improve business decisions.

---

## Dataset Description

The dataset used in this project is based on a **telecom customer dataset** containing details such as:
- Customer demographics (gender, senior citizen, partner, dependents)
- Service information (phone, internet, contract type, payment method)
- Billing details (monthly charges, total charges)
- Customer status (churn or not)

**Dataset shape:** `(10000, 12)`  
**Churn distribution:**

This shows an **imbalanced dataset**, where most customers did not churn.

---

## Modeling & Methods

1. **Data Preprocessing**
   - Missing value imputation
   - Encoding categorical features
   - Feature scaling

2. **Model Used:** `XGBoostClassifier`
   - Chosen for its ability to handle non-linear relationships and feature interactions.
   - Optimized using grid search for parameters:
     ```python
     Best params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01}
     ```

3. **Train-Test Split:**
   - 80% training, 20% testing

---

## Model Performance

| Metric | Value |
|--------|--------|
| **Accuracy** | 0.80 |
| **ROC AUC** | 0.764 |
| **Precision (Churn=1)** | 1.00 |
| **Recall (Churn=1)** | 0.00 |
| **F1-score (Churn=1)** | 0.00 |

### Interpretation:
- The model achieves **80% overall accuracy**, but **recall for churners is low**, indicating that it predicts most customers as non-churners.
- This is common in **imbalanced datasets** and can be improved with techniques like **SMOTE** or **class weighting**.

---

## Feature Importance (XGBoost)

The following chart displays the **top features influencing churn** decisions according to the XGBoost model:

![XGBoost Feature Importance](c7cb0365-67d1-40e4-8710-5688bfe285ce.png)

### What the Chart Shows

Each bar represents a feature, and its length shows how much it contributed to the model’s predictions.

| Feature | Meaning | Insight |
|----------|----------|----------|
| **Contract_Two year** | Customer on a 2-year plan | Strongly reduces churn — long-term contracts promote loyalty. |
| **Contract_One year** | Customer on a 1-year plan | Also reduces churn but less than 2-year contracts. |
| **MonthlyCharges** | Monthly bill amount | Higher bills slightly increase churn risk. |
| **Tenure** | Time with company | Longer tenure = less churn. |
| **Partner / Dependents / SeniorCitizen** | Customer demographics | Minor impact compared to contract type. |

### Interpretation of the Graph:
- **Contract type** dominates the prediction — customers with **long-term contracts** are least likely to leave.
- **Monthly charges** and **tenure** have moderate effects.
- **Demographics and payment methods** contribute minimally.
  
In summary, focusing on **contract retention programs** and **customer engagement** can significantly reduce churn.

---

## Visualization Insights

- The **feature importance plot** indicates contract duration as the most predictive factor.
- The **ROC-AUC score of 0.764** shows good overall discrimination ability between churners and non-churners.
- Further improvement is possible with **SMOTE oversampling** or **threshold tuning**.

---

## Technologies Used

- **Python**
- **Pandas**, **NumPy**, **Scikit-learn**
- **XGBoost**
- **Matplotlib**, **Seaborn**

---

## Project Structure
```
Customer_Churn_Advanced/
│
├── Customer_Churn_Advanced.ipynb       # Main notebook
├── data_customer_churn.csv             # Dataset file
├── README.md                           # Project documentation
└── requirements.txt                    # Required dependencies
```

