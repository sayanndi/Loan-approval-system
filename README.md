<div align="center">

# 🏦 Loan Approval Prediction System


<br/>

> *Predict whether a loan application gets approved or rejected — using a complete end-to-end Machine Learning pipeline.*

<br/>

```
 RAW DATA  ──►  EDA  ──►  ENCODING  ──►  SCALING  ──►  MODEL  ──►  RESULT
```

</div>

---

## 📌 Project Overview

This project builds a **binary classification model** to predict loan approval outcomes based on applicant details. It covers the complete ML workflow — from raw messy data all the way to model evaluation and feature engineering — using a real-world dataset from Kaggle.

---

## 📂 Dataset

| Detail | Info |
|--------|------|
| 📦 Source | [Kaggle — Loan Prediction Dataset](https://www.kaggle.com/) |
| 🎯 Target | `Loan_Status` (Approved / Rejected) |
| 📋 Features | Gender, Marital Status, Income, Loan Amount, Credit History, etc. |

---

## 🔄 Complete ML Pipeline

### `Step 1` — 🧹 Handling Missing Values
- Identified null values across numerical and categorical columns
- Applied **mean imputation** for numerical columns (e.g., `LoanAmount`)
- Applied **mode imputation** for categorical columns (e.g., `Gender`, `Self_Employed`)
- Preserved data integrity without losing significant rows

---

### `Step 2` — 📊 Exploratory Data Analysis (EDA)

Performed thorough visual analysis to understand feature distributions and spot anomalies.

| Plot | Purpose |
|------|---------|
| 📊 **Histogram Plot** (`histplot`) | Distribution of numerical features — income, loan amount, loan term |
| 📦 **Box Plot** (`boxplot`) | Outlier detection, spread across categories, skewness check |

**Key Findings:**
- Strong imbalance noted in `Loan_Status` classes
- `ApplicantIncome` and `LoanAmount` showed significant right skew
- Applicants with positive `Credit_History` had a much higher approval rate

---

### `Step 3` — 🔠 Encoding Categorical Variables
- Converted categorical features into numeric format using **Label Encoding**
- Columns encoded: `Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`, `Loan_Status`

---

### `Step 4` — 🔥 Correlation Heatmap

Generated a **correlation heatmap** to identify feature relationships with the target variable.

| Relationship | Feature | Insight |
|---|---|---|
| ✅ Most Positive | `Credit_History` | Strong indicator of approval |
| ❌ Most Negative | `LoanAmount` | Higher loan amount slightly reduces approval odds |

---

### `Step 5` — ✂️ Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### `Step 6` — ⚖️ Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

- Normalized all numerical features to bring them onto the same scale
- Especially critical for **KNN**, which is sensitive to feature magnitude

---

## 🤖 Model Evaluation

Three classifiers were trained and compared using **Precision Score** as the primary metric.

> 🎯 **Why Precision?** — To minimize false approvals (wrongly approving ineligible applicants), precision is the most important metric in this context.

<br/>

| Model | Precision Score | Result |
|-------|:--------------:|--------|
| ✅ **Logistic Regression** | **Best** | 🏆 Winner |
| K-Nearest Neighbors (KNN) | Lower | — |
| Naive Bayes | Lower | — |

<br/>

**🏆 Logistic Regression** was selected as the best-performing model based on precision score.

---

## 🛠️ Feature Engineering

After initial model evaluation, feature engineering was applied to explore further improvements:

- Created new derived features based on domain knowledge and EDA findings
- Transformed existing features (e.g., log transformation for skewed distributions)
- Re-trained and re-evaluated all three models

**📌 Outcome:**

> Despite applying feature engineering, the precision scores showed **no significant improvement**. This confirms that the original features — after cleaning, encoding, and scaling — already captured the core predictive signal well. Logistic Regression remained the best model.

---

## 🧰 Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

---


---


## 💡 Key Takeaways

- 📌 **Credit History** is the single strongest predictor of loan approval
- 📌 **Logistic Regression** works best for this linearly separable binary problem
- 📌 **Feature Engineering** didn't improve results — confirming the raw features were already informative
- 📌 **Precision** is more meaningful than accuracy for imbalanced classification tasks

---

<div align="center">


<br/>

⭐ **If you found this project useful, drop a star!** ⭐

</div>
