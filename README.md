

# 🚀 **Machine Learning **

*A collection of three end-to-end Machine Learning projects focusing on attrition prediction, pipeline debugging, and productivity optimization.*

This repository contains **three full ML workflows**, each designed to demonstrate practical, industry-level skills:

* 🧠 **Assignment 1 — Employee Attrition Prediction**
* 🔍 **Assignment 2 — ML Pipeline Debugging & Data Leakage Detection**
* ⚙️ **Assignment 3 — Productivity Feature Engineering & Optimization**

Every assignment includes:

* A complete Jupyter Notebook
* Cleaned & processed dataset
* Baseline & optimized models
* Saved artifacts
* Documentation & visualizations

---

# 📁 **Repository Structure**

```
/
├── Assignment1/
│   ├── train.ipynb
│   ├── streamlit_app.py
│   ├── artifacts/
│   └── DOCUMENTATION.md
│
├── Assignment2/
│   ├── debug_broken_notebook.ipynb
│   ├── fixed_pipeline.ipynb
│   ├── artifacts/
│   └── DOCUMENTATION.md
│
├── Assignment3/
│   ├── productivity_feature_engineering.ipynb
│   ├── artifacts/
│   └── DOCUMENTATION.md
│
├── README.md   ← (this file)
└── requirements.txt
```

---

# 🧭 **Assignment 1 — Employee Attrition Prediction**

### 🎯 **Goal**

Build a machine-learning pipeline to predict **employee attrition** and identify the main drivers behind employee turnover.

### ✔️ **Key Features**

* Full EDA + data cleaning
* Encoding of categorical variables
* Train 2+ models (Logistic Regression, RandomForest, etc.)
* Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
* SHAP explanations for interpretability
* Streamlit app for real-time prediction

### 📦 **Artifacts**

* `attrition_pipeline.joblib`
* Feature importance plots
* Model comparison table

### 📝 **Outputs**

* Probability of attrition for any employee
* Ranked list of attrition factors
* Documentation: **Assignment1/DOCUMENTATION.md**

---

# 🔍 **Assignment 2 — ML Pipeline Debugging & Data Leakage Detection**

### 🎯 **Goal**

Identify and fix a deliberately broken ML pipeline suffering from **data leakage**, **incorrect preprocessing**, and **invalid evaluation**.

### ❌ Issues Found In Broken Notebook

* Target copied into features
* Splitting after scaling (leakage)
* Cross-validation applied on the test set
* Meaningless feature engineering
* Missing imputations & no encoder separation

### ✔️ Fixes Applied

* Correct split BEFORE preprocessing
* Proper scaling inside a Pipeline
* Removal of leaking features
* Fixed cross-validation (CV on training set only)
* Added SHAP to compare leaking vs clean model
* Clean reusable ML pipeline created

### 📦 **Artifacts**

* `fixed_pipeline.joblib`
* Comparison plot: leaking vs correct ROC-AUC
* Debugging report (Assignment2/DOCUMENTATION.md)

---

# ⚙️ **Assignment 3 — Productivity Feature Engineering & Optimization**

### 🎯 **Goal**

Predict employee **productivity_score** using advanced feature engineering and model tuning.

### 🔧 **Feature Engineering Performed**

From raw columns:

* `hours_per_day` → Working hours daily
* `hours_week` → Weekly estimate
* `projects_completed` → Alias
* `tasks_per_hour` → Efficiency
* `tasks_per_day` → Output distribution
* `absence_ratio` → Absenteeism
* `work_intensity` → Normalized workload
* `efficiency_adjusted` → Penalized productivity
* `tasks_x_hours` → Interaction term
* `tasks_x_absences` → Absence impact

Unsupervised features:

* `behavior_cluster` via KMeans
* `pca_1, pca_2, pca_3` via PCA

### 📊 **Models**

* Baseline: Linear Regression
* Optimized: RandomForest / XGBoost
* Hyperparameter tuning using RandomizedSearchCV
* SelectKBest for feature selection

### 📈 **Before → After Comparison**


| Metric | Baseline      | Optimized (Tuned) | Improvement      |
| ------ | ------------- | ----------------- | ---------------- |
| MAE    | `14.6558`     | `15.7956`         | ↓ `15.4484`      |
| RMSE   | `17.2980`     | `19.9289	`        | ↓ `18.4308`      |
| R²     | `-0.0071`     | `-0.3367`         | ↓ `-0.1433`      |



### 📦 **Artifacts**

* `best_pipeline.joblib`
* `final_features.joblib`
* `metrics_comparison.joblib`
* Feature-importance dashboard

### 📘 Documentation

See: **Assignment3/DOCUMENTATION.md**

---

# 🔧 **Installation Instructions**

```
git clone https://github.com/tigerhooduday/MLtrainingandDebug
cd project/
python -m venv .venv
source .venv/bin/activate   # (Windows: .\.venv\Scripts\activate)
pip install -r requirements.txt
```

---

# ▶️ **How to Run**

### **Assignment 1**

```
cd Assignment1_Attrition
jupyter notebook train.ipynb
streamlit run streamlit_app.py
```

### **Assignment 2**

```
cd Assignment2_Debugging
jupyter notebook fixed_pipeline.ipynb
```

### **Assignment 3**

```
cd Assignment3_Productivity
jupyter notebook productivity_feature_engineering.ipynb
```

---

# 🔮 **Tech Stack**

* **Python 3.10+**
* **scikit-learn**
* **XGBoost**
* **SHAP**
* **Pandas / Numpy**
* **Matplotlib / Seaborn**
* **Pipeline + ColumnTransformer**
* **Streamlit UI** (Assignment 1)

---

# 🎓 **What This Project Demonstrates**

### ✔ Core ML Development

* Feature engineering
* Pipeline design
* Model selection & evaluation
* Hyperparameter tuning

### ✔ ML Debugging & Anti-Patterns

* Detecting leakage
* Fixing flawed pipelines
* Validating cross-validation strategies

### ✔ Explainability & Interpretability

* SHAP values
* Feature-importance dashboards

### ✔ Deployment Readiness

* Reusable Pipelines
* Saved artifacts
* Prediction interfaces

---


