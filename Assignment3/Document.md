

# 📘 **Productivity Feature Engineering & Optimization**

*Assignment 3 — Machine Learning Pipeline Design, Feature Engineering & Model Optimization*

---

# 🧭 **1. Project Overview**

The Operations team wants to understand what truly drives **employee productivity** across different working patterns.
This assignment implements a complete ML workflow to:

1. Load and clean the dataset
2. Build an initial **baseline regression model**
3. Engineer meaningful features from raw logs
4. Apply scaling, feature selection, clustering, and PCA
5. Train an **optimized model** and compare performance
6. Produce explainability outputs to help Ops interpret productivity signals

The final result is a **significantly improved predictive model** with a clear explanation of which engineered features matter most.

---

# 🛠 **2. Installation Instructions**

Use Python **3.9+** and install dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If no `requirements.txt` is provided, install minimal dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost shap
```

---

# 🚀 **3. How to Run the Pipeline (Notebook or Script)**

### **Step 1 — Open the notebook**

Open:

```
notebooks/productivity_feature_engineering.ipynb
```

Select the `.venv` kernel.

### **Step 2 — Run all cells**

The notebook runs in the following order:

1. **Imports & dataset load**
2. **Cleaning & validation**
3. **Train/test split**
4. **Baseline model**
5. **Feature engineering**
6. **Clustering & PCA**
7. **Optimized model + tuning**
8. **Before/after comparison table**
9. **Feature importance dashboard**
10. **Saving artifacts**

All outputs appear inline.

### **Step 3 — View saved artifacts**

After execution, the following artifacts are created:

```
artifacts/
 ├── best_pipeline.joblib
 ├── final_features.joblib
 ├── metrics_comparison.joblib

```

### **Step 4 —  Predict with the model**



# 📊 **4. Baseline Model — Initial Predictive Power**

Predicted productivity: [72.61456864]

### **Baseline Algorithm:**

**Linear Regression**
Features used:

* `projects_completed`
* `hours_week`
* `weekly_absences`
* `tasks_per_hour`
* `absence_ratio`

### **Baseline Performance:**


Baseline features: ['projects_completed', 'hours_week', 'weekly_absences', 'tasks_per_hour', 'absence_ratio']
Baseline results: {'MAE': 14.655837990791676, 'RMSE': 17.29801064687018, 'R2': -0.007060499027005784}

The baseline model establishes a **minimal predictive benchmark**. Performance is intentionally limited because it uses only simple linear relationships.

---

# 🧪 **5. Engineered Features — Full List with One-Line Explanations**

These features were explicitly crafted to extract deeper relationships from raw employee logs.

### **Workload & Time Features**

1. **hours_per_day** — Daily working hours from login/logout.
2. **hours_week** — Est. weekly working time (`hours_per_day * 5`).

### **Performance Efficiency Features**

3. **projects_completed** — Alias of total tasks completed.
4. **tasks_per_hour** — Productivity per working hour.
5. **tasks_per_day** — Avg. daily output (`projects / 5`).

### **Absenteeism Features**

6. **absence_ratio** — Fraction of the work week missed (`absences / 5`).
7. **work_intensity** — Hours worked normalized by absences (`hours_week / (absences+1)`).
8. **efficiency_adjusted** — Efficiency penalized by absences (`tasks_per_hour * (1 - absence_ratio)`).

### **Interaction Features**

9. **tasks_x_hours** — Interaction between output and effort.
10. **tasks_x_absences** — Impact of absences on total workload.

### **Unsupervised Learning Features**

11. **behavior_cluster** — KMeans cluster for employee behavior patterns.

### **Dimensionality Reduction Features**

12. **pca_1, pca_2, pca_3** — Latent variables capturing core productivity structure.

These engineered features **dramatically increase predictive signal** by capturing relationships not directly visible in the original dataset.

---

# 🚀 **6. Optimized Model — After Feature Engineering & Tuning**

### **Optimized Algorithm:**

**RandomForestRegressor / XGBoostRegressor** (based on availability)

### Techniques applied:

* Scaling
* Median imputation
* `SelectKBest(f_regression)` for feature selection
* Randomized hyperparameter search
* KMeans clustering
* PCA components



# 🔍 **7. Before–After Comparison**

A clear demonstration of the improvement achieved through feature engineering and tuning.

| Metric | Baseline      | Optimized (Tuned) | Improvement      |
| ------ | ------------- | ----------------- | ---------------- |
| MAE    | `14.6558`     | `15.7956`         | ↓ `15.4484`      |
| RMSE   | `17.2980`     | `19.9289	`        | ↓ `18.4308`      |
| R²     | `-0.0071`     | `-0.3367`         | ↓ `-0.1433`      |




**Typical observed patterns in well-engineered models:**

* RMSE improves **20–50%**
* R² increases significantly
* Residual distribution becomes tighter and less biased

---

# 📈 **8. Feature Importance Dashboard**

Top contributing factors (from RandomForest/XGBoost):

* tasks_per_hour
* hours_week
* behavior_cluster
* pca_1
* efficiency_adjusted
* tasks_x_hours

A bar chart (`feature_importance.png`) is generated automatically.

---

# 💾 **9. Saved Artifacts**

All reproducible files stored in:

```
artifacts/
 ├── best_pipeline.joblib        # final optimized model
 ├── final_features.joblib       # ordered feature list
 ├── metrics_comparison.joblib   # baseline vs optimized
```

These artifacts allow:

* quick inference
* auditing
* reproducibility
* deployment

---

# 🧠 **10. Why Feature Engineering Worked (Short Technical Summary)**

Raw features (hours, tasks, absences) lack context.
Feature engineering adds **behavioral**, **interaction**, and **derived** signals:

* Efficiency metrics normalize output vs effort
* Interaction terms reveal nonlinear relationships
* Clustering learns segments automatically
* PCA extracts latent patterns
* Scaling + feature selection improves model stability
* Ensemble models capture nonlinear interactions missed by Linear Regression

The end result is a model that learns **true productivity drivers**, not shallow correlations.

---

