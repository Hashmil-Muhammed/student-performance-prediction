# Project Report: Student Academic Performance Prediction
**Company:** Verveox Technologies (Hiring Assessment)

**Full name:** Hashmil Muhammed

**Email:** hashmilmuhammedparammal@gmail.com

**Submission date:** 25/10/2026

**GitHub / repository link:** https://github.com/Hashmil-Muhammed/student-performance-prediction


---

## 1. Executive Summary & Objective
The goal of this assignment is to develop a machine learning classifier that predicts student academic outcomes (`Pass` vs. `Fail`) using study habits and prior performance. Using the *Student Performance Factors* dataset ($N = 6,607$), we cleaned anomalies, built exploratory visualizations, trained **Logistic Regression** and **K-Nearest Neighbors (KNN)** models inside rigorous Scikit-learn pipelines, evaluated them on a held-out test set, and deployed an interactive Streamlit application. **Logistic Regression** was selected as the final production model due to superior performance across all metrics (`Accuracy: 88.94%`, `Pass F1: 0.9302`, `Fail F1: 0.7336`).

---

## 2. Dataset Source & Preprocessing
- **Source:** Kaggle *Student Performance Factors* dataset.
- **Mapping:** Mapped `Hours_Studied` $\rightarrow$ `study_hours`, `Attendance` $\rightarrow$ `attendance`, `Previous_Scores` $\rightarrow$ `previous_score`, `Tutoring_Sessions` $\rightarrow$ `assignments_completed`, and `Sleep_Hours` $\rightarrow$ `sleep_hours`.
- **Data Cleaning:** Removed 1 invalid record where `Exam_Score > 100` and dropped 6 exact duplicate rows to prevent train-test contamination (`6,607` $\rightarrow$ `6,600` records).
- **Target Derivation:** Binarized `Exam_Score` at a competency threshold of `65` (`Exam_Score >= 65` $\rightarrow$ `Pass` [78.05%], `< 65` $\rightarrow$ `Fail` [21.95%]), and immediately dropped `Exam_Score` to prevent target leakage.

---

## 3. Exploratory Data Analysis Insights
- **Attendance & Study Hours:** Strong positive association with passing. `Pass` students average **83.00% attendance** and **21.12 study hours/week**, compared to **69.23% attendance** and **15.91 study hours/week** for `Fail` students.
- **Class Imbalance:** Observed a 3.55:1 class ratio (`Pass` vs. `Fail`), making stratified splitting and F1-score evaluation mandatory.

---

## 4. Model Training & Comparative Evaluation
Models were trained using an 80/20 stratified split (`X_train = 5,280`, `X_test = 1,320`) with 5-fold cross-validation.Preprocessing (`SimpleImputer` + `StandardScaler`) was fitted exclusively on training folds to prevent data leakage.
| Model | Test Accuracy | Precision (`Pass`) | Recall (`Pass`) | F1 (`Pass`) | Precision (`Fail`) | Recall (`Fail`) | F1 (`Fail`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8894** | **0.9162** | **0.9447** | **0.9302** | **0.7791** | **0.6931** | **0.7336** |
| **KNN ($k=5$)** | 0.8636 | 0.8965 | 0.9330 | 0.9144 | 0.7218 | 0.6172 | 0.6654 |

---

## 5. Model Selection & Limitations
- **Selection:** Logistic Regression is selected because it correctly identifies **69.31% of at-risk students** (`Fail` recall) compared to **61.72%** for KNN, while offering interpretable feature coefficients (`attendance` $\rightarrow$ `+2.77`, `study_hours` $\rightarrow$ `+2.11`).
- **Limitations:** Hard threshold binarization creates boundary ambiguity around score 65; model relies on static observational data without external multi-school validation.
- **Future Improvements:** Implement non-linear tree-based ensembles (Random Forest / XGBoost) and incorporate qualitative behavioral indicators.
