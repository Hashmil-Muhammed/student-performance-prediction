# Student Academic Performance Prediction System
**Verveox Technologies — AI/ML Trainee Practical Assessment**

## 1. Project Overview
This project builds an end-to-end binary classification machine learning solution to predict whether a student is likely to **Pass** or **Fail** based on academic and study-related features[cite: 1, 2]. The primary objective is to provide an early-warning indicator system to help educators identify students who may need additional academic support[cite: 2].

---

## 2. Project Folder Structure
```text
student-performance-prediction/
│
├── data/
│   ├── StudentPerformanceFactors.csv       # Raw Kaggle dataset
│   └── student_performance_cleaned.csv     # Cleaned dataset (6,600 rows)
│
├── notebooks/
│   └── analysis.ipynb                      # Complete EDA & model training Jupyter notebook
│
├── models/
│   └── model_pipeline.joblib               # Serialized preprocessing & Logistic Regression pipeline
│
├── app.py                                  # Interactive Streamlit prediction web application
├── requirements.txt                        # Python package dependencies
├── README.md                               # Project documentation
└── report.pdf (or report.md)               # Formal project evaluation report
3. Installation & Setup Instructions
Clone or Download the Repository:

Bash
cd student-performance-prediction
Install Required Dependencies:

Bash
pip install -r requirements.txt
Run the Jupyter Notebook (Data Exploration, EDA & Model Training):

Bash
jupyter notebook notebooks/analysis.ipynb
(Run all cells sequentially to regenerate cleaned data, plots, models, and evaluation tables).

Run the Streamlit Prediction Interface:

Bash
streamlit run app.py
(This opens an interactive web app in your browser at http://localhost:8501 to test student profiles).
