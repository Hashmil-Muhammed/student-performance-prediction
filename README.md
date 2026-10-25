# Student Academic Performance Prediction System

This repository contains an end-to-end machine learning solution that predicts whether a student is likely to pass or fail using academic and study-related features. It was developed as part of the **Verveox Technologies AI/ML Trainee Hiring Assessment**.

## 1. Project Overview
The primary objective of this project is to explore whether study-related and academic indicators can help identify students who may need additional academic support. The system uses a binary classification approach (Logistic Regression and K-Nearest Neighbors) to predict a `final_result` of **Pass** or **Fail** based on 5 key features: `study_hours`, `attendance`, `previous_score`, `assignments_completed`, and `sleep_hours`.


---

## 2. Project Folder Structure
The repository is structured as follows to clearly separate data, code, models, and documentation:

```text
student-performance-prediction/
│
├── data/
│   ├── StudentPerformanceFactors.csv       # Raw dataset (downloaded/provided)
│   └── cleaned_StudentPerformanceFactors.csv     # Cleaned dataset ready for modeling
│
├── notebooks/
│   └── analysis.ipynb                      # Complete EDA, preprocessing & model training notebook
│
├── models/
│   └── model_pipeline.joblib               # Serialized preprocessing & classification pipeline
│
├── app.py                                  # Interactive Streamlit prediction web application
├── requirements.txt                        # Python package dependencies
├── README.md                               # Project documentation, setup, and execution steps
└── report.pdf                              # Formal project evaluation report covering metrics & limitations
```

---

## 3. Prerequisites
To run this project, you need the following installed on your system:
- **Python 3.11+**
- **pip** (Python package installer)

The core libraries used include:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `streamlit`

---

## 4. Installation & Setup Instructions
Follow these exact steps to set up the environment and run the project:

### Step 1: Clone or Download the Repository
Navigate to your desired directory and open your terminal:
```bash
cd student-performance-prediction
```

### Step 2: Install Required Dependencies
Install all necessary Python packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## 5. Execution Steps

### A. Run the Data Analysis and Model Training
To view the Exploratory Data Analysis (EDA), data cleaning, model training, and evaluation workflows, launch the Jupyter Notebook:
```bash
jupyter notebook notebooks/analysis.ipynb
```

**Instructions:**
Run all cells sequentially from top to bottom. This will regenerate the cleaned dataset in the `data/` folder, display the evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrices), and save the trained model pipeline into the `models/` folder.

### B. Run the Streamlit Prediction Interface
To launch the interactive web application and test the prediction interface:
```bash
streamlit run app.py
```

**Instructions:**
This command will open an interactive web app in your default browser (usually at http://localhost:8501). You can input different student profiles using the sliders and numerical inputs to generate real-time Pass/Fail predictions and probability estimates.
