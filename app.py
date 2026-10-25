import os
import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="centered",
)


@st.cache_resource
def load_pipeline():
    """Load the pre-trained preprocessing and Logistic Regression pipeline once."""
    possible_paths = [
        "models/model_pipeline.joblib",
        "../models/model_pipeline.joblib",
        "../content/model_pipeline.joblib",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(
        "Saved model pipeline not found. Please run the notebook first to generate 'models/model_pipeline.joblib'."
    )


def main():
    st.title("🎓 Student Academic Performance Prediction")
    st.markdown(
        """
        **Verveox Technologies — AI/ML Assessment Demonstration**  
        This tool estimates whether a student is likely to **Pass** or **Fail** based on study and academic indicators so educators can offer timely academic support.
        """
    )
    st.info(
        "⚠️ **Disclaimer:** This is a demonstration support tool. Predictions and probabilities are statistical model estimates—not guarantees—and must never be used as the sole basis for real educational decisions."
    )

    try:
        pipeline = load_pipeline()
    except FileNotFoundError as err:
        st.error(str(err))
        return

    st.subheader("Enter Student Academic & Study Indicators")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            study_hours = st.number_input(
                "Weekly Study Hours (0 – 60 hrs/week)",
                min_value=0.0,
                max_value=60.0,
                value=20.0,
                step=0.5,
            )
            attendance = st.number_input(
                "Attendance Percentage (0% – 100%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
            )
            previous_score = st.number_input(
                "Previous Examination Score (0 – 100)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
            )

        with col2:
            assignments_completed = st.number_input(
                "Assignments / Extra Sessions Completed (0 – 20)",
                min_value=0,
                max_value=20,
                value=2,
                step=1,
            )
            sleep_hours = st.number_input(
                "Average Sleep Hours per Day (1 – 16 hrs/day)",
                min_value=1.0,
                max_value=16.0,
                value=7.0,
                step=0.5,
            )

        submitted = st.form_submit_button("Predict Academic Outcome")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "study_hours": float(study_hours),
                    "attendance": float(attendance),
                    "previous_score": float(previous_score),
                    "assignments_completed": int(assignments_completed),
                    "sleep_hours": float(sleep_hours),
                }
            ]
        )

        pred_class = pipeline.predict(input_df)[0]
        pred_probs = pipeline.predict_proba(input_df)[0]
        classes = list(pipeline.classes_)

        pass_prob = pred_probs[classes.index("Pass")] * 100
        fail_prob = pred_probs[classes.index("Fail")] * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        if pred_class == "Pass":
            st.success(f"### Predicted Outcome: **PASS** ✅")
        else:
            st.error(
                f"### Predicted Outcome: **FAIL (Needs Academic Support)** 🚨"
            )

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Estimated Pass Probability", f"{pass_prob:.2f}%")
        m_col2.metric("Estimated Fail (At-Risk) Probability", f"{fail_prob:.2f}%")

        st.caption(
            "Note: Probabilities represent statistical estimates from the Logistic Regression model and are not deterministic guarantees."
        )


if __name__ == "__main__":
    main()