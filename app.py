from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).parent / "data" / "CDC Diabetes Dataset.csv"
FEATURES = ["BMI", "Age", "HighBP", "HighChol", "GenHlth"]
FACTOR_LABELS = {
    "BMI": "BMI",
    "Age": "Age group",
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "GenHlth": "General health rating",
}


def age_to_cdc_category(age: int) -> int:
    if age < 25:
        return 1
    if age < 30:
        return 2
    if age < 35:
        return 3
    if age < 40:
        return 4
    if age < 45:
        return 5
    if age < 50:
        return 6
    if age < 55:
        return 7
    if age < 60:
        return 8
    if age < 65:
        return 9
    if age < 70:
        return 10
    if age < 75:
        return 11
    if age < 80:
        return 12
    return 13


@st.cache_data(show_spinner=False)
def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(DATA_PATH, usecols=["Diabetes_012", *FEATURES])
    x = data[FEATURES]
    y = (data["Diabetes_012"] > 0).astype(int)
    return x, y


@st.cache_resource(show_spinner=False)
def train_model() -> Pipeline:
    x, y = load_training_data()
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def get_risk_level(probability: float) -> tuple[str, str]:
    if probability < 0.25:
        return "Low", "low"
    if probability < 0.5:
        return "Moderate", "moderate"
    return "High", "high"


def get_top_risk_factors(inputs: dict[str, float]) -> list[tuple[str, str]]:
    factors = []
    if inputs["GenHlth"] >= 4:
        factors.append(("General health rating", "Fair or poor self-rated health is strongly associated with higher diabetes risk."))
    elif inputs["GenHlth"] == 3:
        factors.append(("General health rating", "A good, but not very good or excellent, health rating can still add risk."))

    if inputs["BMI"] >= 30:
        factors.append(("BMI", "BMI is in the obese range, which is one of the strongest modifiable risk signals."))
    elif inputs["BMI"] >= 25:
        factors.append(("BMI", "BMI is in the overweight range and may contribute to elevated risk."))

    if inputs["Age"] >= 9:
        factors.append(("Age group", "The CDC age category is 60-64 or older, where risk tends to rise."))
    elif inputs["Age"] >= 7:
        factors.append(("Age group", "The CDC age category is 50-54 or older, which adds some risk."))

    if inputs["HighBP"] == 1:
        factors.append(("High blood pressure", "Reported high blood pressure increases the predicted risk."))

    if inputs["HighChol"] == 1:
        factors.append(("High cholesterol", "Reported high cholesterol increases the predicted risk."))

    return factors[:3] or [("No dominant factor", "The selected inputs do not show a strong single risk driver.")]


def render_factor_bars(inputs: dict[str, float]) -> None:
    reference = {
        "BMI": min(max((inputs["BMI"] - 18.5) / 21.5, 0), 1),
        "Age": (inputs["Age"] - 1) / 12,
        "HighBP": inputs["HighBP"],
        "HighChol": inputs["HighChol"],
        "GenHlth": (inputs["GenHlth"] - 1) / 4,
    }
    chart_data = (
        pd.DataFrame(
            {
                "Factor": [FACTOR_LABELS[key] for key in reference],
                "Relative contribution": list(reference.values()),
            }
        )
        .sort_values("Relative contribution", ascending=False)
        .set_index("Factor")
    )
    st.bar_chart(chart_data)


def main() -> None:
    st.set_page_config(page_title="Diabetes Risk Demo", page_icon="health", layout="centered")

    st.title("Diabetes Risk Prediction Demo")
    st.caption("A simple interactive demo based on the CDC BRFSS diabetes dataset.")

    with st.spinner("Loading CDC data and training the demo model..."):
        model = train_model()

    with st.form("risk_form"):
        st.subheader("Patient inputs")
        col_a, col_b = st.columns(2)
        with col_a:
            bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.0, step=0.5)
            age = st.slider("Age", min_value=18, max_value=90, value=45)
            gen_health = st.select_slider(
                "General health",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda value: {
                    1: "Excellent",
                    2: "Very good",
                    3: "Good",
                    4: "Fair",
                    5: "Poor",
                }[value],
            )
        with col_b:
            high_bp = st.toggle("High blood pressure")
            high_chol = st.toggle("High cholesterol")

        submitted = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

    if submitted:
        model_inputs = {
            "BMI": float(bmi),
            "Age": age_to_cdc_category(age),
            "HighBP": int(high_bp),
            "HighChol": int(high_chol),
            "GenHlth": int(gen_health),
        }
        input_frame = pd.DataFrame([model_inputs], columns=FEATURES)
        probability = float(model.predict_proba(input_frame)[0, 1])
        risk_label, risk_class = get_risk_level(probability)

        st.divider()
        st.subheader("Prediction")
        metric_col, detail_col = st.columns([1, 2])
        with metric_col:
            st.metric("Diabetes risk", f"{probability:.1%}", risk_label)
        with detail_col:
            st.write(
                f"The model estimates a **{risk_class}** diabetes risk from the selected BMI, age, "
                "blood pressure, cholesterol, and general health inputs."
            )
            st.info(
                "This is not medical diagnosis. It is an educational ML demo and should not replace professional medical advice."
            )

        st.subheader("Top risk factors")
        for factor, explanation in get_top_risk_factors(model_inputs):
            st.markdown(f"**{factor}**  \n{explanation}")

        render_factor_bars(model_inputs)
    else:
        st.info("Enter the values above and click Predict risk to see the demo output.")


if __name__ == "__main__":
    main()
