from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

# Load the saved, fitted Random Forest pipeline
MODEL_PATH = Path(__file__).resolve().parent / "tuned_random_forest.joblib"
model = joblib.load(MODEL_PATH)

FEATURES = list(model.feature_names_in_)

def predict_default(
    limit_bal, sex, education, marriage, age,
    pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
    bill_amt1, bill_amt2, bill_amt3,
    bill_amt4, bill_amt5, bill_amt6,
    pay_amt1, pay_amt2, pay_amt3,
    pay_amt4, pay_amt5, pay_amt6
):
    values = {
        "LIMIT_BAL": limit_bal,
        "SEX": int(sex),
        "EDUCATION": int(education),
        "MARRIAGE": int(marriage),
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,
        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5,
        "BILL_AMT6": bill_amt6,
        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4,
        "PAY_AMT5": pay_amt5,
        "PAY_AMT6": pay_amt6,
    }

    # Use the same feature names and order as training
    input_data = pd.DataFrame([values], columns=FEATURES)

    # Class 1 represents default in the following month
    default_probability = float(
        model.predict_proba(input_data)[0, 1]
    )

    prediction = int(model.predict(input_data)[0])

    label = (
        "Predicted Default (Class 1)"
        if prediction == 1
        else "Predicted Non-Default (Class 0)"
    )

    return label, f"{default_probability:.2%}"


with gr.Blocks(title="Credit Card Default Prediction") as demo:
    gr.Markdown(
        """
        # Credit Card Default Prediction

        This educational application uses a tuned Random Forest
        model trained on the UCI Default of Credit Card Clients
        dataset to estimate next-month payment default risk.

        **Disclaimer:** This is an academic demonstration.
        Predictions should not be used for real lending,
        credit approval, or other financial decisions.
        Do not enter real clients' personal information.
        """
    )

    with gr.Row():
        limit_bal = gr.Number(label="Credit Limit", value=200000)
        age = gr.Number(label="Age", value=35)

    with gr.Row():
        sex = gr.Dropdown(
            choices=[("Male (1)", 1), ("Female (2)", 2)],
            value=1,
            label="Sex"
        )
        education = gr.Dropdown(
            choices=[
                ("Other / Unknown (0)", 0),
                ("Graduate School (1)", 1),
                ("University (2)", 2),
                ("High School (3)", 3),
                ("Other (4)", 4),
            ],
            value=2,
            label="Education"
        )
        marriage = gr.Dropdown(
            choices=[
                ("Other / Unknown (0)", 0),
                ("Married (1)", 1),
                ("Single (2)", 2),
                ("Other (3)", 3),
            ],
            value=2,
            label="Marital Status"
        )

    gr.Markdown("### Repayment Status")
    gr.Markdown(
        "Enter the dataset's repayment status codes. "
        "-1 indicates payment duly, 0 indicates use of revolving "
        "credit, and positive values indicate months of delay. "
        "Other dataset codes may also occur."
    )

    repayment_inputs = [
        gr.Number(label=name, value=0, precision=0)
        for name in [
            "PAY_0", "PAY_2", "PAY_3",
            "PAY_4", "PAY_5", "PAY_6"
        ]
    ]

    gr.Markdown("### Bill Statement Amounts")
    bill_inputs = [
        gr.Number(label=f"BILL_AMT{i}", value=0)
        for i in range(1, 7)
    ]

    gr.Markdown("### Previous Payment Amounts")
    payment_inputs = [
        gr.Number(label=f"PAY_AMT{i}", value=0)
        for i in range(1, 7)
    ]

    predict_button = gr.Button(
        "Predict Default Risk",
        variant="primary"
    )

    prediction_output = gr.Textbox(
        label="Predicted Classification"
    )
    probability_output = gr.Textbox(
        label="Estimated Probability of Default"
    )

    predict_button.click(
        fn=predict_default,
        inputs=[
            limit_bal, sex, education, marriage, age,
            *repayment_inputs,
            *bill_inputs,
            *payment_inputs,
        ],
        outputs=[prediction_output, probability_output]
    )

    gr.Markdown(
        """
        **Model limitation:** The tuned Random Forest achieved
        approximately 36.17% recall for the default class on
        the held-out test set. It missed many actual defaults.
        Its probabilities are model estimates, not guaranteed
        or independently calibrated risk probabilities.
        """
    )

if __name__ == "__main__":
    demo.launch()