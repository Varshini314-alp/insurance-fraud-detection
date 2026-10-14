
from fastapi import FastAPI, Request, Form



from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Create FastAPI application
app = FastAPI(title="Insurance Claim Fraud Detection")

# Use the project folder as the base directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "insurance.joblib"

# Load the trained model
saved_data = joblib.load(MODEL_PATH)

# Support a dictionary or a model saved directly
if isinstance(saved_data, dict):
    model = saved_data.get("model")
    features = saved_data.get("features", saved_data.get("feature"))
else:
    model = saved_data
    features = None

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None, "error": None}
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    claim_amount: float = Form(...),
    age: int = Form(...),
    months_as_customer: int = Form(...)
):
    try:
        if model is None or features is None:
            raise ValueError(
                "The saved file must contain the model and feature names. "
                "Update train_model.py and retrain the model."
            )

        if claim_amount < 0 or age < 0 or months_as_customer < 0:
            raise ValueError("Please enter non-negative values.")

        # Create one input row with the exact training columns
        input_data = pd.DataFrame(
            [{column: None for column in features}]
        )

        # Map form values to columns in your training dataset
        possible_names = {
            "claim_amount": [
                "claim_amount", "total_claim_amount"
            ],
            "age": ["age"],
            "months_as_customer": ["months_as_customer"]
        }

        supplied_values = {
            "claim_amount": claim_amount,
            "age": age,
            "months_as_customer": months_as_customer
        }

        for field, value in supplied_values.items():
            matching_columns = [
                name for name in possible_names[field]
                if name in input_data.columns
            ]

            if matching_columns:
                input_data.loc[0, matching_columns[0]] = value

        # Predict with the saved pipeline
        prediction = int(model.predict(input_data)[0])

        if prediction == 1:
            result = "Potential Fraud Detected"
        else:
            result = "Claim Classified as Non-Fraud"

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"result": result, "error": None}
        )

    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": None,
                "error": str(exc)
            },
            status_code=400
        )


@app.get("/health")
def health():
    return {"status": "running"}
