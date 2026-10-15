# Insurance Fraud Detection Using Machine Learning

## Project Overview
This project uses Machine Learning to identify potentially fraudulent insurance claims based on insurance claim data. It uses Python and FastAPI to provide predictions through a web application.

## Objectives
- Analyze insurance claim data.
- Clean and preprocess the dataset.
- Train a Machine Learning model.
- Predict potentially fraudulent insurance claims.
- Build a web application for predictions.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- HTML and CSS
- Joblib
- Jupyter Notebook

## Dataset
The project uses an insurance claims dataset stored in CSV format.

**Dataset file:** `insurance_claims.csv`

## Project Features
- Data preprocessing and cleaning
- Machine Learning model training
- Model saving and loading
- Insurance claim prediction
- Web-based user interface
- Prediction results

## Project Structure
```text
insurance_fraud/
├── templates/
│   └── index.html
├── insurance_claims.csv
├── insurance_fraud.ipynb
├── insurance.joblib
├── main.py
├── requirements.txt
├── run_prediction.py
├── train_model.py
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Start the FastAPI application:

```bash
uvicorn main:app --reload
```

Open your browser and visit:

http://127.0.0.1:8000

FastAPI documentation:

http://127.0.0.1:8000/docs

## Machine Learning Workflow
1. Load the dataset.
2. Explore and clean the data.
3. Preprocess the features.
4. Split the dataset into training and testing sets.
5. Train the Machine Learning model.
6. Evaluate the model.
7. Save the trained model using Joblib.
8. Use FastAPI to make predictions.

## Future Enhancements
- Improve model accuracy.
- Compare different Machine Learning algorithms.
- Add more features to the web application.
- Deploy the application online.

## Conclusion
This project demonstrates how Machine Learning can help identify potentially fraudulent insurance claims and support insurance claim assessment.

## Author
"Varshini M"
