import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("insurance_claims.csv")

# 2. Remove unnecessary columns
df = df.drop(columns=["_c39"], errors="ignore")
df = df.drop(columns=["policy_number", "incident_location"], errors="ignore")

# 3. Define target column
target = "fraud_reported"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=[target])

# 4. Separate features and target
X = df.drop(columns=[target])
y = df[target].astype(str).str.strip().str.upper()

# 5. Convert target labels to 0 and 1
y = y.map({"N": 0, "Y": 1})

if y.isna().any():
    raise ValueError("Target column must contain Y/N fraud labels.")

# 6. Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 7. Preprocess data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 8. Create model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# 9. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 10. Train model
model.fit(X_train, y_train)

# 11. Evaluate model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# 12. Save trained model and feature names
joblib.dump({
    "model": model,
    "features": X.columns.tolist()
}, "insurance.joblib")

print("Model saved successfully as insurance.joblib")