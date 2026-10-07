"""Train and evaluate a simple loan approval classifier."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Keep the feature list in one place so training and the app use the same order.
FEATURE_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
TARGET_COLUMN = "Loan_Status"

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset" / "loan_data.csv"
MODEL_PATH = PROJECT_DIR / "model" / "loan_model.pkl"


def load_and_clean_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the CSV, normalize simple formatting issues, and separate the label."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing_columns)
        )

    # Blank strings become missing values. Numeric fields with unexpected text
    # are also treated as missing, then imputed inside the training pipeline.
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].map(
            lambda value: value.strip() if isinstance(value, str) and value.strip() else np.nan
        )
    data[TARGET_COLUMN] = data[TARGET_COLUMN].map(
        lambda value: value.strip() if isinstance(value, str) else np.nan
    )
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # The source data uses Y/N. Drop rows whose outcome is absent or unknown.
    target = data[TARGET_COLUMN].map({"N": 0, "Y": 1})
    usable_rows = target.notna()
    features = data.loc[usable_rows, FEATURE_COLUMNS].copy()
    target = target.loc[usable_rows].astype(int)

    if target.nunique() != 2:
        raise ValueError("Loan_Status must contain both approved (Y) and rejected (N) rows.")

    return features, target


def build_model() -> Pipeline:
    """Create preprocessing and logistic regression as one reusable pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def main() -> None:
    """Train on 80% of the records, report test metrics, and save the model."""
    X, y = load_and_clean_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        # Rows are actual labels, columns are predicted labels: rejected, approved.
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
    }

    print("Loan approval model evaluation (test split)")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f} (approved applications)")
    print(f"Recall:    {metrics['recall']:.3f} (approved applications)")
    print(f"F1-score:  {metrics['f1_score']:.3f} (approved applications)")
    print("Confusion matrix (rows=actual, columns=predicted; rejected, approved):")
    print(metrics["confusion_matrix"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Save preprocessing and classifier together so app inputs are transformed
    # exactly the same way as training inputs.
    joblib.dump(
        {
            "model": model,
            "metrics": metrics,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    print(f"\nSaved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
