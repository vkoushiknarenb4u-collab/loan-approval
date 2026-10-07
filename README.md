# Loan Approval Prediction

A beginner-friendly machine learning project that estimates whether a historical loan application would be approved or rejected. It uses a CSV file, scikit-learn preprocessing, logistic regression, and a dark Streamlit dashboard with applicant form, prediction reasons, model metrics, and dataset/model information pages.

> **Learning demo only:** this model reflects patterns in a small historical dataset. It is not a lending tool and should not be used to make real financial decisions. The source data includes sensitive applicant fields such as gender and marital status.

## Problem statement

Loan approval is a yes/no classification task. Given applicant information, the model estimates the probability of an approval (`Y`) and predicts approval when that probability is at least 50%.

## Dataset

`dataset/loan_data.csv` contains 614 historical applications and their recorded outcomes. The included CSV is from a [public GitHub mirror](https://github.com/SrivastavaKhushboo/Loan-Eligibilty-Prediction/blob/main/dataset.csv) of the commonly used Loan Prediction dataset. The rows include blank fields; the training pipeline fills missing numeric values with the training median and missing categorical values with the training mode. `Loan_ID`, when present, is not used as a feature.

The original dataset describes `LoanAmount` in thousands and `Loan_Amount_Term` in months. Check the source and local CSV before using different units.

## Features

The model uses these 11 input fields:

- Gender
- Married
- Dependents
- Education
- Self employed
- Applicant income
- Co-applicant income
- Loan amount
- Loan amount term
- Credit history
- Property area

`Loan_Status` is the target (`Y` = approved, `N` = rejected).

## Machine learning algorithm

The training script splits records into 80% training and 20% test sets, keeping the class proportions similar in both (`stratify`). It then fits one scikit-learn pipeline:

1. Fill missing numeric values with the median and scale numeric columns.
2. Fill missing categorical values with the most common category and one-hot encode them.
3. Fit logistic regression to predict the approval label.

Putting preprocessing and the classifier in the saved pipeline ensures that app inputs receive the same transformations as training data. The local explanation in the app uses the fitted logistic regression coefficients to show the strongest applicant details supporting the predicted outcome and the strongest details weighing toward the other outcome. It displays separate reasons for approvals and rejections, including the entered values. These are model contributions, not causal explanations.

## Installation

Use Python 3.10 or newer. From this project folder, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install -r requirements.txt
```

## Train the model

Run:

```bash
python train_model.py
```

The script prints accuracy, precision, recall, F1-score, and the confusion matrix for the held-out test set. It saves the fitted preprocessing and model pipeline, along with the metrics, to `model/loan_model.pkl` using joblib.

## Run the application

Train the model once, then start Streamlit:

```bash
streamlit run app.py
```

Enter the applicant details and choose **Predict Loan Approval**. The app displays the predicted result, estimated approval probability, model contributions, test metrics, and confusion matrix. If you change the dataset or training code, rerun `python train_model.py` to refresh the saved model.

## Sample prediction

Try this example in the form:

| Field | Example |
| --- | --- |
| Gender | Male |
| Married | Yes |
| Dependents | 0 |
| Education | Graduate |
| Self employed | No |
| Applicant income | 5,000 |
| Co-applicant income | 0 |
| Loan amount | 120 |
| Loan term | 360 months |
| Credit history meets guidelines | Yes |
| Property area | Urban |

With the included model, this example is predicted as **Loan Approved**, with an estimated approval probability of about **81.5%**. The result is calculated from the saved model when you run the app and can change if you retrain the model or edit the CSV.

## Model evaluation

The script reports these metrics on the test split:

- **Accuracy:** share of all test applications classified correctly.
- **Precision:** among predicted approvals, share that were approved in the recorded data.
- **Recall:** among recorded approvals, share the model found.
- **F1-score:** a combined summary of precision and recall.
- **Confusion matrix:** counts actual rejections and approvals against predicted rejections and approvals.

Metric values are generated when you train the model and appear in the terminal and app. Performance on this small historical dataset is not a guarantee of performance on new applications.

With the included CSV and the fixed 80/20 split (`random_state=42`), the current run reports:

| Metric | Result |
| --- | ---: |
| Accuracy | 86.2% |
| Precision | 84.0% |
| Recall | 98.8% |
| F1-score | 90.8% |

Confusion matrix, with rows as actual outcomes and columns as predictions (`rejected`, `approved`): `[[22, 16], [1, 84]]`.

## Project structure

```text
loan-approval-prediction/
├── dataset/
│   └── loan_data.csv
├── model/
│   └── loan_model.pkl          # generated by train_model.py
├── .streamlit/
│   └── config.toml             # dark dashboard theme
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Future improvements

- Compare logistic regression with other simple classifiers.
- Add cross-validation and probability calibration.
- Review data quality, privacy, and fairness before any real-world use.
- Add input validation and a way to download a prediction summary.
- Track model versions and retraining data.
