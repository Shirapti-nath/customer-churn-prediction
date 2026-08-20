# Customer Churn Prediction

This project preprocesses the Telco Customer Churn dataset and trains a churn
classifier with configurable model selection and MLflow tracking.

## Setup

Install the dependencies in a Python 3.12 environment:

```bash
pip install -r requirements.txt
```

## Run preprocessing

```bash
python source/preprocess.py
```

This prints the encoded feature shape, churn distribution, and null count.

## Train a model

```bash
python source/train.py
```

The model is selected in `config/config.yaml`. Supported values are
`logistic_regression`, `random_forest`, and `xgboost`. Training prints an
evaluation report and logs the run and model to the local MLflow store.
