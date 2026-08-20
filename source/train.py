import yaml
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import load_and_preprocess


def get_model(name):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        )
    elif name == "xgboost":
        return XGBClassifier(eval_metric="logloss", random_state=42)
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Load and preprocess data
    X, y = load_and_preprocess(config)

    # Train/test split (stratified since churn is imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )

    # Track this run with MLflow
    mlflow.set_experiment("churn-prediction")
    with mlflow.start_run():
        mlflow.log_params(config)

        # Train
        model = get_model(config["model"])
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall_churn", report["1"]["recall"])
        mlflow.log_metric("precision_churn", report["1"]["precision"])
        mlflow.log_metric("f1_churn", report["1"]["f1-score"])

        # Log the trained model itself (XGBoost needs its own logger)
        if config["model"] == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        # Print results to console
        print(f"\nModel: {config['model']}")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()