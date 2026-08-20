import pandas as pd

def load_and_preprocess(config):
    # 1. Load data
    df = pd.read_csv(config["data_path"])

    # 2. Fix TotalCharges (loaded as string, has blank values)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 3. Drop non-predictive column
    df = df.drop(columns=["customerID"])

    # 4. Encode target: Yes/No -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 5. Separate features and target
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # 6. One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    return X, y


if __name__ == "__main__":
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    X, y = load_and_preprocess(config)
    print("X shape:", X.shape)
    print("y distribution:\n", y.value_counts(normalize=True))
    print("Any nulls in X?", X.isnull().sum().sum())