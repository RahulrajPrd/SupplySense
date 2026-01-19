import pandas as pd
from prophet import Prophet
import mlflow
import mlflow.prophet

DATA_PATH = "data/retail_sales.csv"


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)

    # Parse date
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Aggregate daily total sales per store
    daily = (
        df.groupby(["store_name", "transaction_date"])["final_amount"]
        .sum()
        .reset_index()
    )

    return daily


def train_for_store(store_name, data):
    store_data = data[data["store_name"] == store_name]

    # Prophet format
    prophet_df = store_data.rename(
        columns={
            "transaction_date": "ds",
            "final_amount": "y"
        }
    )

    # Drop NaNs
    prophet_df = prophet_df.dropna()

    # Skip if not enough data
    if len(prophet_df) < 2:
        print(f"Skipping {store_name}: insufficient data")
        return

    # Train model
    model = Prophet()
    model.fit(prophet_df)

    # Training metric (in-sample MSE)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    mse = ((forecast["yhat"] - prophet_df["y"]) ** 2).mean()

    # MLflow logging
    mlflow.set_experiment("supplysense-demand-forecast")

    with mlflow.start_run(run_name=store_name):
        mlflow.log_param("store_name", store_name)
        mlflow.log_metric("mse", mse)
        mlflow.prophet.log_model(model, artifact_path="model")

    print(f"Training complete for store: {store_name}")


if __name__ == "__main__":
    data = load_and_prepare()

    stores = data["store_name"].unique()
    print(f"Total stores found: {len(stores)}")

    for store in stores:
        train_for_store(store, data)
