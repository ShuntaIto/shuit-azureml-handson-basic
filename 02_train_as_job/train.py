import argparse
from typing import TypedDict

import lightgbm as lgb
import mlflow
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


class GetArgsOutput(TypedDict):
    input_train_data: str
    input_test_data: str
    boosting_type: str
    metric: str
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    num_iteration: int
    mode: str


class LoadDatasetOutput(TypedDict):
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame


def get_args() -> GetArgsOutput:
    # 引数取得
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train_data", type=str, default="./data/nyc_taxi_train_dataset.csv")
    parser.add_argument("--input_test_data", type=str, default="./data/nyc_taxi_test_dataset.csv")
    parser.add_argument("--boosting_type", type=str, default="gbdt")
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--num_leaves", type=float, default=10)
    parser.add_argument("--min_data_in_leaf", type=float, default=2)
    parser.add_argument("--num_iteration", type=int, default=100)
    parser.add_argument("--mode", type=str, default="local")


    args = parser.parse_args()

    params: GetArgsOutput = {
        "input_train_data": args.input_train_data,
        "input_test_data": args.input_test_data,
        "boosting_type": args.boosting_type,
        "metric": args.metric,
        "learning_rate": args.learning_rate,
        "num_leaves": int(args.num_leaves),
        "min_data_in_leaf": int(args.min_data_in_leaf),
        "num_iteration": args.num_iteration,
        "mode": args.mode,
    }

    return params


def load_dataset(train_path: str, test_path: str) -> LoadDatasetOutput:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    x_train = train[train.columns[train.columns != "totalAmount"]]
    y_train = train["totalAmount"]

    x_test = test[test.columns[test.columns != "totalAmount"]]
    y_test = test["totalAmount"]

    output: LoadDatasetOutput = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

    return output


def train_lgb_model(args: GetArgsOutput, datasets: LoadDatasetOutput) -> lgb.Booster:
    # mlflow autolog 開始
    # ジョブ実行の場合 Azure ML が初期設定する環境変数をもとに mlflow が自動でセッティングされる
    mlflow.lightgbm.autolog(registered_model_name="nyc_taxi_regressor_lightgbm")

    train_dataset = lgb.Dataset(datasets["x_train"], datasets["y_train"])
    test_dataset = lgb.Dataset(datasets["x_test"], datasets["y_test"], reference=train_dataset)

    # パラメーター記録
    params = {
        "boosting_type": args["boosting_type"],
        "metric": args["metric"],
        "learning_rate": args["learning_rate"],
        "num_leaves": args["num_leaves"],
        "min_data_in_leaf": args["min_data_in_leaf"],
        "num_iteration": args["num_iteration"],
        "task": "train",
        "objective": "regression",
    }

    # 学習
    gbm = lgb.train(params, train_dataset, num_boost_round=50, valid_sets=test_dataset, early_stopping_rounds=10)

    return gbm

if __name__ == "__main__":

    args = get_args()

    if args["mode"] == "local":
        print("mode local")
        ml_client = MLClient.from_config(credential=DefaultAzureCredential(exclude_shared_token_cache_credential=True),
                     logging_enable=True)
        azureml_mlflow_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(azureml_mlflow_uri)

        experiment_name = 'nyc_taxi_regression_local_script'
        mlflow.set_experiment(experiment_name)
    
    datasets = load_dataset(train_path=args["input_train_data"], test_path=args["input_test_data"])

    run = mlflow.start_run()
    train_lgb_model(args=args, datasets=datasets)
    mlflow.end_run()
