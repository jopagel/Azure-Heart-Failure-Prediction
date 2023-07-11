from sklearn.linear_model import LogisticRegression
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset
import joblib



def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength. Smaller values cause stronger regularization",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of iterations to converge",
    )

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    ws = run.experiment.workspace

    key = "heart_failure_data"

    dataset = ws.datasets[key]

    df = dataset.to_pandas_dataframe()

    y = df["DEATH_EVENT"]
    x = df.drop(columns=["DEATH_EVENT"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))

    model_file_name = "trained_model.pkl"
    joblib.dump(value=model, filename=model_file_name)

    run.upload_file(model_file_name, model_file_name)


if __name__ == "__main__":
    main()