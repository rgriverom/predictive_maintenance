import json
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from data_loading import load_train
from features import get_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DIR = PROJECT_ROOT / 'mlflow_data'
EXPORT_DIR = PROJECT_ROOT / 'model_export'
EXPERIMENT_NAME = 'cmapss-rul-prediction'

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")


def main():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f'No experiment named "{EXPERIMENT_NAME}" — run train_final.py first')

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['metrics.val_rmse ASC'],
        max_results=1,
    )
    if not runs:
        raise RuntimeError('Experiment has no runs — run train_final.py first')

    best_run = runs[0]
    model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)

    # Self-contained: no dependency on the MLflow tracking store or any
    # absolute host path — this is what actually ships to the container.
    mlflow.sklearn.save_model(model, path=str(EXPORT_DIR / 'model'))

    train = load_train()
    feature_cols = get_feature_columns(train)
    with open(EXPORT_DIR / 'feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)

    print(f'Exported model from run {best_run.info.run_id} '
          f'(val_rmse={best_run.data.metrics.get("val_rmse"):.2f}) to {EXPORT_DIR}')


if __name__ == '__main__':
    main()