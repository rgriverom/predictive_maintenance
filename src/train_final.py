from pathlib import Path
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import prepare_training_data, RUL_CAP

BEST_PARAMS = {
    'n_estimators': 273,
    'max_depth': 13,
    'min_samples_split': 20,
    'min_samples_leaf': 3,
    'max_features': 'log2',
    'random_state': 42,
    'n_jobs': -1,
}


def main():
    project_root = Path(__file__).resolve().parent.parent
    mlflow_dir = project_root / 'mlflow_data'
    mlflow_dir.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir / 'mlflow.db'}")

    experiment_name = 'cmapss-rul-prediction'
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(mlflow_dir / 'artifacts').as_uri()
        )
    mlflow.set_experiment(experiment_name)

    X_train, X_val, y_train, y_val, feature_cols = prepare_training_data()

    with mlflow.start_run(run_name='rf_optuna_tuned'):
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param('rul_cap', RUL_CAP)
        mlflow.log_param('n_features', len(feature_cols))

        model = RandomForestRegressor(**BEST_PARAMS)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))

        mlflow.log_metric('val_mae', mae)
        mlflow.log_metric('val_rmse', rmse)

        mlflow.sklearn.log_model(model, name='model')

        print(f'Logged run — MAE: {mae:.2f}, RMSE: {rmse:.2f}')
        print(f'MLflow run ID: {mlflow.active_run().info.run_id}')


if __name__ == '__main__':
    main()