from pathlib import Path

import numpy as np
import pytest
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import prepare_training_data, RUL_CAP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / 'model_export'

MAX_ACCEPTABLE_MAE = 15.0
MAX_ACCEPTABLE_RMSE = 20.0


@pytest.fixture(scope='module')
def model_and_validation_data():
    if not EXPORT_DIR.exists():
        pytest.skip(f'{EXPORT_DIR} not found — run export_model.py first')
    model = mlflow.sklearn.load_model(str(EXPORT_DIR / 'model'))
    X_train, X_val, y_train, y_val, feature_cols = prepare_training_data()
    return model, X_val, y_val


def test_model_mae_within_threshold(model_and_validation_data):
    model, X_val, y_val = model_and_validation_data
    mae = mean_absolute_error(y_val, model.predict(X_val))
    assert mae <= MAX_ACCEPTABLE_MAE, f'MAE {mae:.2f} exceeds threshold {MAX_ACCEPTABLE_MAE}'


def test_model_rmse_within_threshold(model_and_validation_data):
    model, X_val, y_val = model_and_validation_data
    rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    assert rmse <= MAX_ACCEPTABLE_RMSE, f'RMSE {rmse:.2f} exceeds threshold {MAX_ACCEPTABLE_RMSE}'


def test_predictions_are_non_negative(model_and_validation_data):
    model, X_val, _ = model_and_validation_data
    predictions = model.predict(X_val)
    assert (predictions >= 0).all(), 'Model produced negative RUL predictions'


def test_predictions_respect_rul_cap(model_and_validation_data):
    model, X_val, _ = model_and_validation_data
    predictions = model.predict(X_val)
    assert predictions.max() <= RUL_CAP * 1.1, (
        f'Predictions exceed the RUL cap ({RUL_CAP}) by more than expected'
    )