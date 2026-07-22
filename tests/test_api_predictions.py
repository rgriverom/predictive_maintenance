import pytest
import requests

from data_loading import load_train, load_test
from features import get_feature_columns, RUL_CAP

API_URL = 'http://127.0.0.1:8000'

# Sanity margin, not an exact-match check — RUL prediction is inherently
# noisy. This catches a badly broken pipeline, not small variance.
MAX_ALLOWED_ERROR = 30


def _api_is_up():
    try:
        requests.get(f'{API_URL}/health', timeout=1)
        return True
    except requests.exceptions.ConnectionError:
        return False


pytestmark = pytest.mark.skipif(
    not _api_is_up(),
    reason='serve_model.py is not running on http://127.0.0.1:8000 — start it first'
)


@pytest.fixture(scope='module')
def test_engines_data():
    train = load_train()
    feature_cols = get_feature_columns(train)
    test, test_rul = load_test()
    last_readings = test.sort_values('time_in_cycles').groupby('unit_number').tail(1)
    return last_readings, test_rul, feature_cols


@pytest.mark.parametrize('unit', [1, 2, 3, 4, 5])
def test_prediction_close_to_actual_rul(unit, test_engines_data):
    last_readings, test_rul, feature_cols = test_engines_data

    row = last_readings[last_readings['unit_number'] == unit].iloc[0]
    readings = {col: float(row[col]) for col in feature_cols}

    response = requests.post(f'{API_URL}/predict', json={'readings': readings})
    assert response.status_code == 200

    predicted_rul = response.json()['predicted_rul_cycles']
    actual_rul = test_rul[test_rul['unit_number'] == unit]['RUL'].iloc[0]
    actual_rul_capped = min(actual_rul, RUL_CAP)

    error = abs(predicted_rul - actual_rul_capped)
    assert error <= MAX_ALLOWED_ERROR, (
        f'Engine {unit}: predicted={predicted_rul:.1f}, actual(capped)={actual_rul_capped}, '
        f'error={error:.1f} exceeds allowed {MAX_ALLOWED_ERROR}'
    )