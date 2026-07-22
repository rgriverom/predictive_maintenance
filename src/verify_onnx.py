import json
from pathlib import Path

import numpy as np
import mlflow.sklearn
import onnxruntime as ort

from features import prepare_training_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / 'model_export'


def main():
    sklearn_model = mlflow.sklearn.load_model(str(EXPORT_DIR / 'model'))

    session = ort.InferenceSession(str(EXPORT_DIR / 'model.onnx'))
    input_name = session.get_inputs()[0].name

    _, X_val, _, y_val, feature_cols = prepare_training_data()
    X_val_np = X_val[feature_cols].to_numpy(dtype=np.float32)

    sklearn_preds = sklearn_model.predict(X_val_np)
    onnx_preds = session.run(None, {input_name: X_val_np})[0].flatten()

    max_diff = np.max(np.abs(sklearn_preds - onnx_preds))
    print(f'Max prediction difference (sklearn vs ONNX): {max_diff:.6f}')

    if max_diff < 0.01:
        print('PASS: ONNX model predictions match the original model.')
    else:
        print('WARNING: predictions differ more than expected — investigate before using the ONNX model.')


if __name__ == '__main__':
    main()