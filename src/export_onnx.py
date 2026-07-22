import json
from pathlib import Path

import mlflow.sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / 'model_export'


def main():
    model = mlflow.sklearn.load_model(str(EXPORT_DIR / 'model'))
    with open(EXPORT_DIR / 'feature_cols.json') as f:
        feature_cols = json.load(f)

    n_features = len(feature_cols)
    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(model, initial_types=initial_type)

    onnx_path = EXPORT_DIR / 'model.onnx'
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    print(f'Exported ONNX model ({n_features} input features) to {onnx_path}')


if __name__ == '__main__':
    main()