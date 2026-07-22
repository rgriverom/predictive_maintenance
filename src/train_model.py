import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import prepare_training_data


def train_baseline_model():
    X_train, X_val, y_train, y_val, feature_cols = prepare_training_data()

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,  # use all CPU cores
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    print(f'Validation MAE:  {mae:.2f} cycles')
    print(f'Validation RMSE: {rmse:.2f} cycles')

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print('\nTop 5 most important sensors:')
    print(importances.sort_values(ascending=False).head(5))

    return model, feature_cols


if __name__ == '__main__':
    train_baseline_model()
