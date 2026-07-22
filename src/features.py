from sklearn.model_selection import GroupShuffleSplit

from data_loading import load_train, COLUMN_NAMES

RUL_CAP = 125


def get_feature_columns(df):
    sensor_cols = [c for c in COLUMN_NAMES if c.startswith('sensor_')]
    variances = df[sensor_cols].var()
    constant_sensors = variances[variances == 0].index.tolist()
    return [c for c in sensor_cols if c not in constant_sensors]


def prepare_training_data(test_size=0.2, random_state=42):
    train = load_train()
    train['RUL'] = train['RUL'].clip(upper=RUL_CAP)

    feature_cols = get_feature_columns(train)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(train, groups=train['unit_number']))

    X_train = train.iloc[train_idx][feature_cols]
    y_train = train.iloc[train_idx]['RUL']
    X_val = train.iloc[val_idx][feature_cols]
    y_val = train.iloc[val_idx]['RUL']

    return X_train, X_val, y_train, y_val, feature_cols


if __name__ == '__main__':
    X_train, X_val, y_train, y_val, feature_cols = prepare_training_data()
    print('Feature columns used:', feature_cols)
    print('Train rows:', len(X_train), '| Validation rows:', len(X_val))
    print('RUL range in train:', y_train.min(), '-', y_train.max())
