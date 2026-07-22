from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

COLUMN_NAMES = (
    ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3']
    + [f'sensor_{i}' for i in range(1, 22)]
)


def load_train(path=None):
    path = path or DATA_DIR / 'train_FD001.txt'
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMN_NAMES)

    max_cycle = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['RUL'] = max_cycle - df['time_in_cycles']
    return df


def load_test(path=None, rul_path=None):
    path = path or DATA_DIR / 'test_FD001.txt'
    rul_path = rul_path or DATA_DIR / 'RUL_FD001.txt'

    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    rul['unit_number'] = rul.index + 1
    return df, rul


if __name__ == '__main__':
    train = load_train()
    test, test_rul = load_test()

    print('Train shape:', train.shape)
    print(train.head())
    print()
    print('Engines in train:', train['unit_number'].nunique())
    print('Engines in test:', test['unit_number'].nunique())
    print()
    print(test_rul.head())