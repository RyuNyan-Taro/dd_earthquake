import pandas as pd
from pathlib import Path


def read_data(train_test: str):
    DATA_DIR = Path('..', '..', '..', 'data', 'final', 'public')
    if train_test == 'training':
        return pd.read_csv(DATA_DIR / 'train_values.csv', index_col='building_id'), \
               pd.read_csv(DATA_DIR / 'train_labels.csv', index_col='building_id')
    if train_test == 'test':
        return pd.read_csv(DATA_DIR / 'test_values.csv', index_col='building_id')
    raise ValueError('You must select {training, test} as train_test.')
