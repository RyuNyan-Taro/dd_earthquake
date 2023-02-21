import pandas as pd
from pathlib import Path


def read_data(data_suffix: str) -> pd.DataFrame or ValueError:
    """
    Return data which selected data_suffix.

    Parameters
    ----------
    data_suffix : {''train', 'test', 'submission'}, optional
        What data is returned

    Returns
    -------
    DataFrame of selected as data_suffix.

    """
    DATA_DIR: Path = Path('..', '..', '..', 'data', 'final', 'public')
    if data_suffix == 'train':
        return pd.read_csv(DATA_DIR / 'train_values.csv', index_col='building_id'), \
               pd.read_csv(DATA_DIR / 'train_labels.csv', index_col='building_id')
    if data_suffix == 'test':
        return pd.read_csv(DATA_DIR / 'test_values.csv', index_col='building_id')
    if data_suffix == 'submission':
        return pd.read_csv(DATA_DIR / 'submission_format.csv', index_col='building_id')

    # Return error if other data_suffix is selected
    raise ValueError('You must select {train, test, submission} as train_test.')
