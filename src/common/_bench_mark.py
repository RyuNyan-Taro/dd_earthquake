import pandas as pd
from pathlib import Path
# ML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


def select_columns(df: pd.DataFrame, features_mode: str, features_list: list) -> pd.DataFrame:
    if features_mode == 'select':
        return pd.get_dummies(df[features_list])
    elif features_mode == 'drop':
        return pd.get_dummies(df.drop(columns=features_list))
    else:
        raise ValueError(f'{features_mode} is not defined as a features_mode.')


def pred_bench_mark(features_mode: str = 'select', features_list=None):
    """
    Act fit, predict and return submit which is the same process as bench_mark.

    Parameters
    ----------
    features_mode : {'select', ''drop}, optional
        Select using or dropping features_list columns.
    features_list : list of str
        Columns list of DataFrame.

    Returns
    -------
    Submitted DataFrame

    """
    DATA_DIR = Path('..', '..', '..', 'data', 'final', 'public')
    train_values = pd.read_csv(DATA_DIR / 'train_values.csv', index_col='building_id')
    train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv', index_col='building_id')

    # set of feature columns to use
    if features_list is None:
        features_list = ['foundation_type',
                         'area_percentage',
                         'height_percentage',
                         'count_floors_pre_eq',
                         'land_surface_condition',
                         'has_superstructure_cement_mortar_stone']

    # select of drop features_list columns
    train_values_subset = select_columns(train_values, features_mode, features_list)

    # fit data to model
    pipe = make_pipeline(StandardScaler(),
                         RandomForestClassifier(random_state=2018))
    param_grid = {'randomforestclassifier__n_estimators': [50, 100],
                  'randomforestclassifier__min_samples_leaf': [1, 5]}
    gs = GridSearchCV(pipe, param_grid, cv=5)
    gs.fit(train_values_subset, train_labels.values.ravel())

    # predict and submit
    test_values = pd.read_csv(DATA_DIR / 'test_values.csv', index_col='building_id')

    test_values_subset = select_columns(test_values, features_mode, features_list)
    predictions = gs.predict(test_values_subset)
    submission_format = pd.read_csv(DATA_DIR / 'submission_format.csv', index_col='building_id')
    my_submission = pd.DataFrame(data=predictions,
                                 columns=submission_format.columns,
                                 index=submission_format.index)

    return my_submission
