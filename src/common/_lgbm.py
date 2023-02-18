import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from . import file


def lgbm_preprocessing(datas, mode='training', features_list=None):
    if mode == 'training':
        values = datas[0]
        labels = datas[1]
    elif mode == 'test':
        values = datas
        labels = None
    else:
        raise ValueError(f'{mode} is not defined.')

    # Use only some columns
    if features_list is None:
        features_list = ['geo_level_1_id',
                         'geo_level_2_id',
                         'geo_level_3_id',
                         'height_percentage',
                         'has_superstructure_adobe_mud',
                         'has_superstructure_mud_mortar_stone',
                         'has_superstructure_rc_non_engineered',
                         'has_superstructure_timber',
                         'foundation_type',
                         'roof_type',
                         'ground_floor_type']
    trian_values = pd.get_dummies(values[features_list])

    # convert obkect to category
    for _col in values.select_dtypes(include='object'):
        values[_col] = values[_col].astype("category")

    if labels is None:
        pass
    # convert labels range [1, 4) -> [0, 3)
    else:
        labels = labels - 1

    return values, labels


def split_modeling(values, labels):
    x_train, x_test, y_train, y_test = train_test_split(values, labels,
                                                        test_size=0.1, random_state=19, stratify=labels)
    trains = lgb.Dataset(x_train, y_train)
    valids = lgb.Dataset(x_test, y_test)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metrics": "multi_logloss",
        'force_row_wise': True,
        "learning_rate": 0.2

    }

    model = lgb.train(params, trains, valid_sets=valids, num_boost_round=1000, early_stopping_rounds=100)

    return x_train, x_test, y_train, y_test, model


