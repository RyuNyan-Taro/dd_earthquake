from typing import Tuple, Any, Iterable

import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from . import file
from pandas import DataFrame


def f1(y_true, y_pred):
    """
    ref: https://qiita.com/ground0state/items/1b7cf1977426bd8f0f28
    """
    N_LABELS = 3  # ラベルの数
    y_pred_ = y_pred.reshape(N_LABELS, len(y_pred) // N_LABELS).argmax(axis=0)
    score = f1_score(y_true, y_pred_, average='micro')
    return 'f1', score, True


def lgbm_preprocessing(datas, mode: str = 'train', features_list=None):
    """
    Preprocessing for lgbm.

    Parameters
    ----------
    datas : DataFrame or list of DataFrame
        Input data
    mode : {'train', 'test'}, optional
        train returns values and label, and test returns values
    features_list : None or list of str
        List orf columns to use as features.

    Returns
    -------
    DataFrame of preprocessed

    """
    if mode == 'train':
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
                         'age',
                         'area_percentage',
                         'height_percentage',
                         'foundation_type',
                         'roof_type',
                         'ground_floor_type',
                         'other_floor_type',
                         'position',
                         'has_superstructure_mud_mortar_stone',
                         'has_superstructure_cement_mortar_brick',
                         'has_superstructure_timber',
                         'count_families',
                         'has_secondary_use']
    values = pd.get_dummies(values[features_list])

    # convert object to category
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
        'force_row_wise': True,
        "learning_rate": 0.15,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'metric': None,
        'n_estimators': 1000,
    }

    model = lgb.train(params, trains, valid_sets=valids, num_boost_round=1000, early_stopping_rounds=100)

    return x_train, x_test, y_train, y_test, model


def predict_submit(x_test, y_test, model):
    # predict of test data
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # print of true and predict
    df_pred = pd.DataFrame({'target': y_test['damage_grade'].values, 'target_pred': y_pred})
    print(df_pred)
    df_pred_prob = pd.DataFrame(
        {'y': y_test['damage_grade'].values, 'target0_prob': y_pred_prob[:, 0], 'target1_prob': y_pred_prob[:, 1],
         'target2_prob': y_pred_prob[:, 2]})
    print(df_pred_prob)

    acc = accuracy_score(y_test, y_pred)
    print('Acc :', acc)

    # submit the model
    test_values = file.read_data('test')
    test_values, _ = lgbm_preprocessing(test_values, mode='test')
    y_test_prob = model.predict(test_values)
    y_test = np.argmax(y_test_prob, axis=1)
    submission_format = file.read_data('submission')
    my_submission = pd.DataFrame(data=y_test + 1,
                                 columns=submission_format.columns,
                                 index=submission_format.index)

    return my_submission
