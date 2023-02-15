import pandas as pd
from pathlib import Path
# ML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


def pred_bench_mark(selected_features=None):
    DATA_DIR = Path('..', '..', '..', 'data', 'final', 'public')
    train_values = pd.read_csv(DATA_DIR / 'train_values.csv', index_col='building_id')
    train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv', index_col='building_id')

    # set of feature columns to use
    if selected_features is None:
        selected_features = ['foundation_type',
                             'area_percentage',
                             'height_percentage',
                             'count_floors_pre_eq',
                             'land_surface_condition',
                             'has_superstructure_cement_mortar_stone']
    train_values_subset = pd.get_dummies(train_values[selected_features])

    # fit data to model
    pipe = make_pipeline(StandardScaler(),
                         RandomForestClassifier(random_state=2018))
    param_grid = {'randomforestclassifier__n_estimators': [50, 100],
                  'randomforestclassifier__min_samples_leaf': [1, 5]}
    gs = GridSearchCV(pipe, param_grid, cv=5)
    gs.fit(train_values_subset, train_labels.values.ravel())

    # predict and submit
    test_values = pd.read_csv(DATA_DIR / 'test_values.csv', index_col='building_id')
    test_values_subset = pd.get_dummies(test_values[selected_features])
    predictions = gs.predict(test_values_subset)
    submission_format = pd.read_csv(DATA_DIR / 'submission_format.csv', index_col='building_id')
    my_submission = pd.DataFrame(data=predictions,
                                 columns=submission_format.columns,
                                 index=submission_format.index)

    return my_submission
