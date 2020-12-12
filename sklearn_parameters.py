from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


KNeighborsClassifier_configs = {
    'model_init': KNeighborsClassifier,
    'name': 'KNeighborsClassifier',
    'init_parameters': {
        'n_neighbors': [1, 2, 3, 5, 7, 13, 19, 31, 43, 59, 71, 89, 109, 139, 167, 193, 229, 263],
        'weights' : ['uniform', 'distance'],
        'p': [1, 2, 3]
    }
}

LogisticRegression_configs = {
    'model_init': LogisticRegression,
    'name': 'LogisticRegression',
    'init_parameters': {
        'solver': ['saga'],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'C': [0.001, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0],
    }
}


LinearSVC_configs = {
    'model_init': LinearSVC,
    'name': 'LinearSVC',
    'init_parameters': {
        'loss' : ['hinge', 'squared_hinge'],
        'tol': [0.01, 0.001, 0.0001, 0.00001],
        'C': [0.001, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0],
    }
}


RandomForestClassifier_configs = {
    'model_init': RandomForestClassifier,
    'name': 'RandomForestClassifier',
    'init_parameters': {
        'n_estimators': [5, 15, 30, 50, 75, 100, 125],
        'max_depth':  [5, 10, 20, 25, 50, 75],
        'min_samples_split': [2, 5, 10, 25, 50, 75],
        'min_samples_leaf': [2, 5, 10, 25, 50, 75]
    }
}


XGBClassifier_configs = {
    'model_init': XGBClassifier,
    'name': 'XGBClassifier',
    'init_parameters': {
        'n_estimators': [5, 15, 30, 50, 75, 100, 125],
        'max_depth': [5, 10, 20, 25, 50, 75],
        'gamma': [None, 0.01, 0.1, 1, 5, 10],
        'learning_rate': [None, 0.0001, 0.001, 0.01, 0.05, 0.1],
        'min_child_weight': [None, 1, 3, 5, 10, 15],
    }
}