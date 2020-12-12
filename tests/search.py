from model_selection_framework.search import *
from sklearn.linear_model import LogisticRegression
import warnings
warnings.simplefilter("ignore")


lr_configs = {
    'folder': 'tests',
    'model_init': LogisticRegression,
    'name': 'LogisticRegression',
    'init_parameters': {
        'solver': ['saga'],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'C': [0.01, 0.1, 0.5, 1.0, 1.5, 2.5, 10, 25, 50],
        'fit_intercept': [True, False],
        'l1_ratio': [0.5]
    }
}

from project_code.data_utils import *

users_df, cls_embedding, mean_embedding = read_data('data')
users_df[y_col] = (users_df[y_col] > 0) * 1
merged_df = merge_tables(users_df, cls_embedding, mean_embedding)
cls_cols = [c for c in merged_df.columns if c.endswith('_cls')]
mean_cols = [c for c in merged_df.columns if c.endswith('_mean')]
train, test = split_data(merged_df, test_size=0.3, random_state=42, regular=False)
preprocess = Preprocess()
train[facebook_cols + demo_cols] = preprocess.scale(train[facebook_cols + demo_cols], True)
test[facebook_cols + demo_cols] = preprocess.scale(test[facebook_cols + demo_cols], False)
train[facebook_cols + demo_cols] = preprocess.fillna(train[facebook_cols + demo_cols], True)
test[facebook_cols + demo_cols] = preprocess.fillna(test[facebook_cols + demo_cols], False)
X_cols = facebook_cols + demo_cols + personal_cols + psychosocial_cols + psychiatric_cols + cls_cols + mean_cols
X_train, X_test, y_train, y_test = train[X_cols], test[X_cols], train[y_col], test[y_col]
results = search(lr_configs, (X_train, y_train), (X_test, y_test), search_type='greedy',
                max_iter=100, metric='accuracy', max_rounds=3, random_start=10, beam_size=3)
