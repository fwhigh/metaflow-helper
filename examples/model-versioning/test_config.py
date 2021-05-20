from scipy.stats import randint, loguniform


n_numeric_features = 10
n_informative_numeric_features = 5
n_categorical_features = 1
make_regression_init_kwargs = {
    f'type_{i}': {
        'n_samples': round(1_00/n_categorical_features),
        'noise': 100,
        'n_features': n_numeric_features,
        'n_informative': n_informative_numeric_features,
        'coef': True,
        'random_state': i,
    }
    for i in range(n_categorical_features)
}
test_size = 0.2
n_splits = 1
score_path = 'sklearn.metrics.r2_score'
contenders_spec = [
    {
        # This is the algo
        '__model': ['metaflow_helper.models.LightGBMRegressor'],
        # These go to the model initializer
        '__model__init_kwargs__learning_rate': loguniform(1e-2, 1e-1),
        '__model__init_kwargs__max_depth': randint(1, 3),
        '__model__init_kwargs__n_estimators': [10],
        # These go to the model fitter
        '__model__fit_kwargs__eval_metric': ['mse'],
        '__model__fit_kwargs__early_stopping_rounds': [2],
        '__model__fit_kwargs__verbose': [0],
        # The presence of this key triggers randomized search
        '__n_iter': 1,
    },
]
dependencies = [
    {'metaflow_helper': 'git+ssh://git@github.com/fwhigh/metaflow-helper.git'},
]
auto_open_figures = True
