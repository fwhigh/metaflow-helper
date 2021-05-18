n_numeric_features = 10
n_informative_numeric_features = 5
n_categorical_features = 1
make_regression_init_kwargs = {
    f'type_{i}': {
        'n_samples': round(1_000/n_categorical_features),
        'noise': 100,
        'n_features': n_numeric_features,
        'n_informative': n_informative_numeric_features,
        'coef': True,
        'random_state': i,
    }
    for i in range(n_categorical_features)
}
test_size = 0.2
n_splits = 5
score_path = 'sklearn.metrics.r2_score'
contenders_spec = [
    {
        # This is the algo
        '__model': ['metaflow_helper.models.LightGBMRegressor'],
        # These go to the model initializer
        '__model__init_kwargs__learning_rate': [0.1],
        '__model__init_kwargs__max_depth': [1, 2, 3],
        '__model__init_kwargs__n_estimators': [10_000],
        # These go to the model fitter
        '__model__fit_kwargs__eval_metric': ['mse'],
        '__model__fit_kwargs__early_stopping_rounds': [10],
        '__model__fit_kwargs__verbose': [0],
    },
    {
        # This is the algo
        '__model': ['metaflow_helper.models.KerasRegressor'],
        # These go to the model initializer
        '__model__init_kwargs__build_model': ['metaflow_helper.models.build_keras_regression_model'],
        '__model__init_kwargs__metric': ['mse'],
        '__model__init_kwargs__dense_layer_widths': [(), (15,), (15, 15,), (15 * 15,)],
        # These go to the model fitter
        '__model__fit_kwargs__batch_size': [None],
        '__model__fit_kwargs__epochs': [10_000],
        '__model__fit_kwargs__validation_split': [0.2],
        '__model__fit_kwargs__monitor': ['val_mse'],
        '__model__fit_kwargs__verbose': [0],
        '__model__fit_kwargs__patience': [10],
        '__model__fit_kwargs__min_delta': [0.1],
    },
]
dependencies = [
    {'metaflow_helper': 'git+ssh://git@github.com/fwhigh/metaflow-helper.git'},
]
auto_open_figures = True
