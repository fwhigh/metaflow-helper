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
contenders_spec = [
    {
        # This is the algo
        '__model': ['metaflow_helper.model_handlers.LightGBMRegressorHandler'],
        # These go to the model initializer
        '__init_kwargs__model__learning_rate': [0.1],
        '__init_kwargs__model__max_depth': [1],
        '__init_kwargs__model__n_estimators': [10],
        # These go to the model fitter
        '__fit_kwargs__model__eval_metric': ['mse'],
        '__fit_kwargs__model__early_stopping_rounds': [2],
        '__fit_kwargs__model__verbose': [0],
    },
    {
        # This is the algo
        '__model': ['metaflow_helper.model_handlers.KerasRegressorHandler'],
        # These go to the model initializer
        '__init_kwargs__model__build_model': ['metaflow_helper.model_handlers.build_keras_regression_model'],
        '__init_kwargs__model__metric': ['mse'],
        '__init_kwargs__model__dense_layer_widths': [(),],
        # These go to the model fitter
        '__fit_kwargs__model__batch_size': [None],
        '__fit_kwargs__model__epochs': [100],
        '__fit_kwargs__model__validation_split': [0.2],
        '__fit_kwargs__model__monitor': ['val_mse'],
        '__fit_kwargs__model__verbose': [0],
        '__fit_kwargs__model__patience': [2],
        '__fit_kwargs__model__min_delta': [0.1],
    },
]
dependencies = [
    {'metaflow_helper': 'git+ssh://git@github.com/fwhigh/metaflow-helper.git'},
]
auto_open_figures = True
