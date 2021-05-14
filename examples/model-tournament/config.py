n_numeric_features = 10
n_informative_numeric_features = 5
n_categorical_features = 5
make_regression_init_kwargs = {
    f'type_{i}': {
        'n_samples': 10_000,
        'noise': 10,
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
        '__model': ['metalearn.model_handlers.LightGBMRegressorHandler'],
        'learning_rate': [0.1],
        'max_depth': [1, 2, 3],
        'n_estimators': [10_000],
        '__fit_kwargs': [{
            'model__eval_metric': 'mse',
            'model__early_stopping_rounds': 10,
            'model__verbose': 0,
        }],
    },
    {
        # Anything with an underscore is a specially handled parameter
        '__model': ['metalearn.model_handlers.KerasRegressorHandler'],
        '__build_model': ['common.build_keras_model'],
        # These go to the model initializer
        'metric': ['mse'],
        'dense_layer_widths': [(100, )],
        'dropout_probabilities': [(0, )],
        # This goes to the pipeline elements' fitters by pipeline step stepname, where f'{stepname}__parameter' gets
        # renamed to parameter and then passed to the fitter for step stepname. The model stepname = 'model'
        # and the preprocessing stepname = 'preprocessor'. See utilities.build_pipeline.
        '__fit_kwargs': [{
            'model__batch_size': None,
            'model__epochs': 10_000,
            'model__validation_split': 0.2,
            'model__eval_metric': 'mse',  # monitor. Examples: 'mse' or 'val_mse'
            'model__verbose': 0,
            'model__patience': 10,
            'model__min_delta': 0.1,
        }],
    },
]
dependencies = ['metalearn']
