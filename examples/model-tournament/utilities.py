from importlib import import_module
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def import_object_from_string(path):
    path, obj_str = path.rsplit('.', 1)
    module_ = import_module(path)
    obj = getattr(module_, obj_str)
    return obj


def generate_data(init_kwargs, n_numeric_features):
    numeric_features = [f'num_{s}' for s in range(n_numeric_features)]
    df_all_list = []
    for category in init_kwargs:
        X_all, y_all, coef = make_regression(
            **init_kwargs[category],
        )
        df = pd.DataFrame(data={'target': y_all})
        df[numeric_features] = pd.DataFrame(
            data=X_all,
        )
        df['cat'] = category
        df_all_list.append(df)

    df_all = pd.concat(df_all_list, ignore_index=True)

    categorical_features = ['cat']

    return df_all, numeric_features, categorical_features


class LightGBMHandler(BaseEstimator, RegressorMixin):

    def __init__(self, mode=None, iterations=None, **kwargs):
        if mode is not None:
            if mode not in ['train', 'test']:
                raise ValueError(f'mode most be one of ["train", "test"], was {mode}')
        self.mode = mode
        self.iterations = iterations
        if self.iterations is not None:
            kwargs['n_estimators'] = self.iterations

        self.model = lgb.LGBMRegressor(**kwargs)

    def fit(self, X, y, validation_data=None, **kwargs):
        kwargs = dict(kwargs)
        if self.mode is 'train':
            for k in ['early_stopping_rounds']:
                try:
                    kwargs.pop(k)
                except KeyError:
                    pass
        if validation_data is not None:
            self.model.fit(X, y, eval_set=validation_data, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)
        if self.mode is 'test':
            self.iterations = self.model.best_iteration_
        return self

    def predict(self, X, *args, **kwargs):
        return self.model.predict(X, *args, **kwargs)


class KerasHandler(BaseEstimator, RegressorMixin):

    def build_model(self, input_dim=-1, dense_layer_widths=(10,), dropout_probabilities=(0.5,), metric='mse',
                    optimizer='adam'):
        if input_dim < 1:
            raise ValueError(f'input_dim must be >= 1, was {input_dim}')
        model = Sequential()
        for i, params in enumerate(zip(dense_layer_widths, dropout_probabilities)):
            dense_layer_width, dropout_probability = params
            print(
                f'input_dim {input_dim} dense_layer_width {dense_layer_width} dropout_probability {dropout_probability}')
            if i == 0:
                model.add(Dense(dense_layer_width, input_dim=input_dim, activation='relu'))
                model.add(Dropout(dropout_probability))
            else:
                model.add(Dense(dense_layer_width, activation='relu'))
            model.add(Dense(1, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[metric])
        return model

    def __init__(self, input_dim=None, mode=None, iterations=None, eval_metric=None, **kwargs):
        self.input_dim = input_dim
        if mode is not None:
            if mode not in ['train', 'test']:
                raise ValueError(f'mode most be one of ["train", "test"], was {mode}')
        self.mode = mode
        self.min_delta = 0
        self.patience = None
        self.callbacks = None
        self.eval_metric = eval_metric
        self.history = []
        self.iterations = iterations

        self.model = self.build_model(input_dim=self.input_dim, **kwargs)

    def fit(self, X, y, validation_data=None, patience=None, min_delta=0, eval_metric=None, **kwargs):
        kwargs = dict(kwargs)
        self.patience = patience
        self.min_delta = min_delta
        if self.iterations is not None:
            kwargs['epochs'] = self.iterations
        if self.mode is 'test':
            self.eval_metric = eval_metric
            self.callbacks = [
                EarlyStopping(
                    monitor=self.eval_metric,
                    min_delta=self.min_delta,
                    mode='min',
                    verbose=1,
                    patience=self.patience,
                ),
            ]
        elif self.mode is 'train':
            for k in ['validation_split', 'eval_metric']:
                try:
                    kwargs.pop(k)
                except KeyError:
                    pass
        if kwargs is not None and 'validation_split' in kwargs:
            result = self.model.fit(X, y, callbacks=self.callbacks, **kwargs)
        else:
            result = self.model.fit(X, y, validation_data=validation_data, callbacks=self.callbacks, **kwargs)
        self.history = result.history
        if self.mode is 'test':
            self.iterations = len(self.history['loss']) - self.patience
        return self

    def predict(self, X, *args, **kwargs):
        return self.model.predict(X, *args, **kwargs)


def build_preprocessor_pipeline(numeric_features,
                                categorical_features,
                                step_name='preprocessor'):
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
    )
    pipeline = Pipeline([
        (step_name, preprocessor),
    ])
    return pipeline


def build_model_pipeline(model, step_name='model'):
    return Pipeline([(step_name, model)])
