from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

from ..constants import RunMode


class KerasRegressorHandler(BaseEstimator, RegressorMixin):

    def __init__(self, build_model=None, input_dim=None, mode=RunMode, iterations=None, eval_metric=None, **kwargs):
        self.input_dim = input_dim
        self.mode = mode
        self.min_delta = 0
        self.patience = None
        self.callbacks = None
        self.eval_metric = eval_metric
        self.history = []
        self.iterations = iterations
        self.build_model = build_model

        self.model = self.build_model(input_dim=self.input_dim, **kwargs)

    def fit(self, X, y, validation_data=None, patience=None, min_delta=0, eval_metric=None, **kwargs):
        kwargs = dict(kwargs)
        self.patience = patience
        self.min_delta = min_delta
        if self.iterations is not None:
            kwargs['epochs'] = self.iterations
        if self.mode is RunMode.TEST:
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
        elif self.mode is RunMode.TRAIN:
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
        if self.mode is RunMode.TEST:
            self.iterations = len(self.history['loss']) - self.patience
        return self

    def predict(self, X, *args, **kwargs):
        return self.model.predict(X, *args, **kwargs)


def build_keras_model(input_dim=None, dense_layer_widths=(10,), dropout_probabilities=(0.5,), metric='mse',
                      optimizer='adam', loss='mean_squared_error', activation=None,
                      kernel_initializer='random_normal', bias_initializer='random_normal'):
    if input_dim is None:
        raise ValueError(input_dim)
    model = Sequential()
    for i, params in enumerate(zip(dense_layer_widths, dropout_probabilities)):
        dense_layer_width, dropout_probability = params
        if i == 0:
            model.add(Dense(
                dense_layer_width, input_dim=input_dim, activation=activation, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer))
            if dropout_probability > 0:
                model.add(Dropout(dropout_probability))
        else:
            model.add(Dense(
                dense_layer_width, activation=activation, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer))
        model.add(Dense(1, activation=activation, kernel_initializer=kernel_initializer))
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    return model
