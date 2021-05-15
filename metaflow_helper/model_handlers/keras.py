from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras import regularizers

from ..constants import RunMode


class KerasRegressorHandler(BaseEstimator, RegressorMixin):

    def __init__(self, build_model=None, input_dim=None, mode=RunMode, iterations=None, eval_metric=None, **kwargs):
        self.build_model = build_model
        self.input_dim = input_dim
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
        if self.mode is RunMode.TEST:
            self.eval_metric = eval_metric
            if self.patience is None:
                ValueError(self.patience)
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
        return self.model.predict(X, *args, **kwargs).T[0]


def build_keras_regression_model(input_dim=None, dense_layer_widths=(10,), dropout_probabilities=(0,),
                                 metric='mse', optimizer='adam', loss='mean_squared_error', activation=None,
                                 kernel_initializer='random_normal', bias_initializer='random_normal',
                                 l1_lambdas=(0,), l2_lambdas=(0,),
                                 l1_lambda_final=0, l2_lambda_final=0):
    if input_dim is None:
        raise ValueError(input_dim)
    model = Sequential()
    model.add(Input(shape=(input_dim, )))
    for i, params in enumerate(zip(dense_layer_widths, dropout_probabilities, l1_lambdas, l2_lambdas)):
        dense_layer_width, dropout_probability, l1_lambda, l2_lambda = params
        model.add(Dense(
            dense_layer_width, activation=activation, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda),
            bias_regularizer=regularizers.l2(l2_lambda),
            activity_regularizer=regularizers.l2(l2_lambda)
        ))
        model.add(Dropout(dropout_probability))
    model.add(Dense(
        1, activation=activation, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l1_l2(l1=l1_lambda_final, l2=l2_lambda_final),
        bias_regularizer=regularizers.l2(l2_lambda_final),
        activity_regularizer=regularizers.l2(l2_lambda_final)
    ))
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    return model
