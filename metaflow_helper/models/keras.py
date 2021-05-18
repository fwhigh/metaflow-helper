import tempfile
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, InputLayer
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import models as keras_models

from ..constants import RunMode
from .base import BaseModel
from ..utils import import_object_from_string


def make_keras_picklable():
    """
    Pure magic from http://zachmoshe.com/2017/04/03/pickling-keras-models.html.

    https://stackoverflow.com/questions/48295661/how-to-pickle-keras-model
    """
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras_models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras_models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras_models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


make_keras_picklable()


class KerasRegressor(BaseModel, BaseEstimator, RegressorMixin):

    def __init__(self, build_model=None, input_dim=None, mode=RunMode, iterations=None, eval_metric=None, **kwargs):
        self.build_model = build_model
        self.input_dim = input_dim
        self.mode = mode
        self.min_delta = 0
        self.patience = None
        self.callbacks = None
        self.monitor = eval_metric
        self.history = []
        self.iterations = iterations
        self.verbose = 0

        self._validate_init_kwargs()
        self.model = import_object_from_string(self.build_model)(input_dim=self.input_dim, **kwargs)

    def fit(self, X, y, validation_data=None, patience=None, min_delta=0, monitor='mse', **kwargs):
        kwargs = dict(kwargs)
        self.patience = patience
        self.min_delta = min_delta
        if self.iterations is not None:
            kwargs['epochs'] = self.iterations
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        if self.mode is RunMode.TEST:
            self.monitor = monitor
            if self.patience is None:
                ValueError(self.patience)
            self.callbacks = [
                EarlyStopping(
                    monitor=self.monitor,
                    min_delta=self.min_delta,
                    verbose=self.verbose,
                    patience=self.patience,
                ),
            ]
        elif self.mode is RunMode.TRAIN:
            for k in ['validation_split', 'eval_metric']:
                try:
                    kwargs.pop(k)
                except KeyError:
                    pass
        self._validate_fit_kwargs()
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
                                 l1_factor=0, l2_factor=0):
    if input_dim is None:
        raise ValueError(input_dim)
    if len(dense_layer_widths) > len(dropout_probabilities):
        dropout_probabilities = tuple([dropout_probabilities[0]]*len(dense_layer_widths))
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dim, )))
    for i, params in enumerate(zip(dense_layer_widths, dropout_probabilities)):
        dense_layer_width, dropout_probability = params
        model.add(Dense(
            dense_layer_width, activation=activation, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor),
            bias_regularizer=regularizers.l2(l2_factor),
            activity_regularizer=regularizers.l2(l2_factor)
        ))
        model.add(Dropout(dropout_probability))
    model.add(Dense(
        1, activation=activation, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor),
        bias_regularizer=regularizers.l2(l2_factor),
        activity_regularizer=regularizers.l2(l2_factor)
    ))
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    return model
