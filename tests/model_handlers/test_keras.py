import numpy as np
from sklearn.metrics import r2_score
from metalearn.model_handlers import KerasRegressorHandler, build_keras_model
from metalearn.constants import RunMode


def test_keras_model_regressor_handler_train():
    n_examples = 10
    n_repeat = 10
    offset = 0
    X = np.repeat(np.arange(n_examples).astype(float)/n_examples, n_repeat)[:, None]
    y = np.repeat(np.arange(n_examples).astype(float)/n_examples + offset, n_repeat)

    model_handler = KerasRegressorHandler(
        build_model=build_keras_model,
        mode=RunMode.TRAIN,
        input_dim=1,
        dense_layer_widths=(),
        dropout_probabilities=(),
    )
    model_handler.fit(X, y, epochs=1000, verbose=0)
    y_pred = model_handler.predict(X)
    np.testing.assert_allclose(y, y_pred.T[0], rtol=2)
    assert r2_score(y, y_pred) > 0.9


def test_keras_model_regressor_handler_test():
    n_examples = 10
    n_repeat = 10
    offset = 0
    X = np.repeat(np.arange(n_examples).astype(float)/n_examples, n_repeat)[:, None]
    y = np.repeat(np.arange(n_examples).astype(float)/n_examples + offset, n_repeat)

    model_handler = KerasRegressorHandler(
        build_model=build_keras_model,
        mode=RunMode.TEST,
        input_dim=1,
        dense_layer_widths=(),
        dropout_probabilities=(),
        eval_metric='mse',
    )
    model_handler.fit(X, y, epochs=1000, verbose=0, validation_split=0.1, patience=2)
    y_pred = model_handler.predict(X)
    np.testing.assert_allclose(y, y_pred.T[0], rtol=2)
    assert r2_score(y, y_pred) > 0.9
