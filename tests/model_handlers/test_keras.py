import numpy as np
from sklearn.metrics import r2_score
from metalearn.model_handlers import KerasRegressorHandler, build_keras_model
from metalearn.constants import RunMode


def test_keras_model_regressor_handler():
    n_examples = 100
    n_repeat = 100
    offset = 10
    X = np.repeat(np.arange(n_examples), n_repeat)[:, None]
    y = np.repeat(np.arange(n_examples).astype(float) + offset, n_repeat)

    model_handler = KerasRegressorHandler(
        build_model=build_keras_model,
        mode=RunMode.TRAIN,
        input_dim=1,
        dense_layer_widths=(1,),
        dropout_probabilities=(0,),
    )
    model_handler.fit(X, y, epochs=100, verbose=0)
    y_pred = model_handler.predict(X)
    np.testing.assert_allclose(y, y_pred, rtol=2)
    assert r2_score(y, y_pred) > 0.9
