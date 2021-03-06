import numpy as np
from sklearn.metrics import r2_score
from metaflow_helper.models import LightGBMRegressor
from metaflow_helper.constants import RunMode


def test_lightgbm_model_regressor_handler_train():
    n_examples = 10
    n_repeat = 10
    offset = 10
    X = np.repeat(np.arange(n_examples), n_repeat)[:, None]
    y = np.repeat(np.arange(n_examples).astype(float) + offset, n_repeat)

    model_handler = LightGBMRegressor(
        mode=RunMode.TRAIN,
        max_depth=1,
        min_child_samples=1,
        iterations=100,
    )
    model_handler.fit(X, y)
    y_pred = model_handler.predict(X)
    np.testing.assert_allclose(y, y_pred, rtol=2)
    assert r2_score(y, y_pred) > 0.9


def test_lightgbm_model_regressor_handler_test():
    n_examples = 10
    n_repeat = 10
    offset = 10
    X = np.repeat(np.arange(n_examples), n_repeat)[:, None]
    y = np.repeat(np.arange(n_examples).astype(float) + offset, n_repeat)

    model_handler = LightGBMRegressor(
        mode=RunMode.TEST,
        max_depth=1,
        min_child_samples=1,
        iterations=100,
    )
    model_handler.fit(X, y)
    y_pred = model_handler.predict(X)
    np.testing.assert_allclose(y, y_pred, rtol=2)
    assert r2_score(y, y_pred) > 0.9
