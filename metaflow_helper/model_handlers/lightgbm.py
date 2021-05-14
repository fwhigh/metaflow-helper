import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin

from ..constants import RunMode


class LightGBMRegressorHandler(BaseEstimator, RegressorMixin):

    def __init__(self, mode: RunMode, iterations=None, **kwargs):
        self.mode = mode
        self.iterations = iterations
        if self.iterations is not None:
            kwargs['n_estimators'] = self.iterations

        self.model = lgb.LGBMRegressor(**kwargs)

    def fit(self, X, y, validation_data=None, **kwargs):
        kwargs = dict(kwargs)
        if self.mode is RunMode.TRAIN:
            for k in ['early_stopping_rounds']:
                try:
                    kwargs.pop(k)
                except KeyError:
                    pass
        if validation_data is not None:
            self.model.fit(X, y, eval_set=validation_data, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)
        if self.mode is RunMode.TEST:
            self.iterations = self.model.best_iteration_
        return self

    def predict(self, X, *args, **kwargs):
        return self.model.predict(X, *args, **kwargs)
