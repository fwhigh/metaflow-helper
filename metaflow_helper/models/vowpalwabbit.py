from vowpalwabbit import pyvw
from sklearn.base import BaseEstimator, RegressorMixin

from ..constants import RunMode
from .base import BaseModel


class VowpalWabbitRegressor(BaseModel, BaseEstimator, RegressorMixin):
    pass
