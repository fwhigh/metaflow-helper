from typing import Callable
from sklearn.pipeline import Pipeline


class FeatureEngineer:
    def __init__(self, pipeline_fn: Callable, **kwargs):
        self.pipeline: Pipeline = pipeline_fn(**kwargs)
        if not isinstance(self.pipeline, Pipeline):
            raise ValueError(f'pipeline_fn must return an instance of sklearn.pipeline.Pipeline, was {type(self.pipeline)}')

    def fit(self, X):
        return self.pipeline.fit(X)

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)

    def predict(self, X):
        return self.pipeline.predict(X)
