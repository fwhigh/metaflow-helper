from typing import Callable, Tuple
from sklearn.pipeline import Pipeline

from .constants import RunMode
from .utils import import_object_from_string


class Modeler:
    MODEL_STEP_NAME = 'model'

    def __init__(self, pipeline_fn: Callable, hyperparameters=None, input_dim=None,
                 mode: RunMode = None, iterations: int = None, **kwargs):
        self.hyperparameters = hyperparameters
        self.input_dim = input_dim
        self.mode = mode
        self.iterations = iterations
        self.model_init_kwargs = None
        self.model_fit_kwargs = None
        self.pipeline: Pipeline

        self.hyperparameters = self.update_hyperparameters(
            self.hyperparameters,
            mode=self.mode,
            input_dim=self.input_dim,
            best_iterations=self.iterations,
        )

        self.model_cls = import_object_from_string(self.hyperparameters[f'__{self.MODEL_STEP_NAME}'])
        self.model_init_kwargs = self.extract_model_init_kwargs(self.hyperparameters)

        self.pipeline = pipeline_fn(
            model=self.model_cls(**self.model_init_kwargs),
        )

    def fit(self, X, y=None, **kwargs):
        self.model_fit_kwargs = self.extract_model_fit_kwargs(self.hyperparameters)
        return self.pipeline.fit(X, y, **self.model_fit_kwargs, **kwargs)

    def transform(self, X, y=None):
        return self.pipeline.transform(X, y)

    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def update_hyperparameters(self, hyperparameters, mode: RunMode, input_dim=None, best_iterations=None):
        if mode is RunMode.TEST:
            pass
        elif mode is RunMode.TRAIN:
            if best_iterations is None:
                ValueError(f'best_iterations must be set in {mode}, was {best_iterations}')
            hyperparameters.update({
                f'__{self.MODEL_STEP_NAME}__init_kwargs__iterations': best_iterations,
            })
        hyperparameters.update({
            f'__{self.MODEL_STEP_NAME}__init_kwargs__input_dim': input_dim,
            f'__{self.MODEL_STEP_NAME}__init_kwargs__mode': mode,
        })
        return hyperparameters

    def extract_model_init_kwargs(self, hyperparameters):
        return self.parse_hyperparameters(
            hyperparameters,
            prefix_filter=f'__{self.MODEL_STEP_NAME}__init_kwargs__',
            substring_replace=(
                f'__{self.MODEL_STEP_NAME}__init_kwargs__',
                '',
            )
        )

    def extract_model_fit_kwargs(self, hyperparameters):
        """
        This deals with sklearn Pipeline's way of handing fit kwargs, which is to prepend with f'{step_name}__'. See
        https://scikit-learn.org/stable/modules/compose.html#nested-parameters.
        """
        return self.parse_hyperparameters(
            hyperparameters,
            prefix_filter=f'__{self.MODEL_STEP_NAME}__fit_kwargs__',
            substring_replace=(
                f'__{self.MODEL_STEP_NAME}__fit_kwargs__',
                f'{self.MODEL_STEP_NAME}__'
            ),
        )

    @staticmethod
    def parse_hyperparameters(hyperparameters, prefix_filter, substring_replace: Tuple[str, str]):
        return {
            str(k).replace(substring_replace[0], substring_replace[1]): v
            for k, v in hyperparameters.items()
            if k.startswith(prefix_filter)
        }

    def get_iterations(self):
        return self.pipeline.named_steps[self.MODEL_STEP_NAME].iterations
