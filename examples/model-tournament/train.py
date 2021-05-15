import pickle
import json
import collections
from pathlib import Path
from scipy import stats
import numpy as np
from metaflow import FlowSpec, step, Parameter, current
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from sklearn.metrics import r2_score
import config
import common

from metaflow_helper.constants import RunMode


class Train(FlowSpec):

    @step
    def start(self):
        common.install_dependencies(config.dependencies)
        self.df, self.numeric_features, self.categorical_features = common.generate_data(
            n_numeric_features=config.n_numeric_features,
            init_kwargs=config.make_regression_init_kwargs,
        )
        self.make_regression_init_kwargs = config.make_regression_init_kwargs
        print(f'generated {len(self.df)} rows and {len(self.df.columns)} columns')

        self.train_validation_index, self.test_index = train_test_split(
            self.df.index, test_size=config.test_size,
        )

        self.contenders = ParameterGrid(config.contenders_spec)
        if config.n_splits > 1:
            self.k_fold = KFold(n_splits=config.n_splits)
        else:
            self.k_fold = None
        self.folds = list(range(config.n_splits))

        self.next(self.foreach_contender, foreach='contenders')

    @step
    def foreach_contender(self):
        common.install_dependencies(config.dependencies)
        self.contender = self.input

        self.next(self.foreach_fold, foreach='folds')

    @step
    def foreach_fold(self):
        common.install_dependencies(config.dependencies)
        self.fold = self.input
        contender = self.contender
        fit_kwargs = contender['__fit_kwargs']

        X = self.df.loc[self.train_validation_index, :]
        y = self.df.loc[self.train_validation_index, 'target']
        if config.n_splits > 1:
            train, test = list(self.k_fold.split(X))[self.fold]
        else:
            train, test = train_test_split(list(range(X.shape[0])), test_size=config.test_size)
        X_train = X.iloc[train, :]
        y_train = y.iloc[train]
        X_test = X.iloc[test, :]
        y_test = y.iloc[test]

        preprocessor_pipeline = common.build_preprocessor_pipeline(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
        )
        X_train_transformed = preprocessor_pipeline.fit_transform(X_train)
        X_test_transformed = preprocessor_pipeline.transform(X_test)

        contender = common.update_contender(contender, mode=RunMode.TEST, input_dim=X_train_transformed.shape[1])

        model_handler_class = common.import_object_from_string(contender['__model'])

        model_pipeline = common.build_model_pipeline(
            model=model_handler_class(**common.parse_contender_model_init(contender)),
        )
        model_pipeline.fit(
            X_train_transformed,
            y_train,
            model__validation_data=(X_test_transformed, y_test),
            **fit_kwargs,
        )
        self.score = r2_score(y_test, model_pipeline.predict(X_test_transformed))
        self.iterations = model_pipeline.named_steps['model'].iterations
        print(f'fold {self.fold}, score {self.score}, contender {contender}')

        self.next(self.end_foreach_fold)

    @step
    def end_foreach_fold(self, inputs):
        common.install_dependencies(config.dependencies)
        self.merge_artifacts(inputs, exclude=['fold', 'score', 'iterations'])
        self.contender_results = {
            'scores': [ii.score for ii in inputs],
            'iterations': [ii.iterations for ii in inputs],
        }
        self.contender_results['mean_score'] = np.mean(self.contender_results['scores'])
        self.contender_results['sem_score'] = stats.sem(self.contender_results['scores'])
        self.contender_results['mean_iteration'] = np.mean(self.contender_results['iterations'])
        self.contender_results['sem_iteration'] = stats.sem(self.contender_results['iterations'])
        print(f'contender_results {self.contender_results}')
        self.next(self.end_foreach_contender)

    @step
    def end_foreach_contender(self, inputs):
        common.install_dependencies(config.dependencies)
        self.merge_artifacts(inputs, exclude=['contender', 'contender_results'])
        self.contender_results = {
            pickle.dumps(ii.contender): ii.contender_results
            for ii in inputs
        }
        self.next(self.train_test)

    @step
    def train_test(self):
        common.install_dependencies(config.dependencies)
        self.best_contender_ser = max(self.contender_results.keys(), key=lambda k: self.contender_results[k]['mean_score'])
        self.best_contender = pickle.loads(self.best_contender_ser)
        print(f'best_contender {self.best_contender}, contender_results {self.contender_results[self.best_contender_ser]}')
        self.best_iterations = round(self.contender_results[self.best_contender_ser]['mean_iteration'])
        contender = self.best_contender
        fit_kwargs = contender['__fit_kwargs']

        X_train = self.df.loc[self.train_validation_index, :]
        y_train = self.df.loc[self.train_validation_index, 'target']
        X_test = self.df.loc[self.test_index, :]
        y_test = self.df.loc[self.test_index, 'target']

        preprocessor_pipeline = common.build_preprocessor_pipeline(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
        )
        X_train_transformed = preprocessor_pipeline.fit_transform(X_train)
        X_test_transformed = preprocessor_pipeline.transform(X_test)

        contender = common.update_contender(
            contender, mode=RunMode.TEST, input_dim=X_train_transformed.shape[1],
        )

        model_handler_class = common.import_object_from_string(contender['__model'])

        model_pipeline = common.build_model_pipeline(
            model=model_handler_class(**common.parse_contender_model_init(contender)),
        )
        model_pipeline.fit(
            X_train_transformed,
            y_train,
            model__validation_data=(X_test_transformed, y_test),
            **fit_kwargs,
        )
        self.score = r2_score(y_test, model_pipeline.predict(X_test_transformed))
        print(f'score {self.score}, contender {contender}')

        self.next(self.train)

    @step
    def train(self):
        common.install_dependencies(config.dependencies)
        contender = self.best_contender
        fit_kwargs = contender['__fit_kwargs']

        X = self.df.loc[:, :]
        y = self.df.loc[:, 'target']

        preprocessor_pipeline = common.build_preprocessor_pipeline(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
        )
        X_transformed = preprocessor_pipeline.fit_transform(X)

        contender = common.update_contender(
            contender, mode=RunMode.TRAIN, input_dim=X_transformed.shape[1],
            best_iterations=self.best_iterations,
        )

        model_handler_class = common.import_object_from_string(contender['__model'])

        model_pipeline = common.build_model_pipeline(
            model=model_handler_class(**common.parse_contender_model_init(contender)),
        )
        model_pipeline.fit(X_transformed, y, **fit_kwargs)

        self.next(self.end)

    @step
    def end(self):
        indent = 4
        Path("results").mkdir(parents=True, exist_ok=True)
        with open(f'results/results-{current.run_id}.txt', 'w') as f:
            print(f'data set:\n{json.dumps(self.make_regression_init_kwargs, indent=indent)}', file=f)
            print('\n', file=f)
            for i, k in enumerate(sorted(self.contender_results.keys(), key=lambda k: -1 * self.contender_results[k]['mean_score'])):
                v = self.contender_results[k]
                contender = pickle.loads(k)
                for p in [ck for ck, cv in contender.items() if isinstance(cv, collections.Callable)]:
                    contender.pop(p)
                print(f'In place {i+1} with score {v["mean_score"]}', file=f)
                print(f'contender:\n{json.dumps(contender, indent=indent)}', file=f)
                print(f'results:\n{json.dumps(v, indent=indent)}', file=f)
                print('\n', file=f)

if __name__ == '__main__':
    Train()
