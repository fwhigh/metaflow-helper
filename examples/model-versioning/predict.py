from metaflow import FlowSpec, step, Parameter

import common
from metaflow_helper.utils import install_dependencies


class Predict(FlowSpec):

    configuration = Parameter(
        'configuration',
        help="Which config.py file to use",
        type=str,
        default='config',
    )

    train_tag = Parameter(
        'train_tag',
        help="Which train tag to use",
        type=str,
        default='v1',
    )

    @step
    def start(self):
        config = common.get_config(self.configuration)
        print(f'Running {config.__name__}')
        install_dependencies(config.dependencies)
        generate_data_run = common.get_generate_data_run()
        self.df = generate_data_run.data.predict_df
        self.numeric_features = generate_data_run.data.numeric_features
        self.categorical_features = generate_data_run.data.categorical_features
        self.make_regression_init_kwargs = generate_data_run.data.make_regression_init_kwargs
        if len(self.df) == 0:
            raise ValueError(self.df)
        print(f'prediction data contains {len(self.df)} rows and {len(self.df.columns)} columns')

        train_data_pointer = common.get_train_run(tags=[self.train_tag])
        self.train_run_id = train_data_pointer.id
        self.feature_engineer = train_data_pointer.data.feature_engineer
        self.modeler = train_data_pointer.data.modeler

        self.next(self.predict)

    @step
    def predict(self):
        X = self.feature_engineer.transform(self.df)
        self.y = self.modeler.predict(X)
        print(self.y)
        if len(self.y) != len(X):
            raise ValueError(
                f'The number of predictions {len(self.y)} does not match the number of feature vectors {len(X)}'
            )
        print(len(self.y))
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    Predict()
