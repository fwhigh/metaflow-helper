from metaflow import FlowSpec, step, Parameter
from sklearn.model_selection import train_test_split, KFold

import common
from metaflow_helper.utils import explode_parameters, install_dependencies


class GenerateData(FlowSpec):

    configuration = Parameter(
        'configuration',
        help="Which config.py file to use",
        type=str,
        default='config',
    )

    @step
    def start(self):
        config = common.get_config(self.configuration)
        print(f'Running {config.__name__}')
        install_dependencies(config.dependencies)
        df, self.numeric_features, self.categorical_features = common.generate_data(
            n_numeric_features=config.n_numeric_features,
            init_kwargs=config.make_regression_init_kwargs,
        )
        self.make_regression_init_kwargs = config.make_regression_init_kwargs
        print(f'generated {len(df)} rows and {len(df.columns)} columns')

        # Reserve some rows of the dataframe for prediction.
        train_validation_test_index, predict_index = train_test_split(
            df.index, test_size=config.test_size,
        )
        self.train_df = df.loc[train_validation_test_index, :]
        self.predict_df = df.loc[predict_index, :]

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    GenerateData()
