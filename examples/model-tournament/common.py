from importlib import import_module
import subprocess
import time
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

from metaflow_helper.constants import RunMode


def import_object_from_string(path):
    print(f'path {path}')
    path, obj_str = path.rsplit('.', 1)
    module_ = import_module(path)
    obj = getattr(module_, obj_str)
    return obj


def system_command_with_retry(cmd: list):
    for i in range(0, 5):
        wait_seconds = 2**i
        try:
            status = subprocess.run(cmd)
            if status.returncode != 0:
                print(f'command status was {status}, retrying after {wait_seconds} seconds')
                time.sleep(wait_seconds)
                continue
        except subprocess.CalledProcessError:
            print(f'command failed, retrying after {wait_seconds} seconds')
            time.sleep(wait_seconds)
            continue
        break


def install_dependecies(dependencies: list):
    system_command_with_retry(['pip', 'install', *dependencies])


def generate_data(init_kwargs, n_numeric_features):
    numeric_features = [f'num_{s}' for s in range(n_numeric_features)]
    df_all_list = []
    for category in init_kwargs:
        X_all, y_all, coef = make_regression(
            **init_kwargs[category],
        )
        df = pd.DataFrame(data={'target': y_all})
        df[numeric_features] = pd.DataFrame(
            data=X_all,
        )
        df['cat'] = category
        df_all_list.append(df)

    df_all = pd.concat(df_all_list, ignore_index=True)

    categorical_features = ['cat']

    return df_all, numeric_features, categorical_features


def build_preprocessor_pipeline(numeric_features,
                                categorical_features,
                                step_name='preprocessor'):
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
    )
    pipeline = Pipeline([
        (step_name, preprocessor),
    ])
    return pipeline


def build_model_pipeline(model, step_name='model'):
    return Pipeline([(step_name, model)])


def parse_contender_model_init(contender):
    return {k: v for k, v in contender.items() if not k.startswith('__')}


def update_contender(contender, mode: RunMode, input_dim=None, best_iterations=None):
    if mode is RunMode.TEST:
        pass
    elif mode is RunMode.TRAIN:
        contender.update({
            'iterations': best_iterations,
        })
    contender.update({
        'input_dim': input_dim,
        'mode': mode,
    })
    if '__build_model' in contender:
        contender.update({
            'build_model': import_object_from_string(contender['__build_model']),
        })
    return contender


def build_keras_model(input_dim=None, dense_layer_widths=(10,), dropout_probabilities=(0.5,), metric='mse', optimizer='adam',
                      loss='mean_squared_error', activation='relu'):
    if input_dim is None:
        raise ValueError(input_dim)
    model = Sequential()
    for i, params in enumerate(zip(dense_layer_widths, dropout_probabilities)):
        dense_layer_width, dropout_probability = params
        if i == 0:
            model.add(Dense(dense_layer_width, input_dim=input_dim, activation=activation))
            model.add(Dropout(dropout_probability))
        else:
            model.add(Dense(dense_layer_width, activation=activation))
        model.add(Dense(1, activation=activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    return model
