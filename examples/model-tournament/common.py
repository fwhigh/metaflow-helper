import os, errno
from importlib import import_module
import subprocess
import time
import re
from pathlib import Path
import pickle
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.graph_objects as go
import plotly

from metaflow_helper.constants import RunMode


def silent_rm_file(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


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


def install_dependencies(dependencies: list):
    for dependency in dependencies:
        for k, v in dependency.items():
            try:
                module_ = import_module(k)
            except ModuleNotFoundError:
                system_command_with_retry(['pip', 'install', v])


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
    return parse_contender(contender, prefix_filter='__init_kwargs__model__', prefix_to_remove='__init_kwargs__model__')


def parse_contender_model_fit(contender):
    return parse_contender(contender, prefix_filter='__fit_kwargs__model__', prefix_to_remove='__fit_kwargs__')


def parse_contender(contender, prefix_filter, prefix_to_remove):
    return {re.sub(r'^' + re.escape(prefix_to_remove), '', k): v for k, v in contender.items() if k.startswith(prefix_filter)}


def update_contender(contender, mode: RunMode, input_dim=None, best_iterations=None):
    if mode is RunMode.TEST:
        pass
    elif mode is RunMode.TRAIN:
        contender.update({
            '__init_kwargs__model__iterations': best_iterations,
        })
    contender.update({
        '__init_kwargs__model__input_dim': input_dim,
        '__init_kwargs__model__mode': mode,
    })
    return contender


def plot_all_scores(contender_results, dir, auto_open=True):
    Path(dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame().from_records([{
        **pickle.loads(k),
        **contender_results[k]}
        for k in contender_results
    ])
    fig = go.Figure()
    for index, row in df.iterrows():
        fig.add_trace(
            go.Box(
                name=f"{row.name} {str(row['__model']).rsplit('.', 1)[1]}",
                x=(f"{row.name}",),
                y=row['scores'],
            ),
        )
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Score',
        template='none',
    )
    print(f'ADFDAFDASFDAS writing ' + f"{dir}/all-scores.png")
    silent_rm_file(f"{dir}/all-scores.png")
    if os.path.isfile(f"{dir}/all-scores.png"):
        raise FileExistsError(f"{dir}/all-scores.png")
    fig.write_image(f"{dir}/all-scores.png")
    print(f'writing ' + f"{dir}/all-scores.html")
    silent_rm_file(f"{dir}/all-scores.html")
    plotly.offline.plot(fig, filename=f"{dir}/all-scores.html", auto_open=auto_open)
    return fig
