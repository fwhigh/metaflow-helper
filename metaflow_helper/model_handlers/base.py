import random
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly


class BaseModelHandler:
    def __init__(self):
        pass

    def _validate_init_kwargs(self):
        try:
            self.mode
        except NameError as e:
            print('You must make mode an init kwarg')
            raise e
        try:
            self.iterations
        except NameError as e:
            print('You must make iterations an init kwarg')
            raise e
        try:
            self.input_dim
        except NameError as e:
            print('You must make input_dim an init kwarg')
            raise e

    def _validate_fit_kwargs(self):
        pass

    def plot(self, y_true, y_pred, dir='.', auto_open=True):
        Path(dir).mkdir(parents=True, exist_ok=True)
        if len(y_true) > 1_000:
            idx = random.sample(range(len(y_true)), 1_000)
        else:
            idx = list(range(len(y_true)))
        x = y_pred.iloc[idx] if isinstance(y_pred, pd.Series) else y_pred[idx]
        y = y_true.iloc[idx] if isinstance(y_true, pd.Series) else y_true[idx]
        plot_range = [np.min((x, y)), np.max((x, y))]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
            ),
        )
        fig.add_shape(
            type="line",
            x0=plot_range[0], y0=plot_range[0], x1=plot_range[1], y1=plot_range[1],
            line=dict(
                color="Black",
                width=2,
            )
        )
        fig.update_layout(
            xaxis_title='Predicted',
            yaxis_title='True',
            template='none',
        )
        fig.write_image(f"{dir}/predicted-vs-true.png")
        plotly.offline.plot(fig, filename=f"{dir}/predicted-vs-true.html", auto_open=auto_open)
