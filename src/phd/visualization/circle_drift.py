import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .plot_prediction import predict
from matplotlib import cm
import matplotlib.pyplot as plt

def plot_3d(
    data,
    data_prediction=None,
    key="mz_hull",
):

    layout = go.Layout(
        scene={
            "xaxis": {"title": "beta"},
            "yaxis": {"title": "r"},
            "zaxis": {"title": key},
        }
    )

    if data_prediction is None:
        fig = go.Figure(layout=layout)
    else:
        N = int(np.sqrt(len(data_prediction)))
        fig = go.Figure(
            data=[
                go.Surface(
                    z=data_prediction[key].values.reshape((N, N)),
                    x=data_prediction["beta"].values.reshape((N, N)),
                    y=data_prediction["r"].values.reshape((N, N)),
                )
            ],
            layout=layout,
        )

    if key in data:
        trace = go.Scatter3d(
            x=data["beta"],
            y=data["r"],
            z=data[key],
            mode="markers",
            marker={
                "color": "black",
                "size": 3,
            },
        )
        fig.add_trace(trace)

    fig.update_layout(
        title=key,
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(l=65, r=50, b=65, t=90),
    )

    fig.show()


def predict_circle_drift_matrix(df: pd.DataFrame, model, N=20):

    betas = np.linspace(df["beta"].min(), df["beta"].max(), N)
    rs = np.linspace(df["r"].min(), df["r"].max(), N)
    Betas, Rs = np.meshgrid(betas, rs)
    data_ = pd.DataFrame(np.tile(df.iloc[0], (N * N, 1)), columns=df.columns)
    data_["r"] = Rs.reshape((len(data_), 1))
    data_["beta"] = Betas.reshape((len(data_), 1))
    data_["u"] = data_["V"] * np.cos(data_["beta"])
    data_["v"] = -data_["V"] * np.sin(data_["beta"])
    for key in df:
        data_[key] = data_[key].astype(df[key].dtype)

    result_ = predict(model=model, data=data_)
    return result_


def plot_3d_circle_drift(
    df: pd.DataFrame,
    model,
    key="Y_H",
):
    df_circle_drift_prediction = predict_circle_drift_matrix(df=df, model=model)
    plot_3d(data=df, data_prediction=df_circle_drift_prediction, key=key)

def plot_3d_circle_drift_matplotlib(
    df: pd.DataFrame,
    model,
    key="Y_H",
    elev=25, 
    azim=180+45, 
    roll=0
    ):
    
    df = df.copy()
    
    result = predict_circle_drift_matrix(df=df, model=model)
    result['beta_deg'] = np.rad2deg(result['beta'])
    df['beta_deg'] = np.rad2deg(df['beta'])
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    N = int(np.sqrt(len(result)))
    X = result['beta_deg'].values.reshape(N,N)
    Y = result['r'].values.reshape(N,N)
    Z = result[key].values.reshape(N,N)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.75)
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.scatter(df['beta_deg'],df['r'],df[key], marker='.', color='k', s=0.75)

    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_xlabel(r'$\beta$ [deg]')
    ax.set_ylabel(r'$r$ [rad/s]')
    ax.set_zlabel(fr'${key}$',labelpad=-4, rotation=0)
    ax.xaxis.set_rotate_label(False)  # disable automatic rotation (so that I can actually subsequently change it)
    ax.yaxis.set_rotate_label(False)  # disable automatic rotation (so that I can actually subsequently change it)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation (so that I can actually subsequently change it)
    
    return fig