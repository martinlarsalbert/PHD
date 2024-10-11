import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path
from collections import OrderedDict


def plot_compare(
    df_results: pd.DataFrame,
    datasets: {},
    test_type="Drift angle",
    y_keys=["X_D", "Y_D", "N_D"],
    y_labels=[r"$X_D'$ $[-]$", r"$Y_D'$ $[-]$", r"$N_D'$ $[-]$"],
    legend=True,
):
    df_ = df_results.groupby(by="test type").get_group(test_type)

    if test_type == "resistance":
        ncols = 1
    else:
        ncols = int(np.ceil(len(df_["Fn"].unique())))

    nrows = len(y_keys)

    if legend:
        fig, axes, ax_legend = subplot_with_legend(ncols=ncols, nrows=nrows)
    else:
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows)

    axes = np.array(axes).reshape(nrows, ncols)

    by_label = {}
    for row, (y_key, y_label) in enumerate(zip(y_keys, y_labels)):
        if test_type == "resistance":
            resistance_plot(
                axes=axes,
                row=row,
                datasets=datasets,
                test_type=test_type,
                y_key=y_key,
                df_=df_,
                df_results=df_results,
                by_label=by_label,
            )
        else:
            V_plots(
                axes=axes,
                row=row,
                datasets=datasets,
                test_type=test_type,
                y_key=y_key,
                df_=df_,
                df_results=df_results,
                by_label=by_label,
            )

        if axes.ndim == 2:
            axes[row, 0].set_ylabel(y_label)
        else:
            axes[0].set_ylabel(y_label)

    for ax_row in axes[0:-1, :]:
        for ax in ax_row:
            ax.set_xlabel("")
            ax.set_xticklabels([])

    for ax_row in axes[1:, :]:
        for ax in ax_row:
            ax.set_title("")

    for ax_col in axes[:, 1:]:
        for ax in ax_col:
            ax.set_yticklabels([])

    # fig.suptitle(f"{test_type}")

    # fig.legend(
    #    by_label.values(),
    #    by_label.keys(),
    #    bbox_to_anchor=(1, 1.35),
    #    ncols=2,
    #    fontsize=9,
    # )
    if legend:
        ax_legend.legend(
            by_label.values(),
            by_label.keys(),
            # bbox_to_anchor=(1, 1.35),
            ncols=3,
            fontsize=10,
        )

    # fig.tight_layout()
    # fig.subplots_adjust(right=right)

    return fig


def V_plots(axes, row, datasets, test_type, y_key, df_, df_results, by_label):
    for col, (V, df_V) in enumerate(df_.groupby(by="Fn")):
        ax = axes[row, col]
        ## VCT:
        ms = ["x", "+", "o", "^", "*", "s", "v"]
        colors = ["k", "r", "b", "g", "m", "c", "y"]
        colors += colors
        markers = []
        for i in range(int(len(colors) / 2)):
            c = colors[i : i + len(colors)]
            markers += [ms[j] + c[j] for j in range(len(ms))]
        for dataset_name, dataset in datasets.items():
            vct_groups = dataset.groupby(by=["test type", "Fn"])
            if len(markers) > 1:
                marker = markers.pop(0)
            else:
                marker = markers[0]
            try:
                df_vct = vct_groups.get_group((test_type, V))
            except:
                pass
            else:
                plot_group(
                    df=df_vct,
                    dof=y_key,
                    ax=ax,
                    style=marker,
                    label=dataset_name,
                    zorder=10,
                    ms=10,
                    mfc="none",
                )
        ## Regression

        for name, df__ in df_V.groupby(by="model_name"):
            style = df__.iloc[0]["style"]
            plot_group(df=df__, dof=y_key, ax=ax, style=style, label=name)
        ax.set_title(f"Fn={np.round(V,2)} [-]")
        ax.grid(True)
        y_mins = [dataset[y_key].min() for dataset in datasets.values()]
        y_mins.append(df_results[y_key].min())
        y_maxs = [dataset[y_key].max() for dataset in datasets.values()]
        y_maxs.append(df_results[y_key].max())
        y_min = np.min(y_mins)
        ylims = (y_min - 0.01 * y_min, np.max(y_maxs))
        ax.set_ylim(ylims)
        # try:
        #    ax.set_ylim(ylims)
        # except Exception as e:
        #    pass
        ax.get_legend().set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        by_label.update(zip(labels, handles))


def subplot_with_legend(nrows=3, ncols=3, first_height_ratio=0.30):
    fig = plt.figure()

    first_height_ratio = 0.3
    others = (1 - first_height_ratio) / nrows
    height_ratios = np.concatenate(([first_height_ratio], others * np.ones(nrows)))

    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=height_ratios)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("None")
    ax1.grid(False)
    ax1.axis("off")

    axess = []
    for row in range(nrows):
        axess_row = []
        for col in range(ncols):
            ax = fig.add_subplot(gs[row + 1, col])
            axess_row.append(ax)

        axess.append(axess_row)

    axess = np.array(axess)
    return fig, axess, ax1


def resistance_plot(axes, row, datasets, test_type, y_key, df_, df_results, by_label):
    ax = axes[row, 0]
    ## VCT:
    ms = ["x", "+", "o", "^", "*", "s", "v"]
    colors = ["k", "r", "b", "g", "m", "c", "y"]
    colors += colors
    markers = []
    for i in range(int(len(colors) / 2)):
        c = colors[i : i + len(colors)]
        markers += [ms[j] + c[j] for j in range(len(ms))]
    for dataset_name, dataset in datasets.items():
        if len(markers) > 1:
            marker = markers.pop(0)
        else:
            marker = markers[0]

        df_vct = dataset.groupby(by="test type").get_group(test_type)
        plot_group(
            df=df_vct,
            dof=y_key,
            ax=ax,
            style=marker,
            label=dataset_name,
            zorder=10,
            ms=10,
            mfc="none",
        )
    ## Regression
    colors = [
        "r",
        "g",
        "b",
        "k",
        "m",
        "c",
        "y",
        "m",
        "r",
        "g",
        "b",
        "k",
        "m",
        "c",
        "y",
        "m",
    ]
    for name, df__ in df_.groupby(by=["model_name"]):
        if len(colors) > 1:
            color = colors.pop(0)
        else:
            color = colors[0]
        plot_group(df=df__, dof=y_key, ax=ax, style=f"{color}-", label=name)
    ax.grid(True)
    y_mins = [dataset[y_key].min() for dataset in datasets.values()]
    y_mins.append(df_results[y_key].min())
    y_maxs = [dataset[y_key].max() for dataset in datasets.values()]
    y_maxs.append(df_results[y_key].max())
    y_min = np.min(y_mins)
    ylims = (y_min - 0.01 * y_min, np.max(y_maxs))
    ax.set_ylim(ylims)
    # try:
    #    ax.set_ylim(ylims)
    # except Exception as e:
    #    pass
    ax.get_legend().set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    by_label.update(zip(labels, handles))


def plot_VCT(df_VCT:pd.DataFrame, df_prediction:pd.DataFrame=None, test_type='Drift angle', y_keys=['X_D','Y_D','N_D'], prime=False):
    
    data_VCT = df_VCT.groupby(by='test type').get_group(test_type)
    
    n_rows = len(y_keys)
    Vs = data_VCT['V'].unique()
    n_cols = len(Vs)     
    
    fig,axes = plt.subplots(ncols=n_cols, nrows=n_rows)
    if (n_cols==0) and (n_rows==0):
        axes=np.array([axes])
    
    axes = axes.reshape((n_rows,n_cols))
    
    axes_map={}
    
    for row,y_key in enumerate(y_keys):
        axes_map[y_key] = {}
        for col,V in enumerate(Vs):
            axes_map[y_key][V] = axes[row,col]
            

    def plot_dataset(data, label:str, style='.--'):
        for V, group in data.groupby(by='V'):
            first_row=True
            for y_key in y_keys:
                ax = axes_map[y_key][V]
                plot_group(df=group, dof=y_key, ax=ax, style=style, label=label, prime=prime)

                if first_row:
                    first_row=False
                    ax.set_title(f"V:{V:0.2f} [m/s]")  

                ax.grid()
                ax.set_ylabel(y_key)

        if n_rows > 1:
            # Remove xlabels for all but the last row
            for ax in axes[:-1,:].flatten():
                ax.set_xlabel('')
                
    plot_dataset(data=data_VCT, label='VCT')
    
    if not df_prediction is None:
        data_prediction = df_prediction.groupby(by='test type').get_group(test_type)
        plot_dataset(data=data_prediction, label='prediction', style='-')

    fig.suptitle(test_type)
    
    return fig

    

def plot_group(df, dof, ax, style, label, annotate=False, prime=False, **kwargs):
    test_type = df.iloc[0]["test type"]
    if test_type == "Rudder and drift angle":
        plot_rudder_drift(df, dof, ax, style, label, **kwargs)
        ax.get_legend().set_visible(False)
    elif (test_type == "Thrust variation") and (dof != "fx"):
        plot_thrust_variation(df, dof, ax, style, label, **kwargs)
        # ax.get_legend().set_visible(False)
    else:
        plot_standard(df, dof, ax, style, label, annotate, prime=prime, **kwargs)


def plot_rudder_drift(df, dof, ax, style, label, **kwargs):
    x = "delta"
    df_ = df.copy()
    df_[r"$\delta$ $[deg]$"] = np.rad2deg(df_["delta"])
    for beta, group in df_.groupby(by=["beta"]):
        group.sort_values(by=x).plot(
            x=r"$\delta$ $[deg]$", y=dof, ax=ax, style=style, label=label, **kwargs
        )


def plot_thrust_variation(df, dof, ax, style, label, **kwargs):
    x = "thrust"
    df = df.copy()
    df["delta"] = df["delta"].round(decimals=2)
    for beta, group in df.groupby(by=["delta"]):
        group.sort_values(by=x).plot(
            x=x, y=dof, ax=ax, style=style, label=label, **kwargs
        )


def plot_standard(df, dof, ax, style, label, annotate=False, prime=False, **kwargs):
    df_ = df.copy()
    
    df_["vr"] = df_["v"] * df_["r"]
    
    
    #if prime:
    #    xs = {
    #        "resistance": r"$Fn$ $[-]$",
    #        "Rudder angle": r"$\delta$ $[deg]$",
    #        "Rudder angle resistance (no propeller)": r"$\delta$ $[deg]$",
    #        "Drift angle": r"$\beta$ $[deg]$",
    #        "Circle": r"$r'$ $[-]$",
    #        "Circle + rudder angle": r"$\delta$ $[deg]$",
    #        "Circle + Drift": r"$v' \cdot r'$ $[-]$",
    #        "Thrust variation": r"$T'$ $[-]$",
    #    }
    
    xs = {
            "resistance": "Fn",
            "Rudder angle": "delta",
            "Rudder angle resistance (no propeller)": "delta",
            "Drift angle": "beta",
            "Circle": "r",
            "Circle + rudder angle": "delta",
            "Circle + Drift": "vr",
            "Thrust variation": "thrust",
        }
    
    xlabels = {
        "Fn" : r"$Fn$",
        "delta" : r"$\delta$",
        "beta" : r"$\beta$",
        "r" : r"$r$",
        "vr": r"$v \cdot r$",
        "thrust": r"$T$",
    }
    
    if prime:
        units = {key: "$[-]$" for key in xlabels.keys()}
    else:
        units = {
            "Fn" : "$[-]$",
            "delta" : "$[deg]$",
            "beta" : "$[deg]$",
            "r" : r"$[rad/s]$",
            "vr": "$[-]$",
            "thrust": "$[N]$",
        }
    
    df_["index"] = df_.index
    test_type = df.iloc[0]["test type"]
    x = xs.get(test_type, "index")

    df_.sort_values(by=x).plot(x=x, y=dof, ax=ax, style=style, label=label, **kwargs)

    xlabel=f"{xlabels[x]} {units[x]}"
    ax.set_xlabel(xlabel)

    if annotate:
        for index, row in df_.iterrows():
            ax.annotate(" %s" % row["name"], xy=(row[x], row[dof]))
