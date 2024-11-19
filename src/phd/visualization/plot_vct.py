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


def flat_keys(y_keys):
    flat = []
    for key in y_keys:
        if isinstance(key,str):
            flat.append(key)
        else:
            for sub_key in key:
                flat.append(sub_key)
    
    return flat
    

def plot_VCT(df_VCT:pd.DataFrame, predictions={}, test_type='Drift angle', y_keys=['X_D','Y_D','N_D'], colors={},prime=False, styles={}):
    
    data_VCT = df_VCT.groupby(by='test type').get_group(test_type)
    
    if isinstance(y_keys,str):
        y_keys=[y_keys]
    
    n_rows = len(y_keys)
    Vs = data_VCT['V'].unique()
    n_cols = len(Vs)     
    
    fig,axes = plt.subplots(ncols=n_cols, nrows=n_rows)
    if (n_cols==1) and (n_rows==1):
        axes=np.array([axes])
    
    axes = axes.reshape((n_rows,n_cols))
    
    axes_map={}
    
    # Create a dict of axes:
    for row,y_key in enumerate(y_keys):
        
        if isinstance(y_key,str): 
            axes_map[y_key] = {}
        else:
            for sub_key in y_key:
                axes_map[sub_key] = {}
        
        for col,V in enumerate(Vs):
            if isinstance(y_key,str):
                axes_map[y_key][V] = axes[row,col]
                
            else:
                # y_key can also be a list of keys that should end up on the same axis
                for sub_key in y_key:
                    axes_map[sub_key][V] = axes[row,col]
                    
    #y_keys = flat_keys(y_keys)
    
    def plot_dataset(data, label:str, style='.--'):
        for V, group in data.groupby(by='V'):
            first_row=True
            for y_key in y_keys:
                
                keys = []           
                if isinstance(y_key,str):
                    if not y_key in data:
                        continue
                
                    ax = axes_map[y_key][V]
                    #color = colors.get(y_key,'b')
                    plot_group(df=group, dof=y_key, ax=ax, style=style, label=label, prime=prime)
                    ax.set_ylabel(y_key)
                    keys.append(y_key)
                else:
                    for sub_key in y_key:
                        
                        if not sub_key in data:
                            continue
                
                        ax = axes_map[sub_key][V]
                        #color = colors.get(sub_key,'b')
                        plot_group(df=group, dof=sub_key, ax=ax, style=style, label=f"{sub_key} {label}", prime=prime)
                        keys.append(sub_key)
                    
                    if len(keys) > 0:
                        ax.set_ylabel(",".join(keys))
                        
                
                if len(keys) > 0:
                    if first_row:
                        first_row=False
                        ax.set_title(f"V:{V:0.2f} [m/s]")  

                    ax.grid()
                
        if n_rows > 1:
            # Remove xlabels for all but the last row
            for ax in axes[:-1,:].flatten():
                ax.set_xlabel('')
    
    predictions['VCT'] = data_VCT
    predictions_unordered = predictions.copy()
    predictions={}
    predictions['VCT'] = predictions_unordered['VCT']
    predictions.update(predictions_unordered)
                
    for name, df_prediction in predictions.items():
        data_prediction = df_prediction.groupby(by='test type').get_group(test_type)
        style_all = styles.get(name,{})
        label=style_all.get('label',name)
        style = style_all.get('style','x-')
        plot_dataset(data=data_prediction, label=label, style=style)

    fig.suptitle(test_type)
    
    return fig

    

def plot_group(df, dof, ax, style, label, annotate=False, prime=False, **kwargs):
    test_type = df.iloc[0]["test type"]
    if test_type == "Rudder and drift angle":
        plot_rudder_drift(df, dof, ax, style, label, **kwargs)
        ax.get_legend().set_visible(False)
    elif test_type == "Circle + Drift":
        plot_circle_drift(df, dof, ax, style, label, **kwargs)
    elif test_type == "Circle + Drift + rudder angle":
        plot_circle_drift_rudder_angle(df, dof, ax, style, label, **kwargs)
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
        
def plot_circle_drift(df, dof, ax, style, label, **kwargs):
            
    x = "beta_deg"
    df_ = df.copy()
    df_['beta_deg'] = np.rad2deg(df_['beta'])
    
    groups = df_.groupby(by="r")
    colors = list(plt.cm.Wistia(np.linspace(0,1,len(groups))))
    
    the_style = style[1:]
    
    for r, group in groups:
        the_label = f"{label} r={r:0.2f} rad/s"        
        
        color = colors.pop(0)
        group.sort_values(by=x).plot(
            x=x, y=dof, ax=ax, style=the_style, label=the_label, color=color,**kwargs
        )

def plot_circle_drift_rudder_angle(df, dof, ax, style, label, **kwargs):
            
    x = "beta_deg"
    df_ = df.copy()
    df_['beta_deg'] = np.rad2deg(df_['beta'])
    
    groups = df_.groupby(by=["r",'delta'])
    colors = list(plt.cm.Wistia(np.linspace(0,1,len(groups))))
    
    the_style = style[1:]
    
    for (r,delta), group in groups:
        
        the_label = f"{label} r={r:0.2f} rad/s delta={np.rad2deg(delta):0.0f} deg"        
        
        color = colors.pop(0)
        group.sort_values(by=x).plot(
            x=x, y=dof, ax=ax, style=the_style, label=the_label, color=color,**kwargs
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
    df_["beta_deg"] = np.rad2deg(df_["beta"])
    df_["delta_deg"] = np.rad2deg(df_["delta"])
    
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
            "Rudder angle": "delta_deg",
            "Rudder angle resistance (no propeller)": "delta_deg",
            "Drift angle": "beta_deg",
            "Circle": "r",
            "Circle + rudder angle": "delta_deg",
            "Circle + Drift": "vr",
            "Thrust variation": "thrust",
        }
    
    xlabels = {
        "Fn" : r"$Fn$",
        "delta_deg" : r"$\delta$",
        "beta_deg" : r"$\beta$",
        "r" : r"$r$",
        "vr": r"$v \cdot r$",
        "thrust": r"$T$",
    }
    
    if prime:
        units = {key: "$[-]$" for key in xlabels.keys()}
    else:
        units = {
            "Fn" : "$[-]$",
            "delta_deg" : "$[deg]$",
            "beta_deg" : "$[deg]$",
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
