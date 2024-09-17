import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
import matplotlib.pyplot as plt
from vessel_manoeuvring_models.substitute_dynamic_symbols import get_function_subs
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from vessel_manoeuvring_models.angles import smallest_signed_angle
from vessel_manoeuvring_models.substitute_dynamic_symbols import run

def predict(model: ModularVesselSimulator, data: pd.DataFrame, main_equation_excludes=[], VCT=True) -> pd.DataFrame:
    """

    Args:
        model (ModularVesselSimulator): _description_
        data (pd.DataFrame): _description_
        main_equation_excludes (list, optional): [''X_W', 'Y_W', 'N_W'] will exclude the wind system from the main equation . Defaults to [].

    Returns:
        pd.DataFrame: _description_
    """
    
    df_force_predicted = pd.DataFrame(
        model.calculate_forces(
            data[model.states_str], control=data[model.control_keys], main_equation_excludes=main_equation_excludes,
        ),
        index=data.index,
    )

    columns = list(set(data.columns) - set(df_force_predicted))
    df_force_predicted = pd.concat((data[columns], df_force_predicted), axis=1)

    if VCT:
        added_masses_SI = model.added_masses_SI.copy()
        
        df_force_predicted['X_VCT'] = run(model.lambda_VCT_X,inputs=df_force_predicted, **added_masses_SI)
        df_force_predicted['Y_VCT'] = run(model.lambda_VCT_Y,inputs=df_force_predicted, **added_masses_SI)
        df_force_predicted['N_VCT'] = run(model.lambda_VCT_N,inputs=df_force_predicted, **added_masses_SI)
        
    
    return df_force_predicted


def plot_total_force(model: ModularVesselSimulator, data: pd.DataFrame, window=None):
    if not isinstance(model, dict):
        models = {"Prediction": model}
    else:
        models = model

    model = models[list(models.keys())[0]]
    df_force = model.forces_from_motions(data=data)

    if window is None:
        df_force_plot = df_force
    else:
        df_force_plot = (
            df_force.select_dtypes(exclude="object").rolling(window=window).mean()
        )

    fig, axes = plt.subplots(nrows=3)
    ax = axes[0]
    ax.set_title("X force")
    df_force_plot.plot(y="fx", label="Experiment", ax=ax)
    ax.set_xticklabels([])
    ax.set_xlabel("")

    ax = axes[1]
    ax.set_title("Y force")

    df_force_plot.plot(y="fy", label="Experiment", ax=ax)
    ax.set_xticklabels([])
    ax.set_xlabel("")

    ax = axes[2]
    ax.set_title("N Moment")

    df_force_plot.plot(y="mz", label="Experiment", ax=ax)

    for name, model in models.items():
        df_force_predicted = pd.DataFrame(
            model.calculate_forces(
                data[model.states_str], control=data[model.control_keys]
            )
        )
        df_force_predicted["fx"] = df_force_predicted["X_D"]
        df_force_predicted["fy"] = df_force_predicted["Y_D"]
        df_force_predicted["mz"] = df_force_predicted["N_D"]

        if window is None:
            df_force_predicted_plot = df_force_predicted
        else:
            df_force_predicted_plot = (
                df_force_predicted.select_dtypes(exclude="object")
                .rolling(window=window)
                .mean()
            )

        ax = axes[0]
        ax.set_title("X force")
        df_force_predicted_plot.plot(y="fx", label=name, ax=ax)
        ax.set_xticklabels([])
        ax.set_xlabel("")

        ax = axes[1]
        ax.set_title("Y force")
        df_force_predicted_plot.plot(y="fy", label=name, ax=ax)
        ax.set_xticklabels([])
        ax.set_xlabel("")

        ax = axes[2]
        ax.set_title("N Moment")
        df_force_predicted_plot.plot(y="mz", label=name, ax=ax)

    for ax in axes:
        make_y_axis_symmetrical(ax)

    return fig


def make_y_axis_symmetrical(ax):
    ylims = ax.get_ylim()
    ylim = np.max(np.abs(ylims))
    ax.set_ylim(-ylim, ylim)


def plot_force_components(model, data, window=None):
    df_force_predicted = pd.DataFrame(
        model.calculate_forces(data[model.states_str], control=data[model.control_keys])
    )

    if window is None:
        df_plot = df_force_predicted
    else:
        df_plot = df_force_predicted.rolling(window=window).mean()

    keys = [arg.name for arg in model.X_D_eq.rhs.args]
    fig, axes = plt.subplots(nrows=3)

    ax = axes[0]
    ax.set_title("X forces")
    for key in keys:
        df_plot.plot(y=key, ax=ax)

        ax.set_xlabel("time [s]")
    ax.set_xticklabels([])
    ax.set_xlabel("")

    keys = [arg.name for arg in model.Y_D_eq.rhs.args]
    ax = axes[1]
    ax.set_title("Y forces")

    for key in keys:
        df_plot.plot(y=key, ax=ax)
        ax.set_xlabel("time [s]")
    ax.set_xticklabels([])
    ax.set_xlabel("")

    keys = [arg.name for arg in model.N_D_eq.rhs.args]
    ax = axes[2]
    ax.set_title("N moments")

    for key in keys:
        df_plot.plot(y=key, ax=ax)
        ax.set_xlabel("time [s]")

    for ax in axes:
        make_y_axis_symmetrical(ax)

    return fig


def get_delta_corners(data, limit_factor=0.95):
    mask = (
        (data["delta"] > limit_factor * data["delta"].max())
        | (data["delta"] < limit_factor * data["delta"].min())
    ).values
    mask_start = ~mask[0:-1] & mask[1:]
    mask_start = np.concatenate(([False], mask_start))
    starts = data.loc[mask_start]

    mask_end = ~mask[1:] & mask[0:-1]
    mask_end = np.concatenate((mask_end,[False]))

    ends = data.loc[mask_end]
    
    delta0 = data.iloc[0]['delta']
    start_time = ((data.loc[-1:starts.index[0]]['delta'] - delta0).abs() > (1-limit_factor) * data["delta"].max()).idxmax()
    i = data.index.get_loc(start_time)
    start_time=data.index[i-1]
    start = data.loc[[start_time]]
    
    corners = pd.concat((start, starts, ends), axis=0).sort_index()
    return starts, ends, corners


def plot_compare_model_forces(
    model: ModularVesselSimulator,
    models: dict,
    data: pd.DataFrame,
    keys=["N_D", "N_H", "N_R"],
    styles= {
            "Experiment": {
                "style": "-",
                "color": "green",
                "zorder": -10,
                "lw": 1.0,
                "label": "Experiment",
            },
    },
    delta_corners=True,
):
       
    
    if len(styles) == 1:
        for name,model in models.items():
            styles[name] = {"style": "b-", "lw": 0.5, "label": name}
        

    height_ratios=np.ones(len(keys)+1)
    height_ratios[0]=0.5                          
    fig, axes = plt.subplots(nrows=len(keys) + 1, height_ratios=height_ratios, constrained_layout = True)
    #fig.set_size_inches(13, 13)
    #model = models[list(models.keys())[0]]
    forces_from_motions = model.forces_from_motions(data=data)
    force_predictions = {name:predict(model=model, data=data) for name, model in models.items()}
    
    ax = axes[0]

    

    style='--'
    data.plot(y="beta", style=style,color="red", label=r'$\beta$', ax=ax)
    data.plot(y="delta",  style=style,color="g", label=r'$\delta$', ax=ax)
    ax.set_xlim(data.index[0], data.index[-1])
    ax.legend(loc="upper left")
    
    if delta_corners:
        starts, ends, corners = get_delta_corners(data=data)
        ax.set_xticks(corners.index)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid(False)

    ax2 = ax.twinx()
    data.plot(y="r",  style=style,color="b", label=r'$r$', ax=ax2)
    ax2.legend(loc="lower left")
    ax2.grid(False)
    ax2.set_yticklabels([])

    units={
        'X':'N',
        'Y':'N',
        'N':'Nm',
    }
    
    for key, ax in zip(keys, axes[1:]):
        if key in forces_from_motions:
            forces_from_motions.plot(y=key, **styles["Experiment"], ax=ax)

        for name, forces_predicted in force_predictions.items():
            forces_predicted.plot(y=key, **styles[name], ax=ax)
        
        unit = units.get(key[0],'-')
        ax.set_ylabel(rf"${key}$ [{unit}]")
        ax.get_legend().set_visible(False)
        ax.get_legend().set_visible(False)
        ax.set_xlim(data.index[0], data.index[-1])
        
        if delta_corners:
            ax.set_xticks(corners.index)
            ax.set_xticklabels([])

    for ax in axes[0:-1]:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    
    axes[1].legend()
    axes[-1].set_xlabel("Time [s]")
        
    #axes[-1].set_xticklabels(np.round(axes[-1].get_xticks(), 0))
    #axes[-1].set_xticklabels([])
    #axes[-1].set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    ylims=[]
    for ax in axes[1:]:
        ylims.append(ax.get_ylim())
    ylims = (np.min(np.min(ylims)),np.max(np.max(ylims)))
    for ax in axes[1:]:
        ax.set_ylim(ylims)
    
    for ax in axes:
        ax.grid(True)
    
    plt.tight_layout()
    
    return fig

def group_parameters(df:pd.DataFrame, joins=['Nv','Nr','Nvr']):
    
    groups = {}
    for join in joins:
        join_set = set(join)
        columns = []
        for key in df.columns:
            if set(key)==join_set:
               columns.append(key) 
    
        if len(columns) > 0:
            groups[join] = columns
        
    return groups

def joined_parameter_contributions(model, data:pd.DataFrame, ax=None, prefix='N', unit='moment'):

    
    hull = model.subsystems['hull']
    df_parameters_contributions = hull.calculate_parameter_contributions(eq=hull.equations[f'{prefix}_H'], data=data, unit=unit)
    
    groupers = ["v","r","vr"]
    joins = [f"{prefix}{group}" for group in groupers]
    groups = group_parameters(df=df_parameters_contributions, joins=joins)

    df_parameters_contributions_clean = pd.DataFrame(index=df_parameters_contributions.index)
    for name, group in groups.items():
        label = group[0] + "".join([f"+{item}" for item in group[1:]])
        df_parameters_contributions_clean[label] = df_parameters_contributions[group].sum(axis=1)
    
    return df_parameters_contributions_clean

def plot_parameter_contributions(model, data:pd.DataFrame, ax=None, prefix='N', unit='moment'):

    if ax is None:
        fig,ax=plt.subplots()
    
    hull = model.subsystems['hull']
    df_parameters_contributions = hull.calculate_parameter_contributions(eq=hull.equations[f'{prefix}_H'], data=data, unit=unit)
    
    groupers = ["v","r","vr"]
    joins = [f"{prefix}{group}" for group in groupers]
    groups = group_parameters(df=df_parameters_contributions, joins=joins)

    df_parameters_contributions_clean = pd.DataFrame(index=df_parameters_contributions.index)
    for name, group in groups.items():
        label = group[0] + "".join([f"+{item}" for item in group[1:]])
        df_parameters_contributions_clean[label] = df_parameters_contributions[group].sum(axis=1)
        
    SI_units = {
        'moment':'Nm',
        'force':'N'
    }
    
    ax.set_ylabel(f"${prefix}_H$ $[{SI_units.get(unit,unit)}]$")
    
    return df_parameters_contributions_clean.plot(ax=ax)

def same_ylims(axes):
    ylims = []
    for ax in axes:
        ylims.append(ax.get_ylim())
    ylims = np.array(ylims)
    new_ylims = (np.min(ylims[:,0]),np.max(ylims[:,1]),)
    for ax in axes:
        ax.set_ylim(new_ylims)
        
def preprocess(data):

    # Cut from the start:
    starts, ends, corners = get_delta_corners(data)
    data = data.loc[corners.index[0]:].copy()
    data.index-=data.index[0]
    data.index.name='time'
    
    delta0 = (data['delta'].max() + data['delta'].min())/2
    #data['delta']-=delta0
    data['-delta_deg'] = -np.rad2deg(data['delta'])
    
    psi0 = data.iloc[0]['psi'].copy()
    data['psi']-=psi0
    data['twa']-=psi0
    data['awa']=smallest_signed_angle(data['awa'])
    
    data['x0']-=data['x0'].iloc[0]
    data['y0']-=data['y0'].iloc[0]
    x0 = data['x0'].copy()
    y0 = data['y0'].copy()
    data['x0'] = x0*np.cos(psi0) + y0*np.sin(psi0)
    data['y0'] = -x0*np.sin(psi0) + y0*np.cos(psi0)
    
    data['psi_deg'] = np.rad2deg(smallest_signed_angle(data['psi']))

    return data