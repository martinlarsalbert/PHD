"""
This is a boilerplate pipeline 'load_wPCC'
generated using Kedro 0.18.7
"""
import pandas as pd
import matplotlib.pyplot as plt

from numpy import cos as cos
from numpy import sin as sin
import numpy as np
from phd.helpers import identity_decorator

# from src.data.lowpass_filter import lowpass_filter


from vessel_manoeuvring_models.symbols import *
from phd.pipelines.load_7m.reference_frames import O_B, B, I, O, subs
from sympy.physics.vector import ReferenceFrame, Point
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from phd.helpers import derivative

## Equations:
P = Point("P")  # New origo
P.set_pos(O_B, -x_G * B.x)
position = P.pos_from(O).express(I).to_matrix(I)
velocity = P.vel(I).express(B).to_matrix(B)
acceleration = P.acc(I).express(B).to_matrix(B)

## Lambdas:
eqs = position.subs(subs)
lambda_x0 = sp.lambdify(list(eqs[0].free_symbols), eqs[0], "numpy")
lambda_y0 = sp.lambdify(list(eqs[1].free_symbols), eqs[1], "numpy")

eqs = velocity.subs(subs)
lambda_u = sp.lambdify(list(eqs[0].free_symbols), eqs[0], "numpy")
lambda_v = sp.lambdify(list(eqs[1].free_symbols), eqs[1], "numpy")

eqs = acceleration.subs(subs)
lambda_u1d = sp.lambdify(list(eqs[0].free_symbols), eqs[0], "numpy")
lambda_v1d = sp.lambdify(list(eqs[1].free_symbols), eqs[1], "numpy")



def add_thrust(
    df: pd.DataFrame, thrust_channels: list, rev_channels: list
) -> pd.DataFrame:
    assert isinstance(thrust_channels, list), "'thrust_channels' should be a list"

    if len(thrust_channels) > 0:
        df["thrust"] = df[thrust_channels].sum(axis=1)

    if len(rev_channels) > 0:
        df["rev"] = df[rev_channels].abs().mean(axis=1)

    return df


def load(
    raw_data: dict,
    thrust_channels: list = [],
    rev_channels: list = [],
    replace_velocities=False,
):
    datas = {}
    for name, loader in raw_data.items():
        datas[name] = load_lazy(
            loader=loader,
            replace_velocities=replace_velocities,
            thrust_channels=thrust_channels,
            rev_channels=rev_channels,
        )  # This is lazy saving
        # datas[name] = _load(
        #    loader=loader,
        #    replace_velocities=replace_velocities,
        #    thrust_channels=thrust_channels,
        #    rev_channels=rev_channels,
        # )  # This is lazy saving

    return datas


def load_lazy(
    loader,
    replace_velocities=False,
    thrust_channels: list = [],
    rev_channels: list = [],
):
    def the_loader():
        return _load(
            loader=loader,
            replace_velocities=replace_velocities,
            thrust_channels=thrust_channels,
            rev_channels=rev_channels,
        )

    return the_loader


@identity_decorator
def _load(
    loader,
    replace_velocities=False,
    thrust_channels: list = [],
    rev_channels: list = [],
):
    data = loader()

    ## Zeroing:
    data.index -= data.index[0]
    data["x0"] -= data.iloc[0]["x0"]
    data["y0"] -= data.iloc[0]["y0"]
    # data["psi"] -= data.iloc[0]["psi"]

    ## estimating higher states with numerical differentiation:
    t = data.index

    data = motions(data, replace_velocities=replace_velocities)

    data = add_thrust(data, thrust_channels=thrust_channels, rev_channels=rev_channels)
    
    if 'Prop/PS/Thrust' in data:
        data['thrust_port'] = data['Prop/PS/Thrust']
        data['thrust_stbd'] = data['Prop/SB/Thrust']
    
    if 'Prop/1/Thrust' in data:
        data['thrust'] = data['Prop/1/Thrust']
    
    
    data['g'] = 9.81
    
    return data

def motions(data, replace_velocities=False)->pd.DataFrame:
    
    dxdt = derivative(data, "x0")
    dydt = derivative(data, "y0")
    psi = data["psi"]

    if not "u" in data or replace_velocities:
        data["u"] = dxdt * np.cos(psi) + dydt * np.sin(psi)

    if not "v" in data or replace_velocities:
        data["v"] = v = -dxdt * np.sin(psi) + dydt * np.cos(psi)

    if not "r" in data or replace_velocities:
        data["r"] = r = derivative(data, "psi")

    data["u1d"] = derivative(data, "u")
    data["v1d"] = derivative(data, "v")
    data["r1d"] = derivative(data, "r")

    data["V"] = data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    data["beta"] = -np.arctan2(data["v"], data["u"])
    
    data["twa"] = 0
    data["tws"] = 0

    data['phi'] = data['roll']
    data["p"] = derivative(data, "phi")
    data["p1d"] = derivative(data, "p")
    
    data['theta'] = data['pitch']
    data["q"] = derivative(data, "theta")
    data["q1d"] = derivative(data, "q")
    
    return data

def filter(df: pd.DataFrame, cutoff: float = 1.0, order=1) -> pd.DataFrame:
    """Lowpass filter and calculate velocities and accelerations with numeric differentiation

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    cutoff : float
        Cut off frequency of the lowpass filter [Hz]
    order : int, optional
        order of the filter, by default 1

    Returns
    -------
    pd.DataFrame
        lowpassfiltered positions, velocities and accelerations
    """

    df_lowpass = df.copy()
    t = df_lowpass.index
    ts = np.mean(np.diff(t))
    fs = 1 / ts

    position_keys = ["x0", "y0", "psi"]
    for key in position_keys:
        df_lowpass[key] = lowpass_filter(
            data=df_lowpass[key], fs=fs, cutoff=cutoff, order=order
        )

    df_lowpass["x01d_gradient"] = x1d_ = derivative(df_lowpass, "x0")
    df_lowpass["y01d_gradient"] = y1d_ = derivative(df_lowpass, "y0")
    df_lowpass["r"] = r_ = derivative(df_lowpass, "psi")

    psi_ = df_lowpass["psi"]

    df_lowpass["u"] = x1d_ * cos(psi_) + y1d_ * sin(psi_)
    df_lowpass["v"] = -x1d_ * sin(psi_) + y1d_ * cos(psi_)

    velocity_keys = ["u", "v", "r"]
    for key in velocity_keys:
        df_lowpass[key] = lowpass_filter(
            data=df_lowpass[key], fs=fs, cutoff=cutoff, order=order
        )

    df_lowpass["u1d"] = r_ = derivative(df_lowpass, "u")
    df_lowpass["v1d"] = r_ = derivative(df_lowpass, "v")
    df_lowpass["r1d"] = r_ = derivative(df_lowpass, "r")

    return df_lowpass


def move_to_lpp_half(data_loaders: dict, ship_data: dict) -> pd.DataFrame:
    datas = {}
    for name, loader in data_loaders.items():
        data = loader()
        datas[name] = _move_to_lpp_half(data=data, ship_data=ship_data)

    return datas


def _move_to_lpp_half(data: pd.DataFrame, ship_data: dict) -> pd.DataFrame:
    data = data.copy()

    if not "theta" in data:
        data["theta"] = 0
    if not "q" in data:
        data["q"] = 0
    if not "phi" in data:
        data["phi"] = data["roll"]
    if not "p" in data:
        data["p"] = 0
    if not "q1d" in data:
        data["q1d"] = 0

    data["x0"] = run(lambda_x0, inputs=data, x_G=ship_data["x_G"])
    data["y0"] = run(lambda_y0, inputs=data, x_G=ship_data["x_G"])
    data["u"] = run(lambda_u, inputs=data, x_G=ship_data["x_G"])
    data["v"] = run(lambda_v, inputs=data, x_G=ship_data["x_G"])
    data["u1d"] = run(lambda_u1d, inputs=data, x_G=ship_data["x_G"])
    data["v1d"] = run(lambda_v1d, inputs=data, x_G=ship_data["x_G"])
    return data


def preprocess(data_MDL, ship_data: dict):
    data_MDL["V"] = data_MDL["U"] = np.sqrt(data_MDL["u"] ** 2 + data_MDL["v"] ** 2)
    data_MDL["beta"] = -np.arctan2(data_MDL["v"], data_MDL["u"])
    
    if "Prop/PS/Rpm" in data_MDL and "Prop/SB/Rpm" in data_MDL:
        data_MDL["rev"] = data_MDL[["Prop/PS/Rpm", "Prop/SB/Rpm"]].mean(axis=1)
    
    data_MDL["twa"] = 0
    data_MDL["tws"] = 0
    data_MDL["theta"] = 0
    data_MDL["q"] = 0
    if "roll" in data_MDL:
        data_MDL["phi"] = data_MDL["roll"]
    
    data_MDL["p"] = 0
    data_MDL["q1d"] = 0
    
    if "Prop/PS/Thrust" in data_MDL:
        data_MDL["thrust_port"] = data_MDL["Prop/PS/Thrust"]
        data_MDL["thrust_stbd"] = data_MDL["Prop/SB/Thrust"]
        data_MDL['thrust'] = data_MDL["thrust_port"] + data_MDL["thrust_stbd"]
        
    if "thrust_port" in data_MDL:
        data_MDL['thrust'] = data_MDL["thrust_port"] + data_MDL["thrust_stbd"]
    
    data_MDL['psi_deg'] = np.rad2deg(data_MDL['psi'])
    data_MDL['beta_deg'] = np.rad2deg(data_MDL['beta'])
    data_MDL['delta_deg'] = np.rad2deg(data_MDL['delta'])
    

    # Remove the firs part:
    dt = pd.Series(data_MDL.index).diff().mean()
    rudder_rate = 2.32 * np.sqrt(ship_data["scale_factor"])
    start = (
        data_MDL["delta"].diff().abs() > 0.5 * np.deg2rad(rudder_rate) * dt
    ).idxmax()
    data_MDL.index = pd.Series(data_MDL.index) - start
    data_MDL = data_MDL.loc[0:].copy()

    return data_MDL

def zigzag_angle(data_MDL:pd.DataFrame):
    angle_abs = np.rad2deg(data_MDL['delta'].abs().max())
    angle_sign = np.sign(data_MDL.iloc[0:10]['delta'].mean())
    angle = np.round(angle_sign*angle_abs)
    return angle

def move_to_roll_centre(data_WL: dict,WL_to_roll_centre:float)->dict:
        
    datas = {}
    for name, loader in data_WL.items():
        datas[name] = load_lazy2(loader=loader, WL_to_roll_centre=WL_to_roll_centre)
        
    return datas

def load_lazy2(
    loader,
    WL_to_roll_centre,
):
    def the_loader():
        return _move_to_roll_centre(
            loader=loader,
            WL_to_roll_centre=WL_to_roll_centre,
        )

    return the_loader
        
def _move_to_roll_centre(loader,WL_to_roll_centre:float):
    
    data = loader()
    data['y0']-=WL_to_roll_centre*np.sin(data['phi'])
    data = motions(data, replace_velocities=True)
    
    return data
    
def meta_data(time_series_meta_data:pd.DataFrame, tests:dict)-> pd.DataFrame:
    
    test_meta_data = time_series_meta_data.copy()
    
    for id, row in time_series_meta_data.iterrows():
        
        if not str(id) in tests:
            continue
        
        data = tests[str(id)]()
        test_meta_data.loc[id,'angle'] = find_rudder_angle(data=data)

    test_meta_data['direction'] = test_meta_data['angle'].apply(lambda x: "port" if x > 0 else "stbd")
    
    mask = test_meta_data['test_type'] == 'zigzag'
    test_meta_data['loading condition'] = test_meta_data['name']
    test_meta_data.loc[mask,'name'] = test_meta_data.loc[mask].apply(lambda x: f"{x['test_type']} {x['angle']:0.0f}/{x['angle']:0.0f} {x['direction']}", axis=1) 
    
    return test_meta_data
    
def find_rudder_angle(data:pd.DataFrame):

    s = data.abs().max()
    
    i = (data['delta'].abs() > np.deg2rad(5)).idxmax()
    s['delta']*=np.sign(data.loc[i,'delta'])
    
    return np.round(np.rad2deg(s['delta']))
    