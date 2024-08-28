"""
This is a boilerplate pipeline 'filter_wPCC'
generated using Kedro 0.18.7
"""
"""
This is a boilerplate pipeline 'filter'
generated using Kedro 0.18.7
"""
from vessel_manoeuvring_models.symbols import *
import pandas as pd
import logging
import numpy as np
from vessel_manoeuvring_models.data.lowpass_filter import lowpass_filter
from numpy import cos as cos
from numpy import sin as sin
from vessel_manoeuvring_models.angles import smallest_signed_angle
from phd.helpers import identity_decorator
from phd.helpers import derivative
from vessel_manoeuvring_models.EKF_VMM_1d import ExtendedKalmanFilterVMMWith6Accelerometers
from vessel_manoeuvring_models.KF_multiple_sensors import FilterResult
from vessel_manoeuvring_models.EKF_VMM import ExtendedKalmanFilterVMM

log = logging.getLogger(__name__)
from pyfiglet import figlet_format

def create_kalman_filter(models:dict, model_name:str, SNR=1)->ExtendedKalmanFilterVMMWith6Accelerometers:
    
    model = models[model_name]()
    
    h1_ = sp.ImmutableDenseMatrix([x_0,y_0,psi,
                             ])
    X_1 = sp.MutableDenseMatrix([x_0,y_0,psi,u,v,r])  # state vector,
    H1_ = h1_.jacobian(X_1)    
    H1 = lambdify(H1_)()
    B = np.array([[]])  # No inputs
    
    max_x=0.05
    max_y=0.05
    max_psi=np.deg2rad(2.0)

    sigma_x = max_x/3
    sigma_y = max_y/3
    sigma_psi = max_psi/3
    
    R1 = np.diag([sigma_x**2, sigma_y**2, sigma_psi**2])
    
    max_u = 0.05
    max_v = 0.01
    max_r = 0.01
    sigma_u = max_u/3
    sigma_v = max_v/3
    sigma_r = max_r/3
    
    Q1 = 1/50*np.diag([0,0,0,sigma_u**2,sigma_v**2,sigma_r**2,])
    
    ekf = ExtendedKalmanFilterVMM(model=model, B=B, H=H1, Q=Q1, R=R1, input_columns=[], 
                              control_columns=['delta',
                                               'thrust_port', 
                                               'thrust_stbd',
                                               'phi',
                                               'theta',
                                               'g',
                                               'Hull/Acc/X1',
                                               'Hull/Acc/Y1',
                                               'Hull/Acc/Y2',
                                               'Hull/Acc/Z1',
                                               'Hull/Acc/Z2',
                                               'Hull/Acc/Z3',], 
                               X=X_1)
    
    return ekf


def create_kalman_filter_acc(models:dict, model_name:str, SNR=1)->ExtendedKalmanFilterVMMWith6Accelerometers:
    
    model = models[model_name]()
    
    bl_to_wl = model.ship_parameters['T']
    model.ship_parameters['point1'] = {
    'x_P': 1.625,
    'y_P': 0.025,
    'z_P': -0.564+bl_to_wl,
    }

    model.ship_parameters['point2'] = {
        'x_P': -1.9,
        'y_P': 0.43,
        'z_P': -0.564+bl_to_wl,
    }
    
    h2_ = sp.ImmutableDenseMatrix([x_0,y_0,psi,
                              #u,v,r,
                              u1d,v1d,r1d
                             ])
    X = sp.MutableDenseMatrix([x_0,y_0,psi,u,v,r,u1d,v1d,r1d])  # state vector,
    H2_ = h2_.jacobian(X)    
    H = lambdify(H2_)()

    B = np.array([[]])  # No inputs

    Q = 1/100*np.diag([0,0,0,0,0,0,0.1,0.1,0.005])
    R = np.diag([0.025, 0.025, 0.00305, 9.81, 9.81, 0.175])
    
    
    ekf = ExtendedKalmanFilterVMMWith6Accelerometers(model=model, B=B, H=H, Q=Q, R=R,
                         state_columns=['x0','y0','psi','u','v','r','u1d','v1d','r1d'], measurement_columns=['x0', 'y0', 'psi','u1d','v1d','r1d'], 
                         control_columns=['delta','delta1d','thrust_port','thrust_stbd','thrust_port1d','thrust_stbd1d','phi','theta','g',
                                          'Hull/Acc/X1',
                                          'Hull/Acc/Y1',
                                          'Hull/Acc/Y2',
                                          'Hull/Acc/Z1',
                                          'Hull/Acc/Z2',
                                          'Hull/Acc/Z3',

                                         ], input_columns=[],
                              )
    
    return ekf

def create_kalman_filter_old(models:dict, model_name:str, SNR:float)->ExtendedKalmanFilterVMMWith6Accelerometers:
    
    model = models[model_name]()
    
    bl_to_wl = model.ship_parameters['T']
    model.ship_parameters['point1'] = {
    'x_P': 1.625,
    'y_P': 0.025,
    'z_P': -0.564+bl_to_wl,
    }

    model.ship_parameters['point2'] = {
        'x_P': -1.9,
        'y_P': 0.43,
        'z_P': -0.564+bl_to_wl,
    }
    
    h2_ = sp.ImmutableDenseMatrix([x_0,y_0,psi,
                              #u,v,r,
                              u1d,v1d,r1d
                             ])
    X = sp.MutableDenseMatrix([x_0,y_0,psi,u,v,r,u1d,v1d,r1d])  # state vector,
    H2_ = h2_.jacobian(X)    
    H2 = lambdify(H2_)()

    B = np.array([[]])  # No inputs


    Q_sigma_x = 0.05
    Q_sigma_y = Q_sigma_x
    Q_sigma_psi = np.deg2rad(1)

    Q_sigma_u = 0.05
    Q_sigma_v = 0.01
    Q_sigma_r = np.deg2rad(1)

    g_=9.81
    Q_sigma_u1d = g_/10
    Q_sigma_v1d = Q_sigma_u1d
    Q_sigma_r1d = np.deg2rad(1)

    Q2_diagonal = [Q_sigma_x**2, Q_sigma_y**2, Q_sigma_psi**2,
                  Q_sigma_u, Q_sigma_v, Q_sigma_r,
                  Q_sigma_u1d, Q_sigma_v1d, Q_sigma_r1d,
                  ]


    Q2 = np.diag(Q2_diagonal)
    
    log.info(f"Signal noise ratio (SNR):{SNR}")
    diagonal_ = np.array(Q2_diagonal)/SNR
    diagonal = np.concatenate((diagonal_[0:3],diagonal_[6:9]))
    R2 = np.diag(diagonal)
    
    ekf = ExtendedKalmanFilterVMMWith6Accelerometers(model=model, B=B, H=H2, Q=Q2, R=R2,
                         state_columns=['x0','y0','psi','u','v','r','u1d','v1d','r1d'], measurement_columns=['x0', 'y0', 'psi','u1d','v1d','r1d'], 
                         control_columns=['delta','delta1d','thrust_port','thrust_stbd','thrust_port1d','thrust_stbd1d','phi','theta','g',
                                          'Hull/Acc/X1',
                                          'Hull/Acc/Y1',
                                          'Hull/Acc/Y2',
                                          'Hull/Acc/Z1',
                                          'Hull/Acc/Z2',
                                          'Hull/Acc/Z3',

                                         ], input_columns=[],
                              )
    
    return ekf  

def initial_state_many(
    datas: dict, ekf:ExtendedKalmanFilterVMMWith6Accelerometers
) -> np.ndarray:
    x0_many = {}
    for name, loader in datas.items():
        data = loader()
        x0 = initial_state(data=data, ekf=ekf)
        x0_many[name] = x0

    return x0_many


def initial_state(
    data: pd.DataFrame, ekf:ExtendedKalmanFilterVMMWith6Accelerometers
) -> np.ndarray:
    #x0 = data.iloc[0][ekf.state_columns]
    #x0 = data.iloc[0:50][ekf.state_columns].mean()
    
    x0_position = data.iloc[0][ekf.state_columns[0:3]]
    x0_velocity = data.iloc[0:50][ekf.state_columns[3:6]].mean()
    x0_acceleration = data.iloc[0:50][ekf.state_columns[6:]].mean()

    x0_velocity['v'] = 0
    x0_velocity['r'] = 0

    if 'u1d' in ekf.state_columns:
        x0_acceleration['u1d'] = 0
        x0_acceleration['v1d'] = 0
        x0_acceleration['r1d'] = 0

        x0 = pd.Series(np.concatenate((x0_position.values,x0_velocity.values,x0_acceleration.values)), index=ekf.state_columns)
    
    else:
        x0 = pd.Series(np.concatenate((x0_position.values,x0_velocity.values,)), index=ekf.state_columns)
    
    
    return {key: float(value) for key, value in x0.items()}

def filter_many(
    datas: dict,
    ekf: ExtendedKalmanFilterVMMWith6Accelerometers,
    x0: dict,
    skip: dict,
) -> dict:
    functions = {}
    
    skip = [str(name) for name in skip]

    for name, loader in datas.items():
        if name in skip:
            log.info(f"Skipping the filtering for: {name}")
            continue

        log.info(f"Filtering: {name}")
        functions[name] = filter_lazy(
            loader=loader,
            ekf=ekf,
            x0=x0[name],
            name=name
        )
    return functions

def filter_lazy(
    loader,
    ekf,
    x0: list,
    name:str
):
    def wrapper():
        try:
            return filter(
            loader=loader,
            ekf=ekf,
            x0=x0(),
            )
        except Exception as e:
            raise ValueError(f"Failed on: {name}")

    return wrapper

def filter(
    loader,
    ekf: ExtendedKalmanFilterVMMWith6Accelerometers,
    x0: dict,
) -> pd.DataFrame:
    data = loader()
    
    # Calculate time derivatives of control variables:
    data['delta1d_'] = derivative(data,'delta')
    rudder_rate = np.deg2rad(2.32)*np.sqrt(ekf.model.ship_parameters['scale_factor'])
    data['delta1d'] = np.round(data['delta1d_']/rudder_rate,0)*rudder_rate

    dt = np.mean(pd.Series(data.index).diff()[1:])
    fs = 1/dt
    data['thrust_port_filtered'] = lowpass_filter(data['thrust_port'], cutoff=10, fs=fs, order=1)
    data['thrust_stbd_filtered'] = lowpass_filter(data['thrust_stbd'], cutoff=10, fs=fs, order=1)
    data['thrust_port1d'] = derivative(data,'thrust_port_filtered')
    data['thrust_stbd1d'] = derivative(data,'thrust_stbd_filtered')
    
    P_0 = np.zeros((ekf.n,ekf.n))
    x0 = pd.Series(x0)[ekf.state_columns].values.reshape(ekf.n,1)
    result = ekf.filter(data=data, P_0=P_0, x0=x0)
    
    return result

def results_to_dataframe(filtered_results:dict)-> dict:
    dataframes={}
    for name, loader in filtered_results.items():
        result = loader()
        dataframes[name] = result.df
        
    return dataframes

def smoother(ekf:ExtendedKalmanFilterVMMWith6Accelerometers, filtered_results:dict)-> dict:
    
    smoothened_results = {}
    for name, loader in filtered_results.items():
        smoothened_results[name] = smoother_lazy(loader=loader, ekf=ekf)
        
    return smoothened_results
    
def smoother_lazy(
    loader,
    ekf,
):
    def wrapper():
        results = loader()
        results_smoother =  ekf.smoother(results=results)
        #df_smoothened = results_smoother.df
        #return df_smoothened
        return results_smoother

    return wrapper

def lowpass(df: pd.DataFrame, cutoff: float = 1.0, order=1) -> pd.DataFrame:
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


def join_tests(tests_ek_smooth: dict, exclude=[]) -> pd.DataFrame:
    _ = []
    log.info(
        f"Creating a joined dataset, that can be used in inverse dynamics regression"
    )
    exclude = list([str(key) for key in exclude])
    for key, loader in tests_ek_smooth.items():
        if key in exclude:
            log.info(f"Excluding {key} from the joined dataset")
            continue

        log.info(f"Adding: {key}")

        df_ = loader()
        df_["id"] = key
        if len(_) > 0:
            dt = df_.index[1] - df_.index[0]
            previous = _[-1]
            df_.index += dt + previous.index[-1]

        _.append(df_)

    df_joined = pd.concat(_, axis=0)
    return df_joined
