import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from phd.helpers import derivative
from vessel_manoeuvring_models.data.lowpass_filter import lowpass_filter
from phd.pipelines.load_wPCC.reference_frames import *
import phd.pipelines.load_wPCC.accelerometers6 as accelerometers6
from phd.pipelines.filter.nodes import lowpass
from numpy.linalg import det, inv

def load(id:int, catalog, smooth=False, tests="wPCC.tests", tests_ek="wPCC.tests_ek_smooth2", ):
    
    if smooth:
        data = catalog.load(tests_ek)[f'{id}']()
    else:
        data = catalog.load(tests)[f'{id}']()
        
    data.index.name='time'    
    
    if not 'phi' in data:
        data['phi'] = data['roll']
    
    
    data["p"] = derivative(data, "phi")
    data["p1d"] = derivative(data, "p")
    
    if not "theta" in data:
        data['theta'] = data['pitch']
    
    data["q"] = derivative(data, "theta")
    data["q1d"] = derivative(data, "q")

    if not 'thrust_port' in data:
        if 'Prop/PS/Thrust' in data:
            data['thrust_port'] = data['Prop/PS/Thrust']
    
    if not 'thrust_stbd' in data:
        if 'Prop/SB/Thrust' in data:
            data['thrust_stbd'] = data['Prop/SB/Thrust']
    
    data['g'] = 9.81
    
    

    return data

def filter(data,cutoff = 10,order= 1, smooth=False, 
        key_filter = [
            'Hull/Acc/X1',
            'Hull/Acc/Y1',
            'Hull/Acc/Y2',
            'Hull/Acc/Z1',
            'Hull/Acc/Z2',
            'Hull/Acc/Z3',
        ],
        derives = [
            ('phi','p','p1d'),
            ('theta','q','q1d'),    
        ],   
        
        ):

    if smooth:
        data_filtered = data.copy()
    else:
        data_filtered = lowpass(data, cutoff=cutoff, order=order, skip_samples=0)
        
    t = data.index
    ts = np.mean(np.diff(t))
    fs = 1 / ts
        
    for key in key_filter:
        data_filtered[key] = lowpass_filter(
            data=data[key], fs=fs, cutoff=cutoff, order=order
        )
    
    
    
    
    for derive in derives:
    
        data_filtered[derive[1]] = derivative(data_filtered, derive[0])
        data_filtered[derive[1]] = lowpass_filter(
                data=data_filtered[derive[1]], fs=fs, cutoff=cutoff, order=order
            )
        
        data_filtered[derive[2]] = derivative(data_filtered, derive[1])

    return data_filtered

def calculate_accelerometer_corrections(data_calc, simplified=True):

    data_corrected = data_calc.copy()

    data_corrected = calculated_accelerometer(data_corrected, simplified=simplified)
    
    keys = [
    'Hull/Acc/X1',
    'Hull/Acc/Y1',
    'Hull/Acc/Y2',
    'Hull/Acc/Z1',
    'Hull/Acc/Z2',
    'Hull/Acc/Z3',
    ]

    keys_pred = [f"{key}_pred" for key in keys]

    corrections = pd.Series((data_corrected[keys].values-data_corrected[keys_pred].values).mean(axis=0), index=keys)
    return corrections

bl_to_wl = 0.2063106796116504
point1 = {
'x_P': 1.625,
'y_P': 0.025,
'z_P': -0.564+bl_to_wl,
}

point2 = {
    'x_P': -1.9,
    'y_P': 0.43,
    'z_P': -0.564+bl_to_wl,
}

point3 = {
        'x_P': -1.9,
        'y_P': -0.43,
        'z_P': -0.564+bl_to_wl,
    }

def calculated_accelerometer(data_filtered, simplified=False):

   
    meta_data = {
    }
    if not 'g' in data_filtered:
        meta_data['g']=9.81
    
    
    data_calc = data_filtered.copy()
    if simplified:
        lambda_x = lambda_x2d_P_simplified
        lambda_y = lambda_y2d_P_simplified
        lambda_z = lambda_z2d_P_simplified
    else:
        lambda_x = lambda_x2d_P
        lambda_y = lambda_y2d_P
        lambda_z = lambda_z2d_P
    
    data_calc[r'Hull/Acc/X1_pred'] = lambda_x(**data_calc, **point1, **meta_data)
    data_calc[r'Hull/Acc/Y1_pred'] = lambda_y(**data_calc, **point1, **meta_data)
    data_calc[r'Hull/Acc/Z1_pred'] = lambda_z(**data_calc, **point1, **meta_data)
    
    data_calc[r'Hull/Acc/Y2_pred'] = lambda_y(**data_calc, **point2, **meta_data)
    data_calc[r'Hull/Acc/Z2_pred'] = lambda_z(**data_calc, **point2, **meta_data)
    
    data_calc[r'Hull/Acc/Z3_pred'] = lambda_z(**data_calc, **point3, **meta_data)
    
    return data_calc



def accelerometers_to_origo(data_calc):


    
    environment = {
    }
    if not 'g' in data_calc:
        environment['g'] = 9.81
    
    
    data_calc['x2d_P'] = data_calc['Hull/Acc/X1']
    data_calc['y2d_P'] = data_calc['Hull/Acc/Y1']
    data_calc['u1d_pred'] = lambda_u1d_from_accelerometer(**data_calc, **point1, **environment)
    data_calc['v1d_pred1'] = lambda_v1d_from_accelerometer(**data_calc, **point1, **environment)
    data_calc['r1d_pred1'] = lambda_r1d_from_accelerometer(**data_calc, **point1, **environment)
    
    data_calc['y2d_P'] = data_calc['Hull/Acc/Y2']
    data_calc['v1d_pred2'] = lambda_v1d_from_accelerometer(**data_calc, **point2, **environment)
    data_calc['r1d_pred2'] = lambda_r1d_from_accelerometer(**data_calc, **point2, **environment)
    
    data_calc['v1d_pred'] = (data_calc['v1d_pred1'] + data_calc['v1d_pred2'])/2
    data_calc['r1d_pred'] = (data_calc['r1d_pred1'] + data_calc['r1d_pred2'])/2

    return data_calc

def accelerometers_6_to_origo(data:pd.DataFrame):
    
    result = data.copy()
    
    environment = {
    }
    if not 'g' in data:
        environment['g'] = 9.81
    
    
    ## proper acceleration at the origin:
    c = acc(
    xacc1=data['Hull/Acc/X1'],
    yacc1=data['Hull/Acc/Y1'],
    yacc2=data['Hull/Acc/Y2'],
    zacc1=data['Hull/Acc/Z1'],
    zacc2=data['Hull/Acc/Z2'],
    zacc3=data['Hull/Acc/Z3'],
    xco=0,
    yco=0,
    zco=0,
    )
        
    df_acc = pd.DataFrame(c.T,columns=['ddotx_P','ddoty_P','ddotz_P'], index=data.index)

    ## Convert from proper acceleration to v1d by removing the centrepetal force and g etc.
    result['v1d'] = accelerometers6.lambda_v1d_from_6_accelerometers(**df_acc, **data, **environment)
    
    x_P0 = point1['x_P']
    x_P1 = point2['x_P']
    result['r1d'] = accelerometers6.lambda_r1d_from_6_accelerometers(ddoty_P0=data['Hull/Acc/Y1'], ddoty_P1=data['Hull/Acc/Y2'], x_P0=x_P0, x_P1=x_P1, phi=data['phi'])
    
    ## u1d
    result['u1d'] = lambda_u1d_from_accelerometer(**data, **point1, **environment)
    
    
    return result

def plot_pred(data_calc:pd.DataFrame, keys_acc:str, ax, t0 = 5, skip=[]):

    if not isinstance(keys_acc, list):
        keys_acc = [keys_acc]

    styles=['-',':']
    alphas = [1,0.5]
    colors = ['b','r']
    
    for key in keys_acc:

        color=colors.pop(0)
        
        if not key in skip:
            plot = data_calc.loc[t0:data_calc.index[-1]-t0].plot(y=key, style=styles.pop(0), ms=0.5, lw=0.7, alpha=alphas.pop(0), color=color, ax=ax)

        key=f"{key}_pred"
        if not key in skip:
            data_calc.loc[t0:data_calc.index[-1]-t0].plot(y=key, style='--', alpha=1, lw=0.7, color=color, ax=ax)

def plot_accelerometers(data_calc, rescale_axis=True, skip=[]):

    skip = set(skip)
    
    fig,axes=plt.subplots(nrows=4)
    
    ax=axes[0]
    data_calc['phi_deg'] = np.rad2deg(data_calc['phi'])
    data_calc.plot(y='phi_deg',ax=ax)
    
    ax=axes[1]
    keys = [r'Hull/Acc/X1']
    if len(keys) > 0:
        plot_pred(data_calc, keys, ax=ax, skip=skip)
    
    ax=axes[2]
    keys = ['Hull/Acc/Y1','Hull/Acc/Y2']
    if len(keys) > 0:
        plot_pred(data_calc, keys, ax=ax, skip=skip)
    
    ax=axes[3]
    keys = ['Hull/Acc/Z2','Hull/Acc/Z3']
    if len(keys) > 0:
        plot_pred(data_calc, keys, ax=ax, skip=skip)
    
    if rescale_axis:
        ymaxs=[]
        ymins=[]
        for ax in axes[1:]:
            ylims = ax.get_ylim()
            ymin = np.min([0,ylims[0]])
            ymax = np.max([0,ylims[1]])
            ymins.append(ymin)
            ymaxs.append(ymax)
            
        for ax in axes[1:]:
            ax.set_ylim(np.min(ymins), np.max(ymaxs))
    else:
        for ax in axes[1:]:
            ylims = ax.get_ylim()
            ymin = np.min([0,ylims[0]])
            ymax = np.max([0,ylims[1]])
            ax.set_ylim(np.min(ymin), np.max(ymax))
    
    plt.tight_layout()

    return fig

def plot_accelerations(data_calc):
    
    fig,axes=plt.subplots(nrows=3)
        
    t0=5
    
    ax=axes[0]
    data_calc.loc[t0:data_calc.index[-1]-t0].plot(y='u1d_pred', ax=ax)
    data_calc.loc[t0:data_calc.index[-1]-t0].plot(y='u1d', ax=ax)
    
    ax=axes[1]
    data_calc.loc[t0:data_calc.index[-1]-t0].plot(y='v1d_pred1', ax=ax)
    data_calc.loc[t0:data_calc.index[-1]-t0].plot(y='v1d_pred2', ax=ax)
    data_calc.loc[t0:data_calc.index[-1]-t0].plot(y='v1d_pred', ax=ax)
    data_calc.loc[t0:data_calc.index[-1]-t0].plot(y='v1d', ax=ax)
    
    ax=axes[2]
    data_calc.loc[t0:data_calc.index[-1]-t0].plot(y='r1d_pred1', ax=ax)
    data_calc.loc[t0:data_calc.index[-1]-t0].plot(y='r1d_pred2', ax=ax)
    data_calc.loc[t0:data_calc.index[-1]-t0].plot(y='r1d_pred', ax=ax)
    data_calc.loc[t0:data_calc.index[-1]-t0].plot(y='r1d', ax=ax)
    
    for ax in axes:
        ylims = ax.get_ylim()
        ymin = np.min([0,ylims[0]])
        ymax = np.max([0,ylims[1]])
        ax.set_ylim(np.min(ymin), np.max(ymax))
    
    plt.tight_layout()

    return fig 

def acc(xacc1,yacc1,yacc2,zacc1,zacc2,zacc3,xco,yco,zco, point1:dict, point2:dict, point3:dict):
    """

    SSPA Sweden AB
    Lennart Byström 98-10-09

    Routine for calculation of accelarations in the x-, y- and
    z-direction,xdd, ydd and zdd,at an arbitrary point, based
    on measurements from model tests.

    Coordinate system:
    -----------------
    x-axis towards the bow
    y-axis to the starboard
    z-axis downwards

    Indata:

    the 1:st accelerometer measures acceleration in the x-direction
    at a position with coordinates  x1,y1,z1.  It is called 'X1'

    the 2:nd accelerometer measures acceleration in the y-direction
    at a position with coordinates x2,y2,z2


    the 3:rd accelerometer measures acceleration in the y-direction
    at a position with coordinates x3,y3,z3


    the 4:th accelerometer measures acceleration in the z-direction
    at a position with coordinates x4,y4,z4


    the 5:th accelerometer measures acceleration in the z-direction
    at a position with coordinates x5,y5,z5


    the 6:th accelerometer measures acceleration in the z-direction
    at a position with coordinates x6,y6,z6

    xco = x-coordinate of the new position
    yco = y-coordinate of the new position
    zco = z-coordinate of the new position

    -----------coordinates of accelerometers-------------------

    x-axeln längs BL och origo i AP 
    y-axel positiv åt styrbord
    z rel. BL, neg uppåt
  
    """

    # Accelerometer no 1 measuring in the x-direction
    y1=point1['y_P']
    z1=point1['z_P']
    
    # Accelerometer no 2 and 3 measuring in the y-direction
    x2=point1['x_P']
    z2=point1['z_P']
    #
    x3=point2['x_P']
    z3=point2['z_P']
    
    # Accelerometer no 4,5 and 6 measuring in the z-direction
    x4=point1['x_P']
    y4=point1['y_P']
    x5=point2['x_P']
    y5=point2['y_P']

    x6=point3['x_P']
    y6=point3['y_P']
    
    #   direction     coord
    a=np.array([
    [1, 0, 0,0 ,  z1, y1],#meas. dir. and coord. of 1. accelerom.
    [0, 1, 0,z2,  0 , x2],#meas. dir. and coord. of 2. accelerom
    [0, 1, 0,z3,  0 , x3],
    [0, 0, 1,y4, -x4, 0 ],
    [0, 0, 1,y5, -x5, 0 ],
    [0, 0, 1,y6, -x6, 0 ],
    ])
    
    ierr=0
    eps=np.finfo(float).eps
    if np.abs(det(a)) < eps: #eps is floating-point relative accuracy
       raise ValueError('Matrisen med koordinater är singulär')

    b=inv(a) #invert matrix with directions and accelerometer coordinates

    #  prepare a matrix for calculation of acclerations in
    #  the x-, y- and z-direction
    aa=np.array([
    [1, 0, 0,   0 ,  zco, -yco],
    [0, 1, 0, -zco,   0 ,  xco],
    [0, 0, 1,  yco, -xco,   0 ],
    ]) #matrix with coordinates of 'new point'
        
    #measured accelerations from 6 sensors (this comes from indata to function acc.m)
    #xacc1=xacc1(:) 
    #yacc1=yacc1(:) 
    #yacc2=yacc2(:)
    #zacc1=zacc1(:)
    #zacc2=zacc2(:) 
    #zacc3=zacc3(:) 
    #accel=[xacc1 yacc1 yacc2 zacc1 zacc2 zacc3] #measured accel of sensors
    accel=np.array([xacc1, yacc1, yacc2, zacc1, zacc2, zacc3]) #measured accel of sensors
    

    #CORE PART of program (calculate acc at 'new'  point:
    accref = b @ accel          #b is inverted matrix from above
    c= aa @ accref               #acc at new point
        
    return c