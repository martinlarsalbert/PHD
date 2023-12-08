import sympy as sp
from vessel_manoeuvring_models.symbols import *
import pandas as pd

from vessel_manoeuvring_models.parameters import df_parameters

p = df_parameters["symbol"]

eq_X_H = sp.Eq(X_H, p.X0 + p.Xu * u)
eq_Y_H = sp.Eq(Y_H, p.Y0 + p.Yv * v + p.Yr * r)
eq_N_H = sp.Eq(N_H, p.N0 + p.Nv * v + p.Nr * r)

eq_X_R = sp.Eq(X_R, 0)
eq_Y_R = sp.Eq(Y_R, p.Ydelta * delta)
eq_N_R = sp.Eq(N_R, p.Ndelta * delta + p.Nvdeltadelta * v * delta**2)

eq_X_R_thrust = sp.Eq(X_R, 0)
eq_Y_R_thrust = sp.Eq(Y_R, p.Ydelta * delta + p.Ythrustdelta * thrust * delta)
eq_N_R_thrust = sp.Eq(N_R, p.Ndelta * delta + p.Nthrustdelta * thrust * delta)