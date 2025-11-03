import sympy.physics.mechanics as me
import sympy as sp
from sympy import Eq, symbols
from sympy.physics.vector import ReferenceFrame, Point
from vessel_manoeuvring_models.substitute_dynamic_symbols import run, lambdify
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.substitute_dynamic_symbols import equation_to_python_method, expression_to_python_method, eq_dottify
import phd.pipelines.load_wPCC.reference_frames as reference_frames

## With 6 accelerometers

## v1d
subs=[
    (reference_frames.x_P,0),
    (reference_frames.y_P,0),
    (reference_frames.z_P,0),
    (reference_frames.theta_,0),
    (q,0),
    
]
eq_y2d_0 = reference_frames.eq_y2d_P.subs(subs)  # Transverse acceleration at the origin
eq_v1d_simplified = Eq(v1d,sp.solve(eq_y2d_0,v1d)[0])  # Calculate sway acceleration (removing the centrepetal force etc.)
lambda_v1d_from_6_accelerometers = expression_to_python_method(eq_v1d_simplified.rhs, function_name='v1d')

## r1d
x_P,y_P,z_P = me.dynamicsymbols("x_P,y_P,z_P")
eq_y_P2d = Eq(y_P.diff().diff(),reference_frames.acceleration_g[1])

x_Ps=[]
y_Ps=[]
z_Ps=[]
n_=2
for i in range(n_):
    x_P_,y_P_,z_P_ = me.dynamicsymbols(f"x_P{i},y_P{i},z_P{i}")
    x_Ps.append(x_P_)
    y_Ps.append(y_P_)
    z_Ps.append(z_P_)

eq_y_P2ds = []
for i in range(n_):
    eq_y_P2ds.append(eq_y_P2d.subs([
        (reference_frames.x_P,x_Ps[i]),
        (reference_frames.y_P,y_Ps[i]),
        (reference_frames.z_P,z_Ps[i]),
        (y_P,y_Ps[i]),
        
    ]    
    ))
    
eq = Eq(eq_y_P2ds[1].lhs-eq_y_P2ds[0].lhs,eq_y_P2ds[1].rhs-eq_y_P2ds[0].rhs, evaluate=False)

eq_2 = eq.subs([
    (z_Ps[0],z_Ps[1],),
    (y_Ps[0],y_Ps[1],), # y_P0==y_P1, same y coord of accelerometers
    (reference_frames.theta.diff().diff(),0),
    (reference_frames.theta.diff(),0),
    (reference_frames.theta,0),
    (reference_frames.phi.diff().diff(),0),
    (reference_frames.phi.diff(),0),
]

)

eq_3 = eq_dottify(Eq(eq_y_P2ds[1].lhs-eq_y_P2ds[0].lhs, eq_2.subs(reference_frames.subs_removing_dynamic_symbols).rhs))
eq_r1d = Eq(r1d,sp.solve(eq_3,r1d)[0])
lambda_r1d_from_6_accelerometers = expression_to_python_method(eq_r1d.rhs, "r1d")