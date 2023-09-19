import sympy.physics.mechanics as me
import sympy as sp
from sympy.physics.vector import ReferenceFrame, Point
from vessel_manoeuvring_models.substitute_dynamic_symbols import run, lambdify
from vessel_manoeuvring_models.symbols import *

theta, phi, psi, V = me.dynamicsymbols("theta,phi,psi,V")
x_0, y_0 = me.dynamicsymbols("x_0 y_0")

I = ReferenceFrame("I")  # Inertial frame
B = ReferenceFrame("B")  # Body frame
B.orient(parent=I, rot_type="Body", amounts=(psi, theta, phi), rot_order="321")

O = Point("O")  # World Origo, fixed in I
O.set_vel(I, 0)

O_B = Point("O_B")  # Origo of B, Point fixed in B with known velocity in I.
O_B.set_pos(O, x_0 * I.x + y_0 * I.y)
u, v = me.dynamicsymbols("u v")
O_B.set_vel(I, u * B.x + v * B.y)


## Move GPS position to origo:
x_GPS_I, y_GPS_I, z_GPS_I = sp.symbols("x_GPS_I,y_GPS_I,z_GPS_I")
P_GPS = Point("P_GPS")  # Point of the GPS, fixed in B
x_GPS_B, y_GPS_B, z_GPS_B = sp.symbols("x_GPS_B,y_GPS_B,z_GPS_B")
P_GPS.set_pos(O_B, x_GPS_B * B.x + y_GPS_B * B.y + z_GPS_B * B.z)

eq_P_GPS = sp.Eq(
    (x_GPS_I * I.x + y_GPS_I * I.y + z_GPS_I * I.z).to_matrix(I),
    P_GPS.pos_from(O).express(I).to_matrix(I),
)

# Origo position from GPS measurement:
eq_x_0 = sp.Eq(x_0, sp.solve(sp.Eq(eq_P_GPS.lhs[0], eq_P_GPS.rhs[0]), x_0)[0])
eq_y_0 = sp.Eq(y_0, sp.solve(sp.Eq(eq_P_GPS.lhs[1], eq_P_GPS.rhs[1]), y_0)[0])

lambda_x_0 = lambdify(eq_x_0.rhs)
lambda_y_0 = lambdify(eq_y_0.rhs)

## Move accelerations from accelerometers to origo:
P = Point("P")  # Point of the accelerometer, fixed in B with known velocity.
x_acc, y_acc, z_acc = sp.symbols("x_acc,y_acc,z_acc")
P.set_pos(O_B, x_acc * B.x + y_acc * B.y + z_acc * B.z)

acceleration = P.acc(I).express(B).to_matrix(B)

u_acc = me.dynamicsymbols("u_acc")
v_acc = me.dynamicsymbols("v_acc")
subs = [
    (psi.diff().diff(), "r1d"),
    (psi.diff(), "r"),
    (psi, "psi"),
    (u.diff(), "u1d"),
    (v.diff(), "v1d"),
    (u, "u"),
    (v, "v"),
    (phi.diff().diff(), "p1d"),
    (phi.diff(), "p"),
    (phi, "phi"),
    (theta.diff().diff(), "q1d"),
    (theta.diff(), "q"),
    (theta, "theta"),
    (u_acc.diff(), "AccelX"),
    (v_acc.diff(), "AccelY"),
    (x_0, "x0"),
    (y_0, "y0"),
]


eq = acceleration.subs(subs)
lambda_acceleration = sp.lambdify(list(eq.free_symbols), eq)

eq_u1d_acc = sp.Eq(u_acc.diff(), (acceleration[0] - sp.sin(theta) * g))
eq_v1d_acc = sp.Eq(v_acc.diff(), (acceleration[1] - sp.sin(phi) * g))
eq = eq_v1d_acc.subs(subs)
lambda_acceleration_y = sp.lambdify(list(eq.rhs.free_symbols), eq.rhs)
eq = eq_u1d_acc.subs(subs)
lambda_acceleration_x = sp.lambdify(list(eq.rhs.free_symbols), eq.rhs)

## Move to origo:
eq_u1d = sp.Eq(u.diff(), sp.solve(eq_u1d_acc, u.diff())[0])
eq = eq_u1d.subs(subs)
lambda_u1d = lambdify(eq.rhs)

eq_v1d = sp.Eq(v.diff(), sp.solve(eq_v1d_acc, v.diff())[0])
eq = eq_v1d.subs(subs)
lambda_v1d = lambdify(eq.rhs)
