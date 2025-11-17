import numpy as np
from numpy.linalg import det, inv


def acc(
    xacc1,
    yacc1,
    yacc2,
    zacc1,
    zacc2,
    zacc3,
    point1,
    point2,
    point3,
    xco,
    yco,
    zco,
    **kwargs
):
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

    point1 = coordinates of accelerometer no 1
    point2 = coordinates of accelerometer no 2
    point3 = coordinates of accelerometer no 3

    xco = x-coordinate of the new position
    yco = y-coordinate of the new position
    zco = z-coordinate of the new position

    -----------coordinates of accelerometers-------------------

    x-axeln längs BL och origo i AP
    y-axel positiv åt styrbord
    z rel. BL, neg uppåt

    """

    # Accelerometer no 1 measuring in the x-direction
    y1 = point1[1]
    z1 = point1[2]

    # Accelerometer no 2 and 3 measuring in the y-direction
    x2 = point1[0]
    z2 = point1[2]
    #
    x3 = point2[0]
    z3 = point2[2]

    # Accelerometer no 4,5 and 6 measuring in the z-direction
    x4 = point1[0]
    y4 = point1[1]
    x5 = point2[0]
    y5 = point2[1]

    x6 = point3[0]
    y6 = point3[1]

    #   direction     coord
    a = np.array(
        [
            [1, 0, 0, 0, z1, y1],  # meas. dir. and coord. of 1. accelerom.
            [0, 1, 0, z2, 0, x2],  # meas. dir. and coord. of 2. accelerom
            [0, 1, 0, z3, 0, x3],
            [0, 0, 1, y4, -x4, 0],
            [0, 0, 1, y5, -x5, 0],
            [0, 0, 1, y6, -x6, 0],
        ]
    )

    ierr = 0
    eps = np.finfo(float).eps
    if np.abs(det(a)) < eps:  # eps is floating-point relative accuracy
        raise ValueError("Matrisen med koordinater är singulär")

    b = inv(a)  # invert matrix with directions and accelerometer coordinates

    #  prepare a matrix for calculation of acclerations in
    #  the x-, y- and z-direction
    aa = np.array(
        [
            [1, 0, 0, 0, zco, -yco],
            [0, 1, 0, -zco, 0, xco],
            [0, 0, 1, yco, -xco, 0],
        ]
    )  # matrix with coordinates of 'new point'

    # measured accelerations from 6 sensors (this comes from indata to function acc.m)
    # xacc1=xacc1(:)
    # yacc1=yacc1(:)
    # yacc2=yacc2(:)
    # zacc1=zacc1(:)
    # zacc2=zacc2(:)
    # zacc3=zacc3(:)
    # accel=[xacc1 yacc1 yacc2 zacc1 zacc2 zacc3] #measured accel of sensors
    accel = np.array(
        [xacc1, yacc1, yacc2, zacc1, zacc2, zacc3]
    )  # measured accel of sensors

    # CORE PART of program (calculate acc at 'new'  point:
    accref = b @ accel  # b is inverted matrix from above
    c = aa @ accref  # acc at new point

    return c
