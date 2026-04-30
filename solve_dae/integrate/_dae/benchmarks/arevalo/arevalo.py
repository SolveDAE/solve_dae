import numpy as np
from solve_dae.integrate._dae.benchmarks.common import benchmark


"""Particle on a circular track subject to tangential force, see Arevalo1995.
   We implement a stabilized index 1 formulation as proposed by Anantharaman1991.

References:
-----------
Arevalo1995: https://link.springer.com/article/10.1007/BF01732606 \\
Anantharaman1991: https://doi.org/10.1002/nme.1620320803
"""
def F(t, vy, vyp):
    # stabilized index 1
    x, y, u, v, _, _ = vy
    x_dot, y_dot, u_dot, v_dot, lap, mup = vyp

    R = np.zeros(6, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - (u + x * mup)
    R[1] = y_dot - (v + y * mup)
    R[2] = u_dot - (2 * y + x * lap)
    R[3] = v_dot - (-2 * x + y * lap)
    R[4] = x * u + y * v
    R[5] = x * x + y * y - 1

    return R


def sol_true(t):
    y =  np.array([
        np.sin(t**2),
        np.cos(t**2),
        2 * t * np.cos(t**2),
        -2 * t * np.sin(t**2),
        -4 / 3 * t**3,
        0 * t,
    ])

    yp =  np.array([
        2 * t * np.cos(t**2),
        -2 * t * np.sin(t**2),
        2 * np.cos(t**2) - 4 * t**2 * np.sin(t**2),
        -2 * np.sin(t**2) - 4 * t**2 * np.cos(t**2),
        -4 * t**2,
        0 * t,
    ])

    return y, yp


def run_arevalo():
    # exponents
    m_max = 24
    ms = np.arange(m_max + 1)

    # tolerances and initial step size
    rtols = 10**(-(3 + ms / 4))
    atols = rtols
    h0s = 1e-2 * rtols

    # time span
    t0 = 1
    t1 = 5

    # initial conditions
    y0, yp0 = sol_true(t0)

    # reference solution
    y_ref = sol_true(t1)[0]

    benchmark(t0, t1, y0, yp0, F, rtols, atols, h0s, "Arevalo", y_ref)


if __name__ == "__main__":
    run_arevalo()
