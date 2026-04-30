import numpy as np
from solve_dae.integrate._dae.benchmarks.common import benchmark


"""Weissinger's implicit differential equation, see mathworks.

References:
-----------
mathworks: https://www.mathworks.com/help/matlab/ref/ode15i.html#bu7u4dt-1
"""
def F(t, y, yp):
    return (
        t * y**2 * yp**3 
        - y**3 * yp**2 
        + t * (t**2 + 1) * yp 
        - t**2 * y
    )


def true_sol(t):
    return np.atleast_1d(np.sqrt(t**2 + 0.5)), np.atleast_1d(t / np.sqrt(t**2 + 0.5))


def run_weissinger():
    # exponents
    m_max = 28
    ms = np.arange(m_max + 1)

    # tolerances and initial step size
    rtols = 10**(-(4 + ms / 4))
    atols = rtols
    h0s = 1e2 * rtols

    # time span
    t0 = np.sqrt(0.5)
    t1 = 10

    # initial conditions
    y0, yp0 = true_sol(t0)

    # reference solution
    y_ref = true_sol(t1)[0]

    benchmark(t0, t1, y0, yp0, F, rtols, atols, h0s, "Weissinger", y_ref)


if __name__ == "__main__":
    run_weissinger()
