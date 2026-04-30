import numpy as np
from solve_dae.integrate._dae.benchmarks.common import benchmark


"""Nonlinear index 1 DAE, see Kvaerno1990.

References:
-----------
Kvaerno1990: https://doi.org/10.2307/2008502
"""
def F(t, y, yp):
    y1, y2 = y
    yp1, yp2 = yp
    return np.array([
        (np.sin(yp1)**2 + np.cos(y2)**2) * yp2**2 - (t - 6)**2 * (t - 2)**2 * y1 * np.exp(-t),
        (4 - t) * (y2 + y1)**3 - 64 * t**2 * np.exp(-t) * y1 * y2,
    ])


def true_sol(t):
    return (
        np.array([
            t**4 * np.exp(-t),
            (4 - t) * t**3 * np.exp(-t),
        ]),
        np.array([
            (4 * t**3 - t**4) * np.exp(-t),
            ((4 - t) * 3 * t**2 - (5 - t) * t**3) * np.exp(-t)
        ])
    )


def run_kvaerno():
    # exponents
    m_max = 32
    ms = np.arange(m_max + 1)

    # tolerances and initial step size
    rtols = 10**(-(4 + ms / 4))
    atols = rtols
    h0s = 1e-3 * np.ones_like(rtols)

    # time span
    t0 = 0.5
    t1 = 1

    # initial conditions
    y0, yp0 = true_sol(t0)

    # reference solution
    y_ref = true_sol(t1)[0]

    benchmark(t0, t1, y0, yp0, F, rtols, atols, h0s, "Kvaerno", y_ref)


if __name__ == "__main__":
    run_kvaerno()
