import numpy as np
from solve_dae.integrate._dae.benchmarks.common import benchmark


"""Robertson problem of semi-stable chemical reaction, see mathworks and Shampine2005.

References:
-----------
mathworks: https://de.mathworks.com/help/matlab/math/solve-differential-algebraic-equations-daes.html#bu75a7z-5 \\
Shampine2005: https://doi.org/10.1016/j.amc.2004.12.011 \\
Sundials IDA (page 6): https://computing.llnl.gov/sites/default/files/ida_examples-5.7.0.pdf \\
Test Set for IVP Solvers: https://archimede.uniba.it/~testset/report/rober.pdf
"""
def F(t, y, yp):
    y1, y2, y3 = y
    y1p, y2p, y3p = yp

    F = np.zeros(3, dtype=float)
    F[0] = y1p - (-0.04 * y1 + 1e4 * y2 * y3)
    F[1] = y2p - (0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2)
    F[2] = y1 + y2 + y3 - 1

    return F


# exponents
m_max = 32
ms = np.arange(m_max + 1)

# tolerances and initial step size
rtols = 10**(-(5.5 + ms / 4))
atols = rtols
h0s = 1e-2 * rtols


def run_robertson():
    # time span
    t0 = 0
    t1 = 1e11

    # initial conditions
    y0 = np.array([1, 0, 0], dtype=float)
    yp0 = np.array([-0.04, 0.04, 0], dtype=float)

    # reference value ODE
    # see https://archimede.uniba.it/~testset/report/rober.pdf
    y_ref = np.array([
        0.2083340149701255e-7,
        0.8333360770334713e-13,
        0.9999999791665050,
    ], dtype=float)

    # # reference value DAE (IDA)
    # y_ref = np.array([
    #     2.08333986263466290e-08,
    #     8.33335962206821274e-14,
    #     9.99999979166517949e-01,
    # ], dtype=float)

    # y_ref = None

    benchmark(t0, t1, y0, yp0, F, rtols, atols, h0s, "Robertson", y_ref)


if __name__ == "__main__":
    run_robertson()
