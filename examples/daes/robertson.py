import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from solve_dae.integrate import solve_dae, consistent_initial_conditions


"""Robertson problem of semi-stable chemical reaction, see mathworks and Shampine2005.

References:
-----------
mathworks: https://de.mathworks.com/help/matlab/math/solve-differential-algebraic-equations-daes.html#bu75a7z-5 \\
Shampine2005: https://doi.org/10.1016/j.amc.2004.12.011 \\
Sundials IDA (page 6): https://computing.llnl.gov/sites/default/files/ida_examples-5.7.0.pdf
"""

def f(t, y):
    y1, y2, y3 = y

    yp = np.zeros(3, dtype=y.dtype)
    yp[0] = -0.04 * y1 + 1e4 * y2 * y3
    yp[1] = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    yp[2] = 3e7 * y2**2

    return yp

def F(t, y, yp):
    y1, y2, y3 = y
    y1p, y2p, y3p = yp

    F = np.zeros(3, dtype=np.common_type(y, yp))
    F[0] = y1p - (-0.04 * y1 + 1e4 * y2 * y3)
    F[1] = y2p - (0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2)
    F[2] = y1 + y2 + y3 - 1

    return F

def jac(t, y, yp):
    y1, y2, y3 = y

    Jyp = np.diag([1, 1, 0])
    Jy = np.array([
        [0.04, -1e4 * y3, -1e4 * y2],
        [-0.04, 1e4 * y3 + 6e7 * y2, 1e4 * y2],
        [1, 1, 1],
    ])
    return Jy, Jyp


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 4e10
    t_span = (t0, t1)
    # t_eval = np.logspace(-6, 7, num=200)
    t_eval = None

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0 = np.array([1, 0, 0], dtype=float)
    yp0 = f(t0, y0)

    yp0 = np.zeros_like(y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, fnorm = consistent_initial_conditions(F, t0, y0, yp0, fixed_y0=[0, 1])
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"fnorm: {fnorm}")

    # solver options
    atol = rtol = 1e-8

    ####################
    # reference solution
    ####################
    start = time.time()
    sol = solve_ivp(f, t_span, y0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t_scipy = sol.t
    y_scipy = sol.y
    yp_scipy = np.array([f(ti, yi) for ti, yi in zip(t_scipy, y_scipy.T)]).T
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"success: {success}")
    print(f"status: {status}")
    print(f"message: {message}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    ##############
    # dae solution
    ##############    
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, jac=jac, dense_output=True)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    y = sol.y
    yp = sol.yp
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"success: {success}")
    print(f"status: {status}")
    print(f"message: {message}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    # y, yp = sol.sol(t)

    # visualization
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, y[0], "-ok", label="y1 DAE" + f" ({method})", mfc="none")
    ax[0].plot(t, y[1] * 1e4, "-ob", label="y2 DAE" + f" ({method})", mfc="none")
    ax[0].plot(t, y[2], "-og", label="y3 DAE" + f" ({method})", mfc="none")
    ax[0].plot(t_scipy, y_scipy[0], "xr", label="y1 scipy" + f" ({method})", markersize=7)
    ax[0].plot(t_scipy, y_scipy[1] * 1e4, "xy", label="y2 scipy" + f" ({method})", markersize=7)
    ax[0].plot(t_scipy, y_scipy[2], "xm", label="y3 scipy" + f" ({method})", markersize=7)
    ax[0].set_xlabel("t")
    ax[0].set_xscale("log")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, yp[0], "-ok", label="yp1 DAE" + f" ({method})", mfc="none")
    ax[1].plot(t, yp[1], "-ob", label="yp2 DAE" + f" ({method})", mfc="none")
    ax[1].plot(t, yp[2], "-og", label="yp3 DAE" + f" ({method})", mfc="none")
    ax[1].plot(t_scipy, yp_scipy[0], "xr", label="yp1 scipy" + f" ({method})", markersize=7)
    ax[1].plot(t_scipy, yp_scipy[1], "xy", label="yp2 scipy" + f" ({method})", markersize=7)
    ax[1].plot(t_scipy, yp_scipy[2], "xm", label="yp3 scipy" + f" ({method})", markersize=7)
    ax[1].set_xlabel("t")
    ax[1].set_xscale("log")
    ax[1].legend()
    ax[1].grid()

    plt.show()