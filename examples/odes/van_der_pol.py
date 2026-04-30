import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from solve_dae.integrate import solve_dae, consistent_initial_conditions


"""Stiff van der Pol equation, see [1]_.

References
----------
.. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
       Stiff and Differential-Algebraic Problems", Sec. IV.8.
"""


eps = 1e-6

def rhs(t, y):
    y1, y2 = y

    yp = np.zeros(2, dtype=y.dtype)
    yp[0] = y2
    yp[1] = ((1 - y1 * y1) * y2 - y1) / eps

    return yp

def F(t, y, yp):
    return yp - rhs(t, y)


def f(t, z):
    y, yp = z[:2], z[2:]
    return np.concatenate((yp, F(t, y, yp)))


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 10
    t_span = (t0, t1)

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0 = np.array([2, -0.66], dtype=float)
    yp0 = rhs(t0, y0)
    z0 = np.concatenate((y0, yp0))

    yp0 = np.zeros_like(y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, fnorm = consistent_initial_conditions(F, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"fnorm: {fnorm}")

    # solver options
    atol = rtol = 7e-5

    t_eval = np.linspace(t0, t1, num=int(1e3))
    t_eval = None
    first_step = 1e-6

    ####################
    # reference solution
    ####################
    start = time.time()
    sol = solve_ivp(rhs, t_span, y0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, first_step=first_step)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t_scipy = sol.t
    y_scipy = sol.y
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
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, first_step=first_step, dense_output=True)
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

    # visualization
    fig, ax = plt.subplots(4, 1)

    t_eval = np.linspace(t0, t1, num=int(1e3))
    y_eval, yp_eval = sol.sol(t_eval)

    ax[0].plot(t, y[0], "ok", label=f"y ({method})", mfc="none")
    ax[0].plot(t_eval, y_eval[0], "-xk", label=f"y_dense ({method})", mfc="none")
    ax[0].plot(t_scipy, y_scipy[0], "-xr", label="y scipy")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[1], "ok", label=f"y_dot ({method})", mfc="none")
    ax[1].plot(t_eval, y_eval[1], "-k", label=f"y_dense ({method})", mfc="none")
    ax[1].plot(t_scipy, y_scipy[1], "-xr", label="y_dot scipy")
    ax[1].legend()
    ax[1].grid()

    yp_scipy = np.array([rhs(ti, yi) for ti, yi in zip(t_scipy, y_scipy.T)]).T

    ax[2].plot(t, yp[0], "-ok", label=f"yp0 ({method})", mfc="none")
    ax[2].plot(t_scipy, yp_scipy[0], "-xr", label="yp0 scipy")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(t, yp[1], "-ok", label=f"yp1 ({method})", mfc="none")
    ax[3].plot(t_scipy, yp_scipy[1], "-xr", label="yp1 scipy")
    ax[3].legend()
    ax[3].grid()

    plt.show()
