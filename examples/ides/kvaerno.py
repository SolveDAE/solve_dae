import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae


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


def jac(t, y, yp):
    y1, y2 = y
    yp1, yp2 = yp

    Jy = np.array([
        [-(t - 6)**2 * (t - 2)**2 * np.exp(-t), -2 * np.cos(y2) * np.sin(y2) * yp2**2],
        [3 * (4 - t) * (y2 + y1)**2 - 64 * t**2 * np.exp(-t) * y2, 3 * (4 - t) * (y2 + y1)**2 - 64 * t**2 * np.exp(-t) * y1]
    ])

    Jyp = np.array([
        [2 * np.sin(yp1) * np.cos(yp1) * yp2**2, (np.sin(yp1)**2 + np.cos(y2)**2) * 2 * yp2],
        [0, 0],
    ])

    return Jy, Jyp


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


if __name__ == "__main__":
    # time span
    t0 = 0.1
    t1 = 1.2
    t_span = (t0, t1)

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0, yp0 = true_sol(t0)

    # solver options
    atol = rtol = 1e-6

    # run the solver
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, jac=jac)
    end = time.time()
    t = sol.t
    y = sol.y
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"message: {message}")
    print(f"elapsed time: {end - start}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    # visualization
    fig, ax = plt.subplots()

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.plot(t, true_sol(t)[0][0], "-ok", label="y1_true")
    ax.plot(t, true_sol(t)[0][1], "-ob", label="y2_true")
    ax.plot(t, y[0], "--xr", label=f"y1 {method}")
    ax.plot(t, y[1], "--xg", label=f"y2 {method}")
    ax.grid()
    ax.legend()

    plt.show()
