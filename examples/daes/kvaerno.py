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
    y1, y2, y3 = y
    yp1, yp2, yp3 = yp
    return np.array([
        yp1 + y3 * yp2 - (y2 + 1) * yp3 + y1 - 1 - np.sin(t),
        (y3 + 1) * yp1 + y1 * yp2 + np.exp(-t),
        y1 * y2 * y3 - 0.5 * np.exp(-t) * np.sin(2 * t),
    ])


def true_sol(t):
    return (
        np.array([
            np.exp(-t),
            np.sin(t),
            np.cos(t),
        ]),
        np.array([
            -np.exp(-t),
            np.cos(t),
            -np.sin(t),
        ])
    )


if __name__ == "__main__":
    # time span
    t0 = -1
    t1 = 1
    t_span = (t0, t1)

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0, yp0 = true_sol(t0)

    # solver options
    atol = rtol = 1e-6

    # run the solver
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method)
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
    ax.plot(t, true_sol(t)[0][0], "or", label="y1_true")
    ax.plot(t, true_sol(t)[0][1], "og", label="y2_true")
    ax.plot(t, true_sol(t)[0][2], "ob", label="y3_true")
    ax.plot(t, y[0], "-xr", label=f"y1 {method}")
    ax.plot(t, y[1], "-xg", label=f"y2 {method}")
    ax.plot(t, y[2], "-xb", label=f"y3 {method}")
    ax.grid()
    ax.legend()

    plt.show()
