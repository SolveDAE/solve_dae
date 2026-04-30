import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae


"""Jackiewicz's implicit differential equation, see Jackiewicz1981.

References:
-----------
Jackiewicz1981: https://eudml.org/doc/15186
"""
def F(t, y, yp):
    return yp - (
        (np.sin(t**2 * yp)) / 16
        - np.sin(np.exp(y)) / 16
        + 1 / t
    )

def true_sol(t):
    return np.atleast_1d(np.log(t)), np.atleast_1d(1 / t)


if __name__ == "__main__":
    # time span
    t0 = 1
    t1 = 4
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
    ax.plot(t, true_sol(t)[0], "-ok", label="y_true")
    ax.plot(t, y[0], "--xr", label=f"y {method}")
    ax.grid()
    ax.legend()

    plt.show()
