import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae

"""
Problem I.542 of E. Kamke.

References:
-----------
..[1] E. Kamke, Differentialgleichungen - Lösungsmethoden und Lösungen, Bd. 1, 1948, p. 389.
"""

C = 1

def F(t, y, yp):
    return 16 * y**2 * yp**3 + 2 * t * yp - y


def true_sol(t):
    return np.atleast_1d(np.sqrt(C * t + 2 * C**3)), np.atleast_1d(C / np.sqrt(C * t + 2 * C**3))


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 1e3
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

    # error
    error = np.linalg.norm(y[:, -1] - true_sol(t1)[0])
    print(f"error: {error}")

    # visualization
    fig, ax = plt.subplots()

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.plot(t, true_sol(t)[0], "-ok", label="y_true")
    ax.plot(t, y[0], "--xr", label=f"y {method}")
    ax.grid()
    ax.legend()

    plt.show()
