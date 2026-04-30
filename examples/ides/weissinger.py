import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae


"""Weissinger's implicit differential equation, see [1] and problem I.543 in [2].

References:
----------- 
..[1]_ https://www.mathworks.com/help/matlab/ref/ode15i.html#bu7u4dt-1
..[2]_ E. Kamke, Differentialgleichungen - Lösungsmethoden und Lösungen, Bd. 1, 1948, p. 389.
"""
def F(t, y, yp):
    return (
        t * y**2 * yp**3 
        - y**3 * yp**2 
        + t * (t**2 + 1) * yp 
        - t**2 * y
    )

def jac(t, y, yp):
    Jy = np.array([
        2 * t * y * yp**3
        - 3 * y**2 * yp**2  
        - t**2 * y,
    ])
    Jyp = np.array([
        3 * t * y**2 * yp**2 
        - 2 * y**3 * yp 
        + t * (t**2 + 1)
    ])
    return Jy, Jyp


def true_sol(t):
    return np.atleast_1d(np.sqrt(t**2 + 0.5)), np.atleast_1d(t / np.sqrt(t**2 + 0.5))


if __name__ == "__main__":
    # time span
    t0 = np.sqrt(0.5)
    t1 = 10
    t_span = (t0, t1)

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0, yp0 = true_sol(t0)

    # solver options
    atol = rtol = 1e-6

    # run the solver
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, jac=jac, atol=atol, rtol=rtol, method=method)
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
    error = np.linalg.norm(y[:, -1] - np.sqrt(t1**2 + 0.5))
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
