import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae, consistent_initial_conditions


"""Cartesian pendulum, see Hairer1996 Section VII Example 2."""
m = 1
l = 1
g = 10

def F(t, vy, vyp):
    # stabilized index 1
    x, y, u, v, _, _ = vy
    x_dot, y_dot, u_dot, v_dot, la, mu = vyp

    R = np.zeros(6, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - u - 2 * x * mu
    R[1] = y_dot - v - 2 * y * mu
    R[2] = m * u_dot - 2 * x * la
    R[3] = m * v_dot - 2 * y * la + m * g
    R[4] = 2 * x * u + 2 * y * v
    R[5] = x * x + y * y - l * l

    return R


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 10
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e3))
    t_eval = None

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0 = np.array([l, 0, 0, 0, 0, 0], dtype=float)
    yp0 = np.array([0, 0, 0, -g, 0, 0], dtype=float)

    yp0 = np.zeros_like(yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, f0 = consistent_initial_conditions(F, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"f: {f0}")
    assert np.allclose(f0, np.zeros_like(f0))

    # solver options
    atol = rtol = 1e-4

    ##############
    # dae solution
    ##############
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)
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

    ax[0].plot(t, y[0], "-ok", label="x")
    ax[0].plot(t, y[1], "--xk", label="y")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[2], "-ok", label="u")
    ax[1].plot(t, y[3], "--xk", label="v")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t, yp[4], "-ok", label="la")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(t, yp[5], "--xk", label="mu")
    ax[3].legend()
    ax[3].grid()

    plt.show()
