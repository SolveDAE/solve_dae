import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.integrate._ivp.tests.test_ivp import compute_error


"""Modified particle on a circular track subject to tangential force, see Arevalo1995.
   We implement a stabilized index 1 formulation as proposed by Anantharaman1991.

References:
-----------
Arevalo1995: https://link.springer.com/article/10.1007/BF01732606 \\
Anantharaman1991: https://doi.org/10.1002/nme.1620320803
"""
omega = 2 * np.pi


def PHI(t):
    """The time derivative of this function has to be phi_dot(t)**2."""
    return omega**2 * (t / 2 + np.sin(2 * t) / 4)


def phi(t):
    return omega * np.sin(t)


def phi_p(t):
    return omega * np.cos(t)


def phi_pp(t):
    return -omega * np.sin(t)


def F(t, vy, vyp):
    x, y, u, v, _, _ = vy
    x_dot, y_dot, u_dot, v_dot, Lap, Mup = vyp

    force = phi_pp(t)

    R = np.zeros(6, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - (u + 2 * x * Mup)
    R[1] = y_dot - (v + 2 * y * Mup)
    R[2] = u_dot - (2 * x * Lap - y * force)
    R[3] = v_dot - (2 * y * Lap + x * force)
    R[4] = 2 * (x * u + y * v)
    R[5] = x**2 + y**2 - 1

    return R


def sol_true(t):
    y = np.array(
        [
            np.cos(phi(t)),
            np.sin(phi(t)),
            -np.sin(phi(t)) * phi_p(t),
            np.cos(phi(t)) * phi_p(t),
            -PHI(t) / 2,
            np.zeros_like(t),
        ]
    )

    yp = np.array(
        [
            -np.sin(phi(t)) * phi_p(t),
            np.cos(phi(t)) * phi_p(t),
            -np.cos(phi(t)) * phi_p(t) ** 2 - np.sin(phi(t)) * phi_pp(t),
            -np.sin(phi(t)) * phi_p(t) ** 2 + np.cos(phi(t)) * phi_pp(t),
            -phi_p(t) ** 2 / 2,
            np.zeros_like(t),
        ]
    )

    return y, yp


if __name__ == "__main__":
    # time span
    t0 = 1
    t1 = 5
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e3))

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0, yp0 = sol_true(t0)

    yp0 = np.zeros_like(y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, fnorm = consistent_initial_conditions(F, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"fnorm: {fnorm}")

    # solver options
    atol = rtol = 1e-5

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

    y_true, yp_true = sol_true(t)
    e = compute_error(y[:, -1], y_true[:, -1], rtol, atol)
    ep = compute_error(yp[:, -1], yp_true[:, -1], rtol, atol)
    print(f"max(e): {e}")
    print(f"max(ep): {ep}")

    # visualization
    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t, y[0], "-ok", label="x")
    ax[0].plot(t, y[1], "-ob", label="y")
    ax[0].plot(t, y_true[0], "--xr", label="x_true")
    ax[0].plot(t, y_true[1], "--xg", label="y_true")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[2], "-ok", label="u")
    ax[1].plot(t, y[3], "-ob", label="v")
    ax[1].plot(t, y_true[2], "--xr", label="u_true")
    ax[1].plot(t, y_true[3], "--xg", label="v_true")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t, yp[4], "-ok", label="lap")
    ax[2].plot(t, yp_true[4], "--xr", label="lap_true")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(t, yp[5], "-ok", label="mup")
    ax[3].plot(t, yp_true[5], "--xr", label="mup_true")
    ax[3].legend()
    ax[3].grid()

    plt.show()
