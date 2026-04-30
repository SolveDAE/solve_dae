import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae, consistent_initial_conditions

m = 1
Theta = 0.1
a = 0.1

"""Index 2 DAE found in Section 1.7 of Bloch2015.

References:
-----------
Bloch2015: https://doi.org/10.1007/978-1-4939-3017-3
"""
def F(t, vy, vyp):
    x, y, phi, u, v, omega, _ = vy
    xp, yp, phip, up, vp, omegap, lap = vyp

    sphi, cphi = np.sin(phi), np.cos(phi)

    F = np.zeros_like(vy, dtype=np.common_type(vy, vyp))

    # F[0] = xp - u
    # F[1] = yp - v
    # F[2] = phip - omega
    # F[3] = up - l * omegap * sphi - l * omega**2 * cphi - sphi * lap
    # F[4] = vp + l * omegap * cphi - l * omega**2 * sphi + cphi * lap
    # F[5] = omegap
    # F[6] = -u * sphi + v * cphi

    # F[0] = xp - u
    # F[1] = yp - v
    # F[2] = phip - omega
    # F[3] = up - a * omegap * sphi - a * omega**2 * cphi + sphi * lap
    # F[4] = vp + a * omegap * cphi - a * omega**2 * sphi - cphi * lap
    # F[5] = omegap - a * lap
    # F[6] = -u * sphi + v * cphi + a * omega

    # Bloch 2005, equation (1.7.6)
    F[0] = xp - u
    F[1] = yp - v
    F[2] = phip - omega
    F[3] = up - a * cphi * omega**2 - a * sphi * omegap + sphi * lap / m
    F[4] = vp - a * sphi * omega**2 + a * cphi * omegap - cphi * lap / m
    F[5] = (Theta + m * a**2) * omegap + m * a * omega * (u * cphi + v * sphi)
    F[6] = -u * sphi + v * cphi

    return F


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 1e1
    t_span = (t0, t1)

    # tolerances
    rtol = atol = 1e-8

    # initial conditions
    c0 = 1 # body fixed velocity

    x0 = 0
    y0 = -1
    phi0 = 0
    u0 = 1
    v0 = 0
    omega0 = 0

    sphi0, cphi0 = np.sin(phi0), np.cos(phi0)
    up0 = a * omega0**2
    vp0 = 0
    omegap0 = - m * a / (Theta + m * a**2) * c0 * omega0


    lap0 = 0
    vy0 = np.array([x0, y0, phi0, u0, v0, omega0, 0], dtype=float)
    vyp0 = np.array([u0, v0, omega0, 0, 0, 0, lap0], dtype=float)

    print(f"vy0: {vy0}")
    print(f"vyp0: {vyp0}")
    vy0, vyp0, fnorm = consistent_initial_conditions(F, t0, vy0, vyp0)
    print(f"vy0: {vy0}")
    print(f"vyp0: {vyp0}")
    print(f"fnorm: {fnorm}")

    ##############
    # dae solution
    ##############
    # method = "BDF"
    method = "Radau"
    start = time.time()
    sol = solve_dae(F, t_span, vy0, vyp0, atol=atol, rtol=rtol, method=method)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    y = sol.y
    tp = t
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
    fig, ax = plt.subplots(2, 3)

    ax[0, 0].plot(t, y[0], "-k", label="x")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].plot(t, y[1], "-k", label="y")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[0, 2].plot(t, y[1], "-k", label="phi")
    ax[0, 2].grid()
    ax[0, 2].legend()

    ax[1, 0].plot(y[0], y[1], "-k", label="(x, y)")
    ax[1, 0].grid()
    ax[1, 0].legend()

    # ax.plot(t, np.exp(-t) + t * np.sin(t), "-r", label="y1 true")
    # ax.plot(t, np.sin(t), "-g", label="y2 true")

    # ax.grid()
    # ax.legend()

    plt.show()
