import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae

m = 1.25
Theta = 0.13
a = 0.1
g = 9.81
np.random.seed(1234)
Omega = 2 * np.random.rand(1)[0]
alpha = np.random.rand(1)[0]
salpha = np.sin(alpha)

"""Index 2 DAE found in Section 1.6 of Bloch2015.

References:
-----------
Bloch2015: https://doi.org/10.1007/978-1-4939-3017-3
"""
def F(t, vy, vyp):
    x, y, phi, u, v, omega, _ = vy
    xp, yp, phip, up, vp, omegap, lap = vyp

    sphi, cphi = np.sin(phi), np.cos(phi)

    F = np.zeros_like(vy, dtype=np.common_type(vy, vyp))

    # Bloch 2005, equation (1.7.6)
    F[0] = xp - u
    F[1] = yp - v
    F[2] = phip - omega
    F[3] = m * up - m * g * salpha - sphi * lap
    F[4] = m * vp + cphi * lap 
    F[5] = Theta * omegap
    F[6] = u * sphi - v * cphi

    return F

def sol_true(t):
    x = (g * salpha / (2 * Omega**2)) * np.sin(Omega * t)**2
    y = (g * salpha / (2 * Omega**2)) * (Omega * t - 0.5 * np.sin(2 * Omega * t))
    phi = Omega * t
    
    u =  (g * salpha / Omega) * np.sin(Omega * t) * np.cos(Omega * t)
    # v = (g * salpha / (2 * Omega)) * (1 - np.cos(2 * Omega * t))
    v = (g * salpha / Omega) * np.sin(Omega * t)**2
    omega = Omega * np.ones_like(t)
    
    La = (2 * m * g * salpha / Omega) * (np.cos(Omega * t) - 1)
    La_dot = -2 * m * g * salpha * np.sin(Omega * t)

    x_dot = u
    y_dot = v
    phi_dot = omega

    u_dot = g * salpha + np.sin(Omega * t) * La_dot / m
    v_dot = - np.cos(Omega * t) * La_dot / m
    omega_dot = np.zeros_like(t)

    vy = np.array([
        x,
        y,
        phi,
        u,
        v,
        omega,
        La,
    ])

    vyp = np.array([
        x_dot,
        y_dot,
        phi_dot,
        u_dot,
        v_dot,
        omega_dot,
        La_dot,
    ])

    return vy, vyp


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = (np.pi / Omega) * 2
    t_span = (t0, t1)

    # tolerances
    rtol = atol = 1e-5

    # initial conditions
    vy0, vyp0 = sol_true(t0)

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
    fig, ax = plt.subplots(3, 3)

    y_true, yp_true = sol_true(t)

    ax[0, 0].plot(t, y_true[0], "-ok", label="x_true")
    ax[0, 0].plot(t, y[0], "--xr", label="x")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].plot(t, y_true[1], "-ok", label="y_true")
    ax[0, 1].plot(t, y[1], "--xr", label="y")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[0, 2].plot(t, y_true[2], "-ok", label="phi_true")
    ax[0, 2].plot(t, y[2], "--xr", label="phi")
    ax[0, 2].grid()
    ax[0, 2].legend()

    ax[1, 0].plot(t, y_true[3], "-ok", label="u_true")
    ax[1, 0].plot(t, y[3], "--xr", label="u")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[1, 1].plot(t, y_true[4], "-ok", label="v_true")
    ax[1, 1].plot(t, y[4], "--xr", label="v")
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[1, 2].plot(t, y_true[5], "-ok", label="omega_true")
    ax[1, 2].plot(t, y[5], "--xr", label="omega")
    ax[1, 2].grid()
    ax[1, 2].legend()

    ax[2, 0].plot(t, y_true[6], "-ok", label="La_true")
    ax[2, 0].plot(t, y[6], "--xr", label="La")
    ax[2, 0].grid()
    ax[2, 0].legend()


    ax[2, 1].plot(t, yp_true[6], "-ok", label="Lap_true")
    ax[2, 1].plot(t, yp[6], "--xr", label="Lap")
    ax[2, 1].grid()
    ax[2, 1].legend()

    ax[2, 2].plot(y_true[0], y_true[1], "-ok", label="(x_true, y_true)")
    ax[2, 2].plot(y[0], y[1], "--xr", label="(x, y)")
    ax[2, 2].grid()
    ax[2, 2].legend()
    ax[2, 2].axis('equal')

    plt.show()
