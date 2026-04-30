import numpy as np
from solve_dae.integrate._dae.benchmarks.common import benchmark


m = 1.25
Theta = 0.13
g = 9.81
Omega = np.pi / 3
alpha = np.pi / 4
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

    # Bloch 2015 - Section 1.6
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
    v = (g * salpha / Omega) * np.sin(Omega * t)**2
    omega = Omega * np.ones_like(t)
    
    La = (2 * m * g * salpha / Omega) * (np.cos(Omega * t) - 1)
    La_dot = -2 * m * g * salpha * np.sin(Omega * t)

    x_dot = u
    y_dot = v
    phi_dot = omega

    u_dot = g * salpha + np.sin(Omega * t) * La_dot / m
    v_dot = - np.cos(Omega * t) * La_dot / m
    omega_dot = 0

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


def run_knife_edge():
    # exponents
    m_max = 32
    ms = np.arange(m_max + 1)

    # tolerances and initial step size
    rtols = 10**(-(1 + ms / 4))
    atols = rtols
    h0s = 1e-2 * rtols

    # time span
    t0 = 0
    t1 = (np.pi / Omega) * 2

    # initial conditions
    vy0, vyp0 = sol_true(t0)

    # reference solution
    y_ref = sol_true(t1)[0]

    benchmark(t0, t1, vy0, vyp0, F, rtols, atols, h0s, "knife_edge", y_ref)


if __name__ == "__main__":
    run_knife_edge()
