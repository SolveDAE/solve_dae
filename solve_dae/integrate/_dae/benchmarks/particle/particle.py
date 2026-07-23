import numpy as np
from solve_dae.integrate._dae.benchmarks.common import benchmark


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


def run_particle():
    # exponents
    # m_max = 10
    # m_max = 24
    m_max = 40
    ms = np.arange(m_max + 1)

    # tolerances and initial step size
    rtols = 10**(-(3 + ms / 4))
    atols = rtols
    h0s = 1e-2 * rtols

    # time span
    t0 = 0.0
    t1 = 2 * np.pi

    # initial conditions
    y0, yp0 = sol_true(t0)

    # reference solution
    y_ref, yp_ref = sol_true(t1)

    # x, y, u, v are the physical state; La, Mu are auxiliary integrated
    # states whose *derivatives* (yp[4], yp[5]) are the actual Lagrange
    # multipliers lambda, mu. Report state and multiplier errors separately
    # since they generally converge at different orders (see common.py's
    # benchmark() docstring), and lumping them into one norm would make
    # this comparable neither internally nor against a GGL/RADAU5
    # formulation, where lambda, mu are algebraic components of y.
    benchmark(
        t0, t1, y0, yp0, F, rtols, atols, h0s, "Particle", y_ref, yp_ref,
        y_idx=np.array([0, 1, 2, 3]),
        mult_idx=[4, 5],
        mult_names=["lambda", "mu"],
    )


if __name__ == "__main__":
    run_particle()
