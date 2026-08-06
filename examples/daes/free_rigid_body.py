import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae

"""Torque-free rotation of an axisymmetric rigid body described by a unit
quaternion p(t) and the body-fixed angular velocity Omega(t), with the
unit-length constraint stabilized by an integrated multiplier nu(t)
(Hiller-type formulation). This is a genuine index-2 DAE: differentiating
the constraint g(p) = 0 once yields nu_dot, which is only fixed by solving
the whole system together with the kinematic and dynamic equations.

Both Omega(t) and p(t) possess a closed-form solution for an axisymmetric
body (Theta = diag(A, A, C)): Omega performs a uniform rotation about the
body's own symmetry axis, and the attitude is the exact composition of that
same rotation with a regular precession about the (constant) angular
momentum vector, see Rucker2018 for the quaternion kinematic map H(p).

This example only investigates the long-term behavior of the *adaptive*
Radau IIA integrator: energy, angular momentum and the unit-length
constraint are exactly conserved along the true solution, so their drift
over a long, many-period integration is a direct measure of the scheme's
long-term error -- the whole point of using an adaptive step-size code
rather than a fixed step. See knife_edge.py / pendulum.py for the analogous
convergence-style DAE examples in this repository.

References:
-----------
Rucker2018: https://rucker.io/quaternions.pdf
"""

# axisymmetric inertia tensor Theta = diag(A, A, C)
A = 2.0
C = 1.0
Theta = np.diag([A, A, C])

p0 = np.array([1.0, 0.0, 0.0, 0.0])
Omega0 = np.array([1.0, 1.0, 1.0])
nu0 = 0.0

# rotation rate of Omega about the body-fixed symmetry axis e3
lambda_ = (A - C) * Omega0[2] / A

# constant angular momentum direction (body frame at t=0) and the rate at
# which the body precesses about it
e3 = np.array([0.0, 0.0, 1.0])
L0 = Theta @ Omega0
ell = L0 / np.linalg.norm(L0)
Omega_pr = np.linalg.norm(L0) / A


def ax2skew(a):
    """Computes the skew symmetric matrix from a 3D vector."""
    # fmt: off
    return np.array([[0,     -a[2],  a[1]],
                      [a[2],   0,   -a[0]],
                      [-a[1],  a[0],  0  ]], dtype=a.dtype)
    # fmt: on


def H(p):
    """Quaternion kinematic map H(p) in R^{4x3} satisfying p^T H(p) = 0,
    see Rucker2018."""
    p0c, pv = p[0], p[1:]
    return np.vstack((-pv[None, :], p0c * np.eye(3, dtype=p.dtype) + ax2skew(pv)))


def F(t, vy, vyp):
    p0c, p1c, p2c, p3c, Omega1, Omega2, Omega3, nu = vy
    p0cp, p1cp, p2cp, p3cp, Omega1p, Omega2p, Omega3p, nu_dot = vyp

    p = np.array([p0c, p1c, p2c, p3c])
    p_dot = np.array([p0cp, p1cp, p2cp, p3cp])
    Omega = np.array([Omega1, Omega2, Omega3])
    Omega_dot = np.array([Omega1p, Omega2p, Omega3p])

    R = np.zeros(8, dtype=np.common_type(vy, vyp))
    R[:4] = p_dot - (0.5 * H(p) @ Omega + p * nu_dot)
    R[4:7] = Theta @ Omega_dot + ax2skew(Omega) @ Theta @ Omega
    R[7] = 0.5 * (p @ p - 1.0)
    return R


def qmul(p, q):
    """Quaternion product p (x) q, vectorized over the trailing axis."""
    p0c, p1c, p2c, p3c = p
    q0, q1, q2, q3 = q
    return np.array([
        p0c * q0 - p1c * q1 - p2c * q2 - p3c * q3,
        p0c * q1 + p1c * q0 + p2c * q3 - p3c * q2,
        p0c * q2 - p1c * q3 + p2c * q0 + p3c * q1,
        p0c * q3 + p1c * q2 - p2c * q1 + p3c * q0,
    ])


def q_axis_angle(axis, angle):
    """Unit quaternion representing a rotation by `angle` about the fixed
    `axis`."""
    c, s = np.cos(0.5 * angle), np.sin(0.5 * angle)
    return np.array([c, s * axis[0], s * axis[1], s * axis[2]])


def _p_dot(p, Omega):
    """0.5 * H(p) @ Omega, for a single quaternion/vector pair (p.ndim == 1)
    or a whole trajectory (p.shape == (4, n), Omega.shape == (3, n))."""
    if p.ndim == 1:
        return 0.5 * H(p) @ Omega
    return 0.5 * np.stack([H(p[:, i]) @ Omega[:, i] for i in range(p.shape[1])], axis=1)


def sol_true(t):
    """Closed-form solution, exact for arbitrarily long time horizons: no
    numerical reference integration is used anywhere in this example. Omega
    performs a uniform rotation about e3 at rate `lambda_`; the attitude is
    the composition p(0) (x) q_ell(Omega_pr * t) (x) q_e3(lambda_ * t) of the
    body's own spin with the regular precession about the constant angular
    momentum direction `ell`."""
    t = np.asarray(t, dtype=float)

    c, s = np.cos(lambda_ * t), np.sin(lambda_ * t)
    Omega1 = c * Omega0[0] + s * Omega0[1]
    Omega2 = -s * Omega0[0] + c * Omega0[1]
    Omega3 = Omega0[2] * np.ones_like(t)
    Omega = np.array([Omega1, Omega2, Omega3])

    Omega1_dot = lambda_ * Omega2
    Omega2_dot = -lambda_ * Omega1
    Omega3_dot = np.zeros_like(t)

    p = qmul(qmul(p0, q_axis_angle(ell, Omega_pr * t)), q_axis_angle(e3, lambda_ * t))
    p_dot = _p_dot(p, Omega)

    nu = np.zeros_like(t)
    nu_dot = np.zeros_like(t)

    vy = np.array([p[0], p[1], p[2], p[3], Omega1, Omega2, Omega3, nu])
    vyp = np.array([p_dot[0], p_dot[1], p_dot[2], p_dot[3], Omega1_dot, Omega2_dot, Omega3_dot, nu_dot])
    return vy, vyp


if __name__ == "__main__":
    ############
    # parameters
    ############
    t0 = 0.0
    # t1 = 1000.0  # long horizon: O(150) precession periods
    t1 = 100.0
    t_span = (t0, t1)

    atol = rtol = 1e-3

    y0, yp0 = sol_true(t0)
    assert np.allclose(F(t0, y0, yp0), 0.0)

    ##############
    # dae solution
    ##############
    method = "Radau"
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, method=method, atol=atol, rtol=rtol)
    elapsed_time = time.time() - start
    print(sol)
    assert sol.success

    stat_names = ["nstep", "naccpt", "nrejct", "nfev", "njev", "nlu", "nlusolve"]
    stats = [getattr(sol, name) for name in stat_names]
    print(f"elapsed time: {elapsed_time:.2f} s")
    print(f"stats ({', '.join(stat_names)}): {stats}")

    t = sol.t
    y = sol.y
    yp = sol.yp
    h = np.diff(t)

    ##########################################
    # invariants and error against sol_true(t)
    ##########################################
    p, Omega, nu = y[:4], y[4:7], y[7]
    y_true, yp_true = sol_true(t)

    energy0 = Omega0 @ Theta @ Omega0
    momentum0 = np.linalg.norm(Theta @ Omega0)

    kinetic_energy = np.einsum("in,ij,jn->n", Omega, Theta, Omega)
    angular_momentum = np.linalg.norm(Theta @ Omega, axis=0)
    g = 0.5 * (np.einsum("in,in->n", p, p) - 1.0)

    err_energy = np.abs(kinetic_energy - energy0)
    err_momentum = np.abs(angular_momentum - momentum0)
    err_g = np.abs(g)
    err_state = np.linalg.norm(y - y_true, axis=0)

    print(f"energy 2T:        max={err_energy.max():.3e}  mean={err_energy.mean():.3e}")
    print(f"angular momentum: max={err_momentum.max():.3e}  mean={err_momentum.mean():.3e}")
    print(f"constraint g(p):  max={err_g.max():.3e}  mean={err_g.mean():.3e}")
    print(f"state error:      max={err_state.max():.3e}  mean={err_state.mean():.3e}")

    ########
    # export
    ########
    header = (
        "t, "
        "p0, p1, p2, p3, Omega1, Omega2, Omega3, nu, "
        "p0_true, p1_true, p2_true, p3_true, Omega1_true, Omega2_true, Omega3_true, nu_true, "
        "err_energy, err_momentum, err_g, err_state"
    )
    data = np.column_stack((
        t, y.T, y_true.T, err_energy, err_momentum, err_g, err_state,
    ))
    np.savetxt(
        f"free_rigid_body_long_term_tol_{atol}.txt",
        data,
        header=header,
        delimiter=", ",
        comments="",
    )

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    ax[0, 0].set_title("angular velocity Omega: adaptive vs. closed form")
    for i in range(3):
        ax[0, 0].plot(t, Omega[i], "-", label=f"Omega{i + 1}")
        ax[0, 0].plot(t, y_true[4 + i], "--", label=f"Omega{i + 1} true")
    ax[0, 0].set_xlabel("t")
    ax[0, 0].grid()
    ax[0, 0].legend(fontsize="small")

    ax[0, 1].set_title("quaternion p: adaptive vs. closed form")
    for i in range(4):
        ax[0, 1].plot(t, p[i], "-", label=f"p{i}")
        ax[0, 1].plot(t, y_true[i], "--", label=f"p{i} true")
    ax[0, 1].set_xlabel("t")
    ax[0, 1].grid()
    ax[0, 1].legend(fontsize="small", ncol=2)

    ax[0, 2].set_title("adaptive step size")
    ax[0, 2].plot(t[1:], h, "-k")
    ax[0, 2].set_xlabel("t")
    ax[0, 2].set_ylabel("h")
    ax[0, 2].set_yscale("log")
    ax[0, 2].grid()

    ax[1, 0].set_title("invariant errors")
    ax[1, 0].plot(t, err_energy, label=r"$|2T - 2T_0|$")
    ax[1, 0].plot(t, err_momentum, label=r"$|\Vert\Theta\Omega\Vert - \Vert\Theta\Omega_0\Vert|$")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_yscale("log")
    ax[1, 0].grid()
    ax[1, 0].legend(fontsize="small")

    ax[1, 1].set_title("unit-length constraint")
    ax[1, 1].plot(t, err_g, "-k", label=r"$|g(p)|$")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].set_yscale("log")
    ax[1, 1].grid()
    ax[1, 1].legend(fontsize="small")

    ax[1, 2].set_title("state error vs. closed form")
    ax[1, 2].plot(t, err_state, "-k", label=r"$\Vert y - y_\mathrm{true}\Vert$")
    ax[1, 2].set_xlabel("t")
    ax[1, 2].set_yscale("log")
    ax[1, 2].grid()
    ax[1, 2].legend(fontsize="small")

    plt.tight_layout()
    plt.show()
