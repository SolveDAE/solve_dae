import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from solve_dae.integrate import solve_dae, consistent_initial_conditions


"""One planar finite element of the total Lagrangian, objective and 
intrinsically locking-free Petrov-Galerkin SE(3) Cosserat rod finite element 
formulation proposed in Harsch2023. 

References:
-----------
Harsch2023: https://doi.org/10.1002/nme.7236
"""


EPS = 1e-6

STEEL = True
# STEEL = False

MOMENT = True
# MOMENT = False


def ax2skew(a: np.ndarray) -> np.ndarray:
    """Computes the skew symmetric matrix from a 3D vector."""
    assert a.size == 3
    return np.array([[0,    -a[2], a[1] ],
                     [a[2],  0,    -a[0]],
                     [-a[1], a[0], 0    ]], dtype=a.dtype)


def T_SO3(psi: np.ndarray) -> np.ndarray:
    angle2 = psi @ psi
    angle = np.sqrt(angle2)
    if angle > EPS:
        # Park2005 (19), actually its the transposed!
        sa = np.sin(angle)
        ca = np.cos(angle)
        psi_tilde = ax2skew(psi)
        alpha = sa / angle
        beta2 = (1.0 - ca) / angle2
        return (
            np.eye(3, dtype=float)
            - beta2 * psi_tilde
            + ((1.0 - alpha) / angle2) * psi_tilde @ psi_tilde
        )
    else:
        # first order approximation
        return np.eye(3, dtype=float) - 0.5 * ax2skew(psi)


def T_SO3_inv(psi: np.ndarray) -> np.ndarray:
    angle2 = psi @ psi
    angle = np.sqrt(angle2)
    psi_tilde = ax2skew(psi)
    if angle > EPS:
        # Park2005 (19), actually its the transposed!
        gamma = 0.5 * angle / (np.tan(0.5 * angle))
        return (
            np.eye(3, dtype=float)
            + 0.5 * psi_tilde
            + ((1.0 - gamma) / angle2) * psi_tilde @ psi_tilde
        )
    else:
        # first order approximation
        return np.eye(3, dtype=float) + 0.5 * psi_tilde


def smoothstep5(x, x_min=0, x_max=1):
    """5th-order smoothstep function, see https://en.wikipedia.org/wiki/Smoothstep."""
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return 6 * x**5 - 15 * x**4 + 10 * x**3


def cot(phi):
    """Cotangens, see https://de.wikipedia.org/wiki/Tangens_und_Kotangens."""
    return np.cos(phi) / np.sin(phi)


class SE3Rod:
    def __init__(self, scale_units=False):
        # geometry of the rod
        length = np.pi / 2
        width = 1e-2 # [m]
        density = 8.e3  # [kg / m^3]

        # material properties
        if STEEL:
            E = 260.0e9 # [N / m^2]
            G = 100.0e9 # [N / m^2]
        else:
            E = 0.5 * (0.1 - 0.01) * 1e9
            G = 0.0006 * 1e9

        # time span
        t0 = 0
        t1 = 2

        self.scale_units = scale_units
        if scale_units:
            # [m] -> [mm]
            length *= 1e3
            # cross section properties
            width *= 1e3
            density *= 1e-9

            # [s] -> [ms]
            t0 *= 1e3
            t1 *= 1e3

            # [N / m^2] -> [kN / mm^2]
            E *= 1e-9
            G *= 1e-9

        self.E = E
        self.G = G

        self.length = length
        self.width = width
        self.density = density

        A = width**2
        I = width**4 / 12

        self.A = A
        self.I = I

        self.EA = E * A
        self.GA = G * A
        self.EI = E * I

        self.K = np.diag([self.EA, self.GA, self.EI])
        self.M = np.diag([A, A, I]) * density * length / 3

        # initial conditions
        self.q0 = np.array([length, 0, 0], dtype=float)
        if not MOMENT:
            c = 0.125
            phi0 = c * 2 * np.pi
            self.q0 = np.array([length / phi0 * np.sin(phi0), length / phi0 * (1 - np.cos(phi0)), phi0], dtype=float)
        
        self.u0 = np.array([0, 0, 0], dtype=float)
        q_dot0 = self.u0
        u_dot0 = np.array([0, 0, 0], dtype=float)

        # reference strain measure
        self.epsilon0 = np.array([1, 0, 0], dtype=float)

        self.t0 = t0
        self.t1 = t1
        self.y0 = np.concatenate([self.q0, self.u0])
        self.yp0 = np.concatenate([q_dot0, u_dot0])

    def A_IK(self, xi, q):
        x, y, phi = q
        sphi, cphi = np.sin(xi * phi), np.cos(xi * phi)
        return np.array([
            [cphi, -sphi, 0],
            [sphi,  cphi, 0],
            [   0,     0, 1],
        ])

    def r_OP(self, xi, q):
        x, y, phi = q
        psi01 = np.array([0, 0, phi])
        psi = xi * psi01
        return (xi * T_SO3(psi).T @ T_SO3_inv(psi01).T @ np.array([x, y, 0]))[:2]

    def epsilon_bar(self, q):
        """Strain measures."""
        x, y, phi = q

        abs_phi = abs(phi)
        if abs_phi > EPS:
            g = 0.5 * abs_phi * cot(abs_phi / 2)
        else:
            # https://www.wolframalpha.com/input?i=taylor+series+abs%28x%29+cot%28abs%28x%29+%2F+2%29+%2F+2
            g = 1 - phi**2 / 12 - phi**4 / 720

        return np.array([
            x * g + 0.5 * phi * y,
            y * g - 0.5 * phi * x,
            phi,
        ])
    
    def W(self, q):
        x, y, phi = q

        if abs(phi) > EPS:
            sphi, cphi = np.sin(phi), np.cos(phi)
            diag = sphi / phi
            off_diag = (1 - cphi) / phi
        else:
            # https://www.wolframalpha.com/input?i=taylor+series+sin%28c+*+x%29+%2F+x
            diag = 1 - phi**2 / 6 + phi**4 / 120

            # https://www.wolframalpha.com/input?i=taylor+series+%281+-+cos%28c+*+x%29%29+%2F+x
            off_diag = phi / 2 - phi**3 / 24 + phi**5 / 720

        ga_x_bar, ga_y_bar, kappa_bar = self.epsilon_bar(q)
        
        return np.array([
            [          diag,       -off_diag, 0],
            [      off_diag,            diag, 0],
            [0.5 * ga_y_bar, -0.5 * ga_x_bar, 1],
        ])
        
    def la(self, q):
        return self.K @ (self.epsilon_bar(q) / self.length - self.epsilon0)

    def f_ext(self, t):
        if MOMENT:
            m_max = 2 * np.pi * self.EI / self.length * 0.49
            if t <= 0.5 * t1:
                m = m_max * smoothstep5(t, self.t0, 0.5 * self.t1)
            else:
                m = 0        
            return np.array([0, 0, m])
        else:
            return np.zeros(3)

    def F(self, t, y, yp):
        q, u = y[:3], y[3:]
        qp, up = yp[:3], yp[3:]

        return np.concatenate([
            qp - u,
            self.M @ up + self.W(q) @ self.la(q) - self.f_ext(t),
        ])


if __name__ == "__main__":
    # rod finite element
    scale_units = False
    # scale_units = True
    rod = SE3Rod(scale_units)
    y0 = rod.y0
    yp0 = rod.yp0
    t0 = rod.t0
    t1 = rod.t1
    length = rod.length

    F0 = rod.F(t0, y0, yp0)
    print(f"F0: {F0}")

    yp0 = np.zeros_like(y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, F0 = consistent_initial_conditions(rod.F, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"F0: {F0}")

    ##############
    # solver setup
    ##############
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e3))
    t_eval = None

    # method = "BDF"
    method = "Radau"

    atol = rtol = 1e-3

    ##############
    # dae solution
    ##############
    start = time.time()
    sol = solve_dae(rod.F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, stages=3)
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

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(5, 1)

    ax[0].set_xlabel("t")
    ax[0].set_xlabel("x")
    ax[0].grid()
    ax[0].plot(t, y[0], "-ok")

    ax[1].set_xlabel("t")
    ax[1].set_xlabel("y")
    ax[1].grid()
    ax[1].plot(t, y[1], "-ok")

    ax[2].set_xlabel("t")
    ax[2].set_xlabel("phi")
    ax[2].grid()
    ax[2].plot(t, y[2], "-ok")

    ax[3].set_xlabel("t")
    ax[3].set_xlabel("m(t)")
    ax[3].grid()
    ax[3].plot(t, np.array([rod.f_ext(ti)[-1] for ti in t]), "-ok")

    ax[4].set_xlabel("t")
    ax[4].set_xlabel("h")
    ax[4].grid()
    ax[4].plot(t[1:], np.diff(t), "-ok")
    ax[4].set_yscale("log")

    ###########
    # animation
    ###########

    # prepare data for animation
    frames = len(t)
    target_frames = min(frames, 100)
    frac = max(1, int(np.ceil(frames / target_frames)))
    animation_time = t1 / 2
    animation_time = 5
    interval = animation_time * 1000 / target_frames

    t = t[::frac]
    y = y[:, ::frac]

    target_frames = y.shape[1]

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5 * length, 1.5 * length)
    ax.set_ylim(-length, length)
    ax.set_aspect('equal')

    centerline, = ax.plot([], [], "-k")
    node, = ax.plot([], [], "or")
    rect = Rectangle((-0.1 * length, -0.25 * length), 0.1 * length, 0.5 * length, hatch='//', edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    xis = np.linspace(0, 1, num=100)

    def update(t, y):
        q, u = y[:3], y[3:]
        r_OP = np.array([rod.r_OP(xi, q) for xi in xis])
        node.set_data([0, q[0]], [0, q[1]])
        centerline.set_data(*r_OP.T)
        return centerline,

    def animate(i):
        update(t[i], y[:, i])

    anim = FuncAnimation(
        fig, animate, frames=target_frames, interval=interval, blit=False, repeat=True
    )

    plt.show()
