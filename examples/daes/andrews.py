import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae, consistent_initial_conditions


"""This example investigates a flexible multibody system called Andrews' 
squeezer mechanism as outlined in [1]_ and [2]_. Since this is a system of 
differential algebraic equatons of index 3, we implement a stabilized index 1 
formulation as proposed by [3]_.

References:
-----------
.. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
        Stiff and Differential-Algebraic Problems", p. 377.
.. [2] https://archimede.uniba.it/~testset/problems/andrews.php
.. [3] https://doi.org/10.1002/nme.1620320803
"""

# parameters
nq = 7
nla = 6

m1 = 0.04325
m2 = 0.00365
m3 = 0.02373
m4 = 0.00706
m5 = 0.07050
m6 = 0.00706
m7 = 0.05498

I1 = 2.194e-6
I2 = 4.410e-7
I3 = 5.255e-6
I4 = 5.667e-7
I5 = 1.169e-5
I6 = 5.667e-7
I7 = 1.912e-5

xa = -0.06934
ya = -0.00227

xb = -0.03635
yb = 0.03273

xc = 0.014
yc = 0.072

d = 0.028
da = 0.0115
e = 0.02

ea = 0.01421
zf = 0.02
fa = 0.01421

rr = 0.007
ra = 0.00092
ss = 0.035

sa = 0.01874
sb = 0.01043
sc = 0.018

sd = 0.02
zt = 0.04
ta = 0.02308

tb = 0.00916
u = 0.04
ua = 0.01228

ub = 0.00449
c0 = 4530
l0 = 0.07785

mom = 0.033

def F(t, y, yp):

    def M(t, vq):
        M = np.zeros((nq, nq))

        # beta, Theta, gamma, Phi, delta, Omega, epsilon
        q1, q2, q3, q4, q5, q6, q7 = vq

        cos_Th = np.cos(q2)
        sin_Phi = np.sin(q4)
        sin_Om = np.sin(q6)

        M[0, 0] = m1 * ra**2 + m2 * (rr**2 - 2 * da * rr * cos_Th + da**2) + I1 + I2
        M[1, 0] = M[0, 1] = m2 * (da**2 - da * rr * cos_Th) + I2
        M[1, 1] = m2 * da**2 + I2
        M[2, 2] = m3 * (sa**2 + sb**2) + I3
        M[3, 3] = m4 * (e - ea)**2 + I4
        M[4, 3] = M[3, 4] = m4 * ((e - ea)**2 + zt * (e - ea) * sin_Phi) + I4
        M[4, 4] = m4 * (zt**2 + 2 * zt * (e - ea) * sin_Phi + (e - ea)**2) + m5 * (ta**2 + tb**2) + I4 + I5
        M[5, 5] = m6 * (zf - fa)**2 + I6
        M[6, 5] = M[5, 6] = m6 * ((zf - fa)**2 - u * (zf - fa) * sin_Om) + I6
        M[6, 6] = m6 * ((zf - fa)**2 - 2 * u * (zf - fa) * sin_Om + u**2) + m7 * (ua**2 + ub**2) + I6 + I7

        return M

    def h(t, vq, vu):
        # beta, Theta, gamma, Phi, delta, Omega, epsilon
        q1, q2, q3, q4, q5, q6, q7 = vq
        u1, u2, u3, u4, u5, u6, u7 = vu

        sin_Th = np.sin(q2)
        sin_ga = np.sin(q3)
        cos_ga = np.cos(q3)
        cos_Phi = np.cos(q4)
        cos_Om = np.cos(q6)

        xd = sd * cos_ga + sc * sin_ga + xb
        yd = sd * sin_ga - sc * cos_ga + yb
        L = np.sqrt((xd - xc)**2 + (yd - yc)**2)
        F = -c0 * (L - l0) / L
        Fx = F * (xd - xc)
        Fy = F * (yd - yc)

        h = np.zeros(nq)
        h[0] = mom - m2 * da * rr * u2 * (u2 + 2 * u1) * sin_Th
        h[1] = m2 * da * rr * u1**2 * sin_Th
        h[2] = Fx * (sc * cos_ga - sd * sin_ga) + Fy * (sd * cos_ga + sc * sin_ga)
        h[3] = m4 * zt * (e - ea) * u5**2 * cos_Phi
        h[4] = -m4 * zt * (e - ea) * u4 * (u4 + 2 * u5) * cos_Phi
        h[5] = -m6 * u * (zf - fa) * u7**2 * cos_Om
        h[6] = m6 * u * (zf - fa) * u6 * (u6 + 2 * u7) * cos_Om

        return h

    def g(t, vq):
        # beta, Theta, gamma, Phi, delta, Omega, epsilon
        q1, q2, q3, q4, q5, q6, q7 = vq

        sin_be = np.sin(q1)
        cos_be = np.cos(q1)
        sin_ga = np.sin(q3)
        cos_ga = np.cos(q3)

        sin_be_Th = np.sin(q1 + q2)
        cos_be_Th = np.cos(q1 + q2)
        sin_Ph_de = np.sin(q4 + q5)
        cos_Ph_de = np.cos(q4 + q5)
        sin_Om_ep = np.sin(q6 + q7)
        cos_Om_ep = np.cos(q6 + q7)

        g = np.zeros(nla)
        g[0] = rr * cos_be - d * cos_be_Th - ss * sin_ga - xb
        g[1] = rr * sin_be - d * sin_be_Th + ss * cos_ga - yb
        g[2] = rr * cos_be - d * cos_be_Th - e * sin_Ph_de - zt * np.cos(q5) - xa
        g[3] = rr * sin_be - d * sin_be_Th + e * cos_Ph_de - zt * np.sin(q5) - ya
        g[4] = rr * cos_be - d * cos_be_Th - zf * cos_Om_ep - u * np.sin(q7) - xa
        g[5] = rr * sin_be - d * sin_be_Th - zf * sin_Om_ep + u * np.cos(q7) - ya
        return g
    
    def g_q(t, vq):
        # beta, Theta, gamma, Phi, delta, Omega, epsilon
        q1, q2, q3, q4, q5, q6, q7 = vq

        sin_be = np.sin(q1)
        cos_be = np.cos(q1)
        sin_ga = np.sin(q3)
        cos_ga = np.cos(q3)

        sin_be_Th = np.sin(q1 + q2)
        cos_be_Th = np.cos(q1 + q2)
        sin_Ph_de = np.sin(q4 + q5)
        cos_Ph_de = np.cos(q4 + q5)
        sin_Om_ep = np.sin(q6 + q7)
        cos_Om_ep = np.cos(q6 + q7)

        g_q = np.zeros((nla, nq))

        g_q[0, 0] = -rr * sin_be + d * sin_be_Th
        g_q[0, 1] = d * sin_be_Th
        g_q[0, 2] = -ss * cos_ga

        g_q[1, 0] = rr * cos_be - d * cos_be_Th
        g_q[1, 1] = - d * cos_be_Th
        g_q[1, 2] = -ss * sin_ga

        g_q[2, 0] = -rr * sin_be + d * sin_be_Th
        g_q[2, 1] = d * sin_be_Th
        g_q[2, 3] = -e * cos_Ph_de
        g_q[2, 4] = -e * cos_Ph_de + zt * np.sin(q5)

        g_q[3, 0] = rr * cos_be - d * cos_be_Th
        g_q[3, 1] = - d * cos_be_Th
        g_q[3, 3] = -e * sin_Ph_de
        g_q[3, 4] = -e * sin_Ph_de - zt * np.cos(q5)

        g_q[4, 0] = -rr * sin_be + d * sin_be_Th
        g_q[4, 1] = d * sin_be_Th
        g_q[4, 5] = zf * sin_Om_ep
        g_q[4, 6] = zf * sin_Om_ep - u * np.cos(q7)

        g_q[5, 0] = rr * cos_be - d * cos_be_Th
        g_q[5, 1] = -d * cos_be_Th
        g_q[5, 5] = -zf * cos_Om_ep
        g_q[5, 6] = -zf * cos_Om_ep - u * np.sin(q7)

        return g_q
   
    vq, vu, _, _ = np.split(y, [nq, 2 * nq, 2 * nq + nla])
    vq_dot, vu_dot, vla, vmu = np.split(yp, [nq, 2 * nq, 2 * nq + nla])

    g_q = g_q(t, vq)

    # # index 1
    # return np.concatenate([
    #     vq_dot - vu,
    #     M(t, vq) @ vu_dot - h(t, vq, vu) - g_q.T @ vla,
    #     g_q @ vu,
    #     vmu,
    # ])

    # stabilized index 1
    return np.concatenate([
        vq_dot - vu + g_q.T @ vmu,
        M(t, vq) @ vu_dot - h(t, vq, vu) - g_q.T @ vla,
        g_q @ vu,
        g(t, vq),
    ])


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 0.03
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e3))

    # tolerances
    rtol = atol = 1e-4

    # initial conditions
    q0 = np.array([
        -0.0617138900142764496358948458001,
        0,
        0.455279819163070380255912382449,
        0.222668390165885884674473185609,
        0.487364979543842550225598953530,
        -0.222668390165885884674473185609,
        1.23054744454982119249735015568,
    ])
    u0 = np.zeros(nq)
    la0 = np.zeros(nla)
    mu0 = np.zeros(nla)

    qp0 = u0.copy()
    up0 = np.array([
        14222.4439199541138705911625887,
        -10666.8329399655854029433719415,
        0,
        0,
        0,
        0,
        0,
    ])
    lap0 = np.array([
        -98.5668703962410896057654982170,
        6.12268834425566265503114393122,
        0,
        0,
        0,
        0,
    ])
    mup0 = np.zeros(nla)

    y0 = np.concatenate([q0, u0, la0, mu0])
    yp0 = np.concatenate([qp0, up0, lap0, mup0])
    F0 = F(t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    # print(f"F0 = {F0}")
    print(f"||F0|| = {np.linalg.norm(F0)}")
    # exit()
    # y0, yp0, fnorm = consistent_initial_conditions(F, t0, y0, yp0)
    # print(f"y0: {y0}")
    # print(f"yp0: {yp0}")
    # print(f"fnorm: {fnorm}")
    # # exit()

    ##############
    # dae solution
    ##############
    start = time.time()
    # method = "BDF"
    method = "Radau"
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, stages=3)
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
    rows = 4
    cols = 2
    fig, ax = plt.subplots(rows, cols)

    for i in range(7):
        ii = i // cols  # Row index
        jj = i % cols   # Column index

        yi = y[i]
        yi = (yi + np.pi) % (2 * np.pi) - np.pi
        ax[ii, jj].plot(t, yi, label=f"y{i} mod(2π)")
        ax[ii, jj].grid()
        ax[ii, jj].legend()

    plt.show()
