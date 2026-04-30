import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae


"""
This example is a stiff system of 6 non-linear differential algebraic 
equations of index 1. The problem originates from Akzo Nobel Central research 
in Arnhern, The Netherlands, and describes a chemical process in which 2 
species are mixed, while carbon dioxide is continuously added, see [1]_.

References
----------
.. [1] https://archimede.uniba.it/~testset/problems/chemakzo.php
"""
# constants
k1 = 18.7
k2 = 0.58
k3 = 0.09
k4 = 0.42
kbig = 34.4
kla = 3.3
ks = 115.83
po2 = 0.9
hen = 737.0

# mass matrix
M = np.eye(6)
M[-1, -1] = 0

def rhs(t, y):
    r1 = k1 * (y[0] ** 4) * np.sqrt(y[1])
    r2 = k2 * y[2] * y[3]
    r3 = k2 / kbig * y[0] * y[4]
    r4 = k3 * y[0] * (y[3] ** 2)
    r5 = k4 * (y[5] ** 2) * np.sqrt(y[1])
    fin = kla * (po2 / hen - y[1])

    return np.array([
        -2.0 * r1 + r2 - r3 - r4,
        -0.5 * r1 - r4 - 0.5 * r5 + fin,
        r1 - r2 + r3,
        -r2 + r3 - 2.0 * r4,
        r2 - r3 + r5,
        ks * y[0] * y[3] - y[5],
    ])

def F(t, y, yp):
    return M @ yp - rhs(t, y)

def rhs_y(t, y):
    r11 = 4.0 * k1 * (y[0] ** 3) * np.sqrt(y[1])
    r12 = 0.5 * k1 * (y[0] ** 4) / np.sqrt(y[1])
    r23 = k2 * y[3]
    r24 = k2 * y[2]
    r31 = (k2 / kbig) * y[4]
    r35 = (k2 / kbig) * y[0]
    r41 = k3 * y[3] ** 2
    r44 = 2.0 * k3 * y[0] * y[3]
    r52 = 0.5 * k4 * (y[5] ** 2) / np.sqrt(y[1])
    r56 = 2.0 * k4 * y[5] * np.sqrt(y[1])
    fin2 = -kla

    jac = np.zeros((6, 6))

    jac[0, 0] = -2.0 * r11 - r31 - r41
    jac[0, 1] = -2.0 * r12
    jac[0, 2] = r23
    jac[0, 3] = r24 - r44
    jac[0, 4] = -r35

    jac[1, 0] = -0.5 * r11 - r41
    jac[1, 1] = -0.5 * r12 - 0.5 * r52 + fin2
    jac[1, 3] = -r44
    jac[1, 5] = -0.5 * r56

    jac[2, 0] = r11 + r31
    jac[2, 1] = r12
    jac[2, 2] = -r23
    jac[2, 3] = -r24
    jac[2, 4] = r35

    jac[3, 0] = r31 - 2.0 * r41
    jac[3, 2] = -r23
    jac[3, 3] = -r24 - 2.0 * r44
    jac[3, 4] = r35

    jac[4, 0] = -r31
    jac[4, 1] = r52
    jac[4, 2] = r23
    jac[4, 3] = r24
    jac[4, 4] = -r35
    jac[4, 5] = r56

    jac[5, 0] = ks * y[3]
    jac[5, 3] = ks * y[0]
    jac[5, 5] = -1.0

    return jac

def jac(t, y, yp):
    return -rhs_y(t, y), M


# time span
t0 = 0
t1 = 180
t_span = (t0, t1)

method = "Radau"
# method = "BDF" # alternative solver

# consistent initial conditions
y0 = np.array([
    0.444,
    0.00123,
    0.0, 
    0.007,
    0.0,
    ks * 0.444 * 0.007
])
yp0 = np.zeros_like(y0)
yp0 = rhs(t0, y0)
print(f"u0: {y0}")
print(f"up0: {yp0}")
print(f"fnorm: {np.linalg.norm(F(t0, y0, yp0))}")

# solver options
atol = 1e-6
rtol = 1e-10

# dae solution
start = time.time()
sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, jac=jac)
end = time.time()
print(f"elapsed time: {end - start}")
t = sol.t
y = sol.y
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
rows = 3
cols = 2
fig, ax = plt.subplots(rows, cols)

for i in range(6):
    ii = i // cols
    jj = i % cols

    ax[ii, jj].plot(t, y[i], label=f"y{i}")
    ax[ii, jj].grid()
    ax[ii, jj].legend()

plt.show()