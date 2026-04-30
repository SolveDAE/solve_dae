import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae, consistent_initial_conditions


"""This example shows how to solve a stiff differential algebraic equation (DAE) 
that describes an electrical circuit, see [1]_ and [2]_. 

References
----------
.. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
        Stiff and Differential-Algebraic Problems", p. 377.
.. [2] https://www.mathworks.com/help/matlab/math/solve-stiff-transistor-dae.html
"""

# Problem parameters
Ub = 6
R0 = 1000
R = 9000
alpha = 0.99
beta = 1e-6
Uf = 0.026
Ue = lambda t: 0.4 * np.sin(200 * np.pi * t)  

# The constant, singular mass matrix
c = 1e-6 * np.arange(1, 4)
M = np.zeros((5, 5))
M[0, 0] = -c[0]
M[0, 1] =  c[0]
M[1, 0] =  c[0]
M[1, 1] = -c[0]
M[2, 2] = -c[1]
M[3, 3] = -c[2]
M[3, 4] =  c[2]
M[4, 3] =  c[2]
M[4, 4] = -c[2]

# Hairer and Wanner's RADAU5 requires consistent initial conditions
# which they take to be
u0 = np.zeros(5)
u0[1] = Ub / 2
u0[2] = Ub / 2
u0[3] = Ub

# Perturb the algebraic variables to test initialization.
u0[3] = u0[3] + 0.1
u0[4] = u0[4] + 0.1


def F(t, u, up):
    f23 = beta * (np.exp((u[1] - u[2]) / Uf) - 1)
    dudt = np.array([
        -(Ue(t) - u[0]) / R0,
        -(Ub / R - u[1] * 2 / R - (1 - alpha) * f23),
        -(f23 - u[2] / R),
        -((Ub - u[3]) / R - alpha * f23),
        u[4] / R,
    ])
    return M @ up - dudt


# time span
t0 = 0
t1 = 0.05
t_span = (t0, t1)

method = "Radau"
# method = "BDF" # alternative solver

# consistent initial conditions
up0 = np.zeros_like(u0)
print(f"u0: {u0}")
print(f"up0: {up0}")
u0, up0, fnorm = consistent_initial_conditions(F, t0, u0, up0)
print(f"u0: {u0}")
print(f"up0: {up0}")
print(f"fnorm: {fnorm}")

# solver options
atol = rtol = 1e-6

# dae solution
start = time.time()
sol = solve_dae(F, t_span, u0, up0, atol=atol, rtol=rtol, method=method)
end = time.time()
print(f"elapsed time: {end - start}")
t = sol.t
u = sol.y
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
fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_title("One transistor amplifier problem")
ax.plot(t, Ue(t), label="input voltage Ue(t)")
ax.plot(t, u[4], label="output voltage U5(t)")
ax.grid()
ax.legend()
plt.show()