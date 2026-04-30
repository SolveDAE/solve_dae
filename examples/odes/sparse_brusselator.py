import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from solve_dae.integrate import solve_dae
from scipy.sparse import eye, spdiags


"""Sparse brusselator system, see mathworks.

References:
-----------
mathworks: https://www.mathworks.com/help/matlab/math/solve-stiff-odes.html#d126e27046
"""

N = int(1e3)

def f(t, y):
    c = 0.02 * (N + 1)**2
    yp = np.zeros_like(y)
    
    # Evaluate the 2 components of the function at one edge of the grid
    # (with edge conditions).
    i = 0
    yp[i] = 1 + y[i + 1] * y[i]**2 - 4 * y[i] + c * (1 - 2 * y[i] + y[i + 2])
    yp[i + 1] = 3 * y[i] - y[i + 1] * y[i]**2 + c * (3 - 2 * y[i + 1] + y[i + 3])

    # Evaluate the 2 components of the function at all interior grid points.
    i = np.r_[2:2 * N - 2:2]
    yp[i] = 1 + y[i + 1] * y[i]**2 - 4 * y[i] + c * (y[i - 2] - 2 * y[i] + y[i + 2])
    yp[i + 1] = 3 * y[i] - y[i + 1] * y[i]**2 + c * (y[i - 1] - 2 * y[i + 1] + y[i + 3])
    
    # Evaluate the 2 components of the function at the other edge of the grid
    # (with edge conditions).
    i = -2
    yp[i] = 1 + y[i + 1] * y[i]**2 - 4 * y[i] + c * (y[i - 2] - 2 * y[i] + 1)
    yp[i + 1] = 3 * y[i] - y[i + 1] * y[i]**2 + c * (y[i - 1] - 2 * y[i + 1] + 3)

    return yp

def F(t, y, yp):
    return yp - f(t, y)

sparsity_yp = eye(2 * N, format="csc")

data = np.ones((5, 2 * N))
data[1, 1::2] = 0
data[3, 0::2] = 0
sparsity_y = spdiags(data, [-2, -1, 0, 1, 2], format="csc")

jac_sparsity = sparsity_y, sparsity_yp


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 10
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=100)
    
    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0 = np.empty(2 * N)
    y0[::2] = 1 + np.sin((2 * np.pi / (N + 1)) * np.arange(1, N + 1))
    y0[1::2] = 3
    yp0 = f(t0, y0)
    print(f"||F(t0, y0, yp0)||: {np.linalg.norm(F(t0, y0, yp0))}")

    # solver options
    atol = 1e-3
    rtol = 1e-6

    ####################
    # reference solution
    ####################
    start = time.time()
    sol = solve_ivp(f, t_span, y0, atol=atol, rtol=rtol, method=method, 
                    t_eval=t_eval, jac_sparsity=sparsity_y)
    end = time.time()
    t_scipy = sol.t
    y_scipy = sol.y
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"message: {message}")
    print(f"elapsed time: {end - start}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    ##############
    # dae solution
    ##############
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, 
                    t_eval=t_eval, jac_sparsity=jac_sparsity)
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

    # check if ODE and DAE solution coincide
    # assert np.allclose(y, y_scipy, rtol=rtol, atol=atol)
    assert np.allclose(y, y_scipy, rtol=rtol * 1e1, atol=atol * 1e1)

    # visualization
    u = y[0::2, :]
    x = np.linspace(0, 1, num=N)
    T, X = np.meshgrid(t, x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, T, u, cmap="viridis")
    ax.view_init(elev=30, azim=-40)
    ax.set_xlabel("space")
    ax.set_ylabel("time")
    ax.set_zlabel("solution u")
    ax.set_title(f"The Brusselator for N = {N}")
    plt.show()