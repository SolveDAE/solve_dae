import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae


solvers = [
    ("Radau", {"stages": 3}),
    ("Radau", {"stages": 5}),
    ("Radau", {"stages": 7}),
    ("BDF", {"NDF_strategy": "stability"}),
    ("BDF", {"NDF_strategy": "accuracy"}),
    ("BDF", {"NDF_strategy": None}),
]


def benchmark(t0, t1, y0, yp0, F, rtols, atols, h0s, name, y_ref=None, y_idx=None):
    # time span
    t_span = (t0, t1)

    # benchmark results
    results = np.zeros((len(solvers), len(rtols), 2))

    if y_ref is None:
        sol = solve_dae(
            F, 
            t_span, 
            y0, 
            yp0, 
            atol=1e-14, 
            rtol=1e-14,  
            method="Radau",
            stages=5,
        )
        y_ref = sol.y[:, -1]
        print(sol)
        assert sol.success

    for i, method_and_kwargs in enumerate(solvers):
        method, kwargs = method_and_kwargs
        print(f" - method: {method}; kwargs: {kwargs}")
        for j, (rtol, atol, h0) in enumerate(zip(rtols, atols, h0s)):
            print(f"   * rtol: {rtol}")
            print(f"   * atol: {atol}")
            print(f"   * h0:   {h0}")

            # solve system
            start = time.time()
            sol = solve_dae(
                F, 
                t_span, 
                y0, 
                yp0, 
                atol=atol, 
                rtol=rtol, 
                method=method, 
                first_step=h0,
                **kwargs,
            )
            end = time.time()
            elapsed_time = end - start
            print(f"     => sol: {sol}")
            assert sol.success

            # error
            if y_idx is not None:
                diff = y_ref[y_idx] - sol.y[y_idx, -1]
            else:
                diff = y_ref - sol.y[:, -1]
            error = np.linalg.norm(diff)
            print(f"     => error: {error}")

            results[i, j] = (error, elapsed_time)

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, ri in enumerate(results):
        ax.plot(ri[:, 0], ri[:, 1], label=solvers[i])

    if name == "Brenan":
        result_IDA = np.loadtxt("solve_dae/integrate/_dae/benchmarks/brenan/brenan_errors_IDA.csv", delimiter=',')
        result_IDA[:, 1] *= 100 # scale elapsed time by 100
        ax.plot(*result_IDA.T, label="sundials IDA (elapsed time *= 100)")
    elif name == "Robertson":
        result_IDA = np.loadtxt("solve_dae/integrate/_dae/benchmarks/robertson/robertson_errors_IDA.csv", delimiter=',')
        result_IDA[:, 1] *= 100 # scale elapsed time by 100
        ax.plot(*result_IDA.T, label="sundials IDA (elapsed time *= 100)")
    elif name == "Knife edge":
        result_IDA = np.loadtxt("solve_dae/integrate/_dae/benchmarks/knife_edge/knife_edge_errors_IDA.csv", delimiter=',')
        result_IDA[:, 1] *= 100 # scale elapsed time by 100
        ax.plot(*result_IDA.T, label="sundials IDA (elapsed time *= 100)")
    elif name == "Arevalo":
        result_IDA = np.loadtxt("solve_dae/integrate/_dae/benchmarks/arevalo/arevalo_errors_IDA.csv", delimiter=',')
        result_IDA[:, 1] *= 100 # scale elapsed time by 100
        ax.plot(*result_IDA.T, label="sundials IDA (elapsed time *= 100)")
    elif name == "Weissinger":
        result_IDA = np.loadtxt("solve_dae/integrate/_dae/benchmarks/weissinger/weissinger_errors_IDA.csv", delimiter=',')
        result_IDA[:, 1] *= 500 # scale elapsed time by 500
        ax.plot(*result_IDA.T, label="sundials IDA (elapsed time *= 500)")

    ax.set_title(f"work-precision: {name}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()
    ax.set_xlabel("||y_ref(t1) - y(t1)||")
    ax.set_ylabel("elapsed time [s]")

    plt.savefig(f"data/img/{name}_work_precision.png", dpi=300)

    plt.show()
