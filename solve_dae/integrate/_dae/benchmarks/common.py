import os
import time
import numpy as np
import matplotlib.pyplot as plt
from solve_dae.integrate import solve_dae


# Resolve paths relative to this file rather than the caller's current
# working directory, since these benchmark scripts may be run either
# directly from within their own subdirectory or via run_benchmarks.py.
_BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCHMARKS_DIR, "..", "..", "..", ".."))


solvers = [
    # ("Radau", {"stages": 3}),
    # ("Radau", {"stages": 5}),
    # ("Radau", {"stages": 7}),
    ("BDF", {"NDF_strategy": "stability"}),
    ("BDF", {"NDF_strategy": "accuracy"}),
    ("BDF", {"NDF_strategy": None}),
]


def benchmark(t0, t1, y0, yp0, F, rtols, atols, h0s, name, y_ref=None, yp_ref=None, y_idx=None):
    # time span
    t_span = (t0, t1)

    # benchmark results
    results = np.zeros((len(solvers), len(rtols), 3))

    if y_ref is None or yp_ref is None:
        # Some benchmark problems only supply a literature-quoted y_ref
        # (e.g. Robertson, which has no closed-form yp), so compute
        # whichever of the two is still missing from a high-accuracy run.
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
        if y_ref is None:
            y_ref = sol.y[:, -1]
        if yp_ref is None:
            yp_ref = sol.yp[:, -1]
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
                diff_y = y_ref[y_idx] - sol.y[y_idx, -1]
                diff_yp = yp_ref[y_idx] - sol.yp[y_idx, -1]
            else:
                diff_y = y_ref - sol.y[:, -1]
                diff_yp = yp_ref - sol.yp[:, -1]
            error_y = np.linalg.norm(diff_y)
            print(f"     => error_y: {error_y}")
            error_yp = np.linalg.norm(diff_yp)
            print(f"     => error_yp: {error_yp}")

            results[i, j] = (elapsed_time, error_y, error_yp)

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, ri in enumerate(results):
        ax.plot(ri[:, 0], ri[:, 1], label=solvers[i])

    def _ida_csv(subdir, filename):
        return os.path.join(_BENCHMARKS_DIR, subdir, filename)

    if name == "Brenan":
        result_IDA = np.loadtxt(_ida_csv("brenan", "brenan_errors_IDA.csv"), delimiter=',')
        result_IDA[:, 1] *= 100 # scale elapsed time by 100
        ax.plot(*result_IDA.T, label="sundials IDA (elapsed time *= 100)")
    elif name == "Robertson":
        result_IDA = np.loadtxt(_ida_csv("robertson", "robertson_errors_IDA.csv"), delimiter=',')
        result_IDA[:, 1] *= 100 # scale elapsed time by 100
        ax.plot(*result_IDA.T, label="sundials IDA (elapsed time *= 100)")
    elif name == "Knife edge":
        result_IDA = np.loadtxt(_ida_csv("knife_edge", "knife_edge_errors_IDA.csv"), delimiter=',')
        result_IDA[:, 1] *= 100 # scale elapsed time by 100
        ax.plot(*result_IDA.T, label="sundials IDA (elapsed time *= 100)")
    elif name == "Arevalo":
        result_IDA = np.loadtxt(_ida_csv("arevalo", "arevalo_errors_IDA.csv"), delimiter=',')
        result_IDA[:, 1] *= 100 # scale elapsed time by 100
        ax.plot(*result_IDA.T, label="sundials IDA (elapsed time *= 100)")
    elif name == "Particle":
        result_IDA = np.loadtxt(_ida_csv("particle", "particle_errors_IDA.csv"), delimiter=',')
        result_IDA[:, 1] *= 100 # scale elapsed time by 100
        ax.plot(*result_IDA[: :2].T, label="sundials IDA (elapsed time *= 100)")
    elif name == "Weissinger":
        result_IDA = np.loadtxt(_ida_csv("weissinger", "weissinger_errors_IDA.csv"), delimiter=',')
        result_IDA[:, 1] *= 500 # scale elapsed time by 500
        ax.plot(*result_IDA.T, label="sundials IDA (elapsed time *= 500)")

    # export errors and elapsed time
    for i, ri in enumerate(results):
        np.savetxt(
            f"{name}_{solvers[i]}.txt",
            ri,
            delimiter=", ",
            header="elapsed_time, error_y, error_yp",
            comments="",
        )

        # ax.plot(ri[:, 0], ri[:, 1], label=solvers[i])

    ax.set_title(f"work-precision: {name}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()
    ax.set_xlabel("||y_ref(t1) - y(t1)||")
    ax.set_ylabel("elapsed time [s]")

    img_dir = os.path.join(_REPO_ROOT, "data", "img")
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, f"{name}_work_precision.png"), dpi=300)

    plt.show()
