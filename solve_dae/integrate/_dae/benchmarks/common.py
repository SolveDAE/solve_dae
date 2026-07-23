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
    ("Radau", {"stages": 5}),
    # ("Radau", {"stages": 7}),
    # ("BDF", {"NDF_strategy": "stability"}),
    ("BDF", {"NDF_strategy": "accuracy"}),
    # ("BDF", {"NDF_strategy": None}),
]


def benchmark(t0, t1, y0, yp0, F, rtols, atols, h0s, name, y_ref=None, yp_ref=None,
              y_idx=None, mult_idx=None, mult_names=None):
    """Run a work-precision benchmark.

    By default this reports the classical (error_y, error_yp) pair, each an
    L2 norm over all components of y / y' (or over `y_idx` if given).

    If a DAE carries Lagrange multipliers that live in y' (as in the
    stabilized-index-1/Hiller formulation used by particle.py), pass
    `mult_idx` (indices into y' picking out the multiplier components,
    e.g. [4, 5] for lambda, mu) together with `y_idx` (indices into y
    picking out the *physical state*, e.g. [0, 1, 2, 3] for x, y, u, v).
    In that case the reported metrics become error_state (over y_idx) and
    one error per entry of `mult_idx` (optionally labelled via
    `mult_names`), *in addition to* the naive combined error_y/error_yp
    (L2 norm over the full y / y' vectors, exactly as computed when
    mult_idx is None). The naive numbers are still reported because they
    remain useful for some purposes (e.g. comparing directly against other
    IDA/DASSL-style benchmarks that report exactly this metric), but they
    should not be used to compare against a solver in which the
    multipliers are structurally different (e.g. algebraic components of
    y rather than of y', as in a GGL/RADAU5 formulation) since the
    multipliers generally converge at a different order than the physical
    state and lumping them together makes such comparisons misleading.
    """
    # time span
    t_span = (t0, t1)

    n_mult = 0 if mult_idx is None else len(mult_idx)

    # benchmark results
    results = np.zeros((len(solvers), len(rtols), 4 + n_mult if mult_idx is not None else 3))

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
            else:
                diff_y = y_ref - sol.y[:, -1]

            # naive combined error: L2 norm over the *full* y / y' vectors,
            # regardless of y_idx/mult_idx (kept for backward compatibility
            # and for comparisons against other naive-error benchmarks)
            error_y = np.linalg.norm(y_ref - sol.y[:, -1])
            error_yp = np.linalg.norm(yp_ref - sol.yp[:, -1])
            print(f"     => error_y: {error_y}")
            print(f"     => error_yp: {error_yp}")

            if mult_idx is not None:
                # state and multiplier errors reported separately, see
                # docstring above for why they must not be lumped together
                error_state = np.linalg.norm(diff_y)
                print(f"     => error_state: {error_state}")

                row = [elapsed_time, error_y, error_yp, error_state]
                for k, idx in enumerate(mult_idx):
                    err_k = abs(yp_ref[idx] - sol.yp[idx, -1])
                    name_k = mult_names[k] if mult_names is not None else f"mult{k}"
                    print(f"     => error_{name_k}: {err_k}")
                    row.append(err_k)
                results[i, j] = row
            else:
                results[i, j] = (elapsed_time, error_y, error_yp)

    mult_labels = None
    if mult_idx is not None:
        mult_labels = mult_names if mult_names is not None else [f"mult{k}" for k in range(n_mult)]

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, ri in enumerate(results):
        if mult_idx is not None:
            # columns: elapsed_time, error_y, error_yp, error_state, *mult_errors
            ax.plot(ri[:, 0], ri[:, 3], label=f"{solvers[i]} (state)")
            for k, mult_label in enumerate(mult_labels):
                ax.plot(ri[:, 0], ri[:, 4 + k], linestyle="--",
                         label=f"{solvers[i]} ({mult_label})")
            ax.plot(ri[:, 0], ri[:, 1], linestyle=":", label=f"{solvers[i]} (naive y)")
            ax.plot(ri[:, 0], ri[:, 2], linestyle=":", label=f"{solvers[i]} (naive yp)")
        else:
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
        # columns: rtol, elapsed_time, error_y, error_yp, error_state, error_la, error_mu
        result_IDA = np.loadtxt(_ida_csv("particle", "particle_errors_IDA.csv"), delimiter=',', skiprows=1)
        elapsed_IDA = result_IDA[:, 1]# * 100 # scale elapsed time by 100
        error_state_IDA = result_IDA[:, 4]
        ax.plot(elapsed_IDA, error_state_IDA, label="sundials IDA (state x, y, u, v)")

        result_RADAU5 = np.loadtxt(_ida_csv("particle", "particle_errors_RADAU5.csv"), delimiter=',', skiprows=1)
        elapsed_RADAU5 = result_RADAU5[:, 1]# * 100 # scale elapsed time by 100
        error_state_RADAU5 = result_RADAU5[:, 2]
        ax.plot(elapsed_RADAU5, error_state_RADAU5, label="RADAU5 (state x, y, u, v)")
    elif name == "Weissinger":
        result_IDA = np.loadtxt(_ida_csv("weissinger", "weissinger_errors_IDA.csv"), delimiter=',')
        result_IDA[:, 1] *= 500 # scale elapsed time by 500
        ax.plot(*result_IDA.T, label="sundials IDA (elapsed time *= 500)")

    # export errors and elapsed time
    if mult_idx is not None:
        header = "elapsed_time, error_y, error_yp, error_state, " + \
            ", ".join(f"error_{nm}" for nm in mult_labels)
    else:
        header = "elapsed_time, error_y, error_yp"

    for i, ri in enumerate(results):
        np.savetxt(
            f"{name}_{solvers[i]}.txt",
            ri,
            delimiter=", ",
            header=header,
            comments="",
        )

    ax.set_title(f"work-precision: {name}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()
    # every ax.plot(ri[:, 0], ri[:, k]) call above puts elapsed_time on the
    # x-axis and an error metric on the y-axis -- labels must match that
    ax.set_xlabel("elapsed time [s]")
    ax.set_ylabel("error (see legend)")

    img_dir = os.path.join(_REPO_ROOT, "data", "img")
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, f"{name}_work_precision.png"), dpi=300)

    plt.show()
