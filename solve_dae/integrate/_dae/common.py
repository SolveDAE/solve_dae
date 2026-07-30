from itertools import groupby
import numpy as np
# note: use sparse QR when available in scipy
from scipy.linalg import qr, solve_triangular
from scipy.sparse import issparse
from scipy.integrate._ivp.common import norm, EPS
from scipy.optimize._numdiff import approx_derivative


class DaeSolution:
    """Continuous DAE solution.
    It is organized as a collection of `DenseOutput` objects which represent
    local interpolants. It provides an algorithm to select a right interpolant
    for each given point.
    The interpolants cover the range between `t_min` and `t_max` (see
    Attributes below). Evaluation outside this interval is not forbidden, but
    the accuracy is not guaranteed.
    When evaluating at a breakpoint (one of the values in `ts`) a segment with
    the lower index is selected.
    Parameters
    ----------
    ts : array_like, shape (n_segments + 1,)
        Time instants between which local interpolants are defined. Must
        be strictly increasing or decreasing (zero segment with two points is
        also allowed).
    interpolants : list of DenseOutput with n_segments elements
        Local interpolants. An i-th interpolant is assumed to be defined
        between ``ts[i]`` and ``ts[i + 1]``.
    alt_segment : boolean
        Requests the alternative interpolant segment selection scheme. At each
        solver integration point, two interpolant segments are available. The
        default (False) and alternative (True) behaviours select the segment
        for which the requested time corresponded to ``t`` and ``t_old``,
        respectively. This functionality is only relevant for testing the
        interpolants' accuracy: different integrators use different
        construction strategies.
    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """
    def __init__(self, ts, interpolants, alt_segment=False):
        ts = np.asarray(ts)
        d = np.diff(ts)
        # The first case covers integration on zero segment.
        if not ((ts.size == 2 and ts[0] == ts[-1])
                or np.all(d > 0) or np.all(d < 0)):
            raise ValueError("`ts` must be strictly increasing or decreasing.")

        self.n_segments = len(interpolants)
        if ts.shape != (self.n_segments + 1,):
            raise ValueError("Numbers of time stamps and interpolants "
                             "don't match.")

        self.ts = ts
        self.interpolants = interpolants
        if ts[-1] >= ts[0]:
            self.t_min = ts[0]
            self.t_max = ts[-1]
            self.ascending = True
            self.side = "right" if alt_segment else "left"
            self.ts_sorted = ts
        else:
            self.t_min = ts[-1]
            self.t_max = ts[0]
            self.ascending = False
            self.side = "left" if alt_segment else "right"
            self.ts_sorted = ts[::-1]

    def _call_single(self, t):
        # Here we preserve a certain symmetry that when t is in self.ts,
        # if alt_segment=False, then we prioritize a segment with a lower
        # index.
        ind = np.searchsorted(self.ts_sorted, t, side=self.side)

        segment = min(max(ind - 1, 0), self.n_segments - 1)
        if not self.ascending:
            segment = self.n_segments - 1 - segment

        return self.interpolants[segment](t)

    def __call__(self, t):
        """Evaluate the solution.
        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points to evaluate at.
        Returns
        -------
        y : ndarray, shape (n_states,) or (n_states, n_points)
            Computed values. Shape depends on whether `t` is a scalar or a
            1-D array.
        """
        t = np.asarray(t)

        if t.ndim == 0:
            return self._call_single(t)

        order = np.argsort(t)
        reverse = np.empty_like(order)
        reverse[order] = np.arange(order.shape[0])
        t_sorted = t[order]

        # See comment in self._call_single.
        segments = np.searchsorted(self.ts_sorted, t_sorted, side=self.side)
        segments -= 1
        segments[segments < 0] = 0
        segments[segments > self.n_segments - 1] = self.n_segments - 1
        if not self.ascending:
            segments = self.n_segments - 1 - segments

        ys = []
        yps = []
        group_start = 0
        for segment, group in groupby(segments):
            group_end = group_start + len(list(group))
            y, yp = self.interpolants[segment](t_sorted[group_start:group_end])
            ys.append(y)
            yps.append(yp)
            group_start = group_end

        ys = np.hstack(ys)
        ys = ys[:, reverse]
        yps = np.hstack(yps)
        yps = yps[:, reverse]

        return ys, yps


def select_initial_step(t0, y0, yp0, t_bound, rtol, atol, max_step):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    yp0 : ndarray, shape (n,)
        Initial value of the dependent variable's derivative.
    t_bound : float
        Final value of the independent variable.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.
    atol : float
        Desired absolute tolerance.
    max_step : float
        Maximum allowed step size.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] L. F. Shampine, "Starting an ODE solver", November 1977.
    """
    safety = 0.8
    min_step = 16 * EPS * abs(t0)
    threshold = atol / rtol
    hspan = abs(t_bound - t0)

    # compute scaling
    wt = np.maximum(np.abs(y0), threshold)

    # error
    e = np.linalg.norm(yp0 / wt, np.inf)

    # reciprocal step size
    rh = e / np.sqrt(rtol) / safety

    # compute an initial step size
    h_abs = min(max_step, hspan)
    if h_abs * rh > 1:
        h_abs = 1 / rh

    # ensure h_abs >= min_step
    h_abs = max(h_abs, min_step)
    return h_abs


def consistent_initial_conditions(fun, t0, y0, yp0, jac=None, fixed_y0=None, 
                                  fixed_yp0=None, rtol=1e-8, atol=1e-8, 
                                  newton_maxiter=10, chord_iter=3,
                                  safety=0.5, *args):
    """Compute consistent initial conditions as discussed in [1]_.

    Only differentiation index <= 1 DAEs are supported: internally,
    `solve_underdetermined_system` raises ``ValueError("Index greater than
    one.")`` once the rank deficiency it detects exceeds the number of fixed
    unknowns, which is the signature of a higher-index problem. For such
    systems, consistent values have to be supplied by other means (see e.g.
    ``examples/daes/andrews.py`` for an index-2/3 example that hardcodes
    literature values instead of calling this function).

    `jac`, if given, may return `Jy` and/or `Jyp` as sparse matrices (e.g.
    the same callable already used for `solve_dae`/`BDF`/`Radau`, which are
    allowed to return sparse Jacobians). Since the underdetermined system is
    solved here with a dense, pivoted QR decomposition, any sparse output is
    converted to a dense array internally before use.

        References
    ----------
    .. [1] L. F. Shampine, "Solving 0 = F(t, y(t), y'(t)) in Matlab", Journal
           of Numerical Mathematics, vol. 10, no. 4, 2002, pp. 291-310.
    """
    n = len(y0)

    if jac is None:
        def jac(t, y, yp):
            n = len(y)
            z = np.concatenate((y, yp))

            def fun_composite(t, z):
                y, yp = z[:n], z[n:]
                return fun(t, y, yp, *args)

            J = approx_derivative(lambda z: fun_composite(t, z),
                                  z, method="2-point")
            J = J.reshape((n, 2 * n))
            Jy, Jyp = J[:, :n], J[:, n:]
            return Jy, Jyp

    user_jac = jac

    def jac(t, y, yp):
        # Accept a Jacobian callable that returns sparse Jy/Jyp (as used
        # elsewhere in solve_dae) and densify it here, since the QR-based
        # underdetermined solve below requires dense arrays.
        Jy, Jyp = user_jac(t, y, yp)
        if issparse(Jy):
            Jy = Jy.toarray()
        if issparse(Jyp):
            Jyp = Jyp.toarray()
        return np.asarray(Jy, dtype=float), np.asarray(Jyp, dtype=float)

    if fixed_y0 is None:
        free_y = np.arange(n)
    else:
        free_y = np.setdiff1d(np.arange(n), fixed_y0)
    
    if fixed_yp0 is None:
        free_yp = np.arange(n)
    else:
        free_yp = np.setdiff1d(np.arange(n), fixed_yp0)
    
    if len(free_y) + len(free_yp) < n:
        raise ValueError(f"Too many components fixed, cannot solve the problem.")

    if not (isinstance(rtol, float) and rtol > 0):
        raise ValueError("Relative tolerance must be a positive scalar.")
    
    if rtol < 100 * EPS:
        rtol = 100 * EPS
        print(f"Relative tolerance increased to {rtol}")
    
    if np.any(np.array(atol) <= 0):
        raise ValueError("Absolute tolerance must be positive.")
    
    assert 0 < safety <= 1, "safety factor has to be in (0, 1]"
    
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    yp0 = np.asarray(yp0, dtype=float).reshape(-1)
    f = fun(t0, y0, yp0, *args)
    Jy, Jyp = jac(t0, y0, yp0)

    scale_f = atol + np.abs(f) * rtol
    atol_arr = np.broadcast_to(np.asarray(atol, dtype=float), y0.shape)
    resnrm0 = norm(f)

    for _ in range(newton_maxiter):
        # Factorize the Jacobian pencil once per Newton (outer) iteration and
        # reuse it for all chord iterations below.
        factors = factorize_underdetermined_system(Jy, Jyp, free_y, free_yp)

        for _ in range(chord_iter):
            dy, dyp = solve_underdetermined_system(f, factors)

            # Trust region: cap the correction to at most a factor of 2
            # relative to the current iterate (with an atol-based floor for
            # iterates close to zero). Without this, a poor initial guess
            # combined with a strongly nonlinear F can make the raw
            # chord/Newton step overshoot and diverge instead of damping
            # towards the solution.
            nrm_state = max(norm(np.concatenate((y0, yp0))), norm(atol_arr))
            nrm_step = norm(np.concatenate((dy, dyp)))
            if nrm_step > 2 * nrm_state:
                step_scale = 2 * nrm_state / nrm_step
                dy = dy * step_scale
                dyp = dyp * step_scale
                nrm_step = 2 * nrm_state

            y0 = y0 + dy
            yp0 = yp0 + dyp

            f = fun(t0, y0, yp0, *args)
            resnrm = norm(f)
            error = norm(f / scale_f)

            # Convergence requires both: the residual must be small in an
            # absolute sense (relative to the tolerances) *and* not worse
            # than where we started, and the correction itself must have
            # become small relative to the current iterate. Checking the
            # residual alone (as before) can declare "convergence" while the
            # Newton/chord step is still moving substantially.
            if (error < safety and resnrm <= resnrm0
                    and nrm_step <= 1e-3 * rtol * nrm_state):
                return y0, yp0, f

        Jy, Jyp = jac(t0, y0, yp0)

    raise RuntimeError("Convergence failed.")


def qrank(A):
    """Compute QR-decomposition with column pivoting of A and estimate the rank."""
    Q, R, p = qr(A, pivoting=True)
    tol = max(A.shape) * EPS * abs(R[0, 0])
    rank = np.sum(abs(np.diag(R)) > tol)
    return rank, Q, R, p


def factorize_underdetermined_system(Jy, Jyp, free_y, free_yp):
    """Factorize the Jacobian pencil ``Jy, Jyp`` of the underdetermined system

        0 = f + Jy @ Delta_y + Jyp @ Delta_yp

    Only the QR-decompositions and rank information are computed here; the
    factors are independent of ``f`` and can be reused for repeated chord
    iterations (see `solve_underdetermined_system`).
    """
    n = Jy.shape[0]
    fixed = (n - len(free_y)) + (n - len(free_yp))

    def check_rankdef(rankdef):
        if rankdef > 0:
            if rankdef <= fixed:
                raise ValueError(f"Too many fixed components, rank deficiency is {rankdef}.")
            else:
                raise ValueError("Index greater than one.")

    if len(free_y) == 0:
        # solve 0 = f + Jyp @ Delta_yp (ODE case)
        rank, Q, R, p = qrank(Jyp)
        check_rankdef(n - rank)
        return ("yp_only", n, free_y, free_yp, rank, Q, R, p)

    if len(free_yp) == 0:
        # solve 0 = f + Jy @ Delta_y (pure algebraic case)
        rank, Q, R, p = qrank(Jy)
        check_rankdef(n - rank)
        return ("y_only", n, free_y, free_yp, rank, Q, R, p)

    # eliminate variables that are not free
    Jy = Jy[:, free_y]
    Jyp = Jyp[:, free_yp]

    # QR-decomposition of Fyp leads to the system
    # [R11, R12] [w1'] + [S11, S12] [w1] = [d1]
    # [  0,   0] [w2'] + [S21, S22] [w2] = [d2]
    # with S = Q.T @ Fy
    rank, Q, R, p = qrank(Jyp)
    if rank == n:
        # Full rank (ODE) case
        return ("full_rank", n, free_y, free_yp, rank, Q, R, p)

    # Rank deficient (DAE) case:
    S = Q.T @ Jy
    rankS, QS, RS, pS = qrank(S[rank:])
    check_rankdef(n - (rank + rankS))

    return ("rank_deficient", n, free_y, free_yp, rank, Q, R, p, S, rankS, QS, RS, pS)


def solve_underdetermined_system(f, factors):
    """Solve the underdetermined system
        0 = f + Jy @ Delta_y + Jyp @ Delta_yp
    for the given right-hand side `f`, using a Jacobian pencil already
    factorized by `factorize_underdetermined_system`.
    A solution is obtained with as many components as possible of
    (transformed) Delta_y and Delta_yp set to zero.
    """
    case = factors[0]
    n, free_y, free_yp = factors[1], factors[2], factors[3]
    Delta_y = np.zeros(n)
    Delta_yp = np.zeros(n)

    match case:
        # if case == "yp_only":
        case "yp_only":
            _, _, _, _, rank, Q, R, p = factors
            d = -Q.T @ f
            Delta_yp_ = np.zeros_like(Delta_yp)
            Delta_yp_[p] = solve_triangular(R, d)
            Delta_yp[free_yp] = Delta_yp_
            return Delta_y, Delta_yp

        # if case == "y_only":
        case "y_only":
            _, _, _, _, rank, Q, R, p = factors
            d = -Q.T @ f
            Delta_y_ = np.zeros_like(Delta_y)
            Delta_y_[p] = solve_triangular(R, d)
            Delta_y[free_y] = Delta_y_
            return Delta_y, Delta_yp

        # if case == "full_rank":
        case "full_rank":
            _, _, _, _, rank, Q, R, p = factors
            d = -Q.T @ f
            # Set all free dy to zero and solve triangular system below
            Delta_y[free_y] = 0
            Delta_yp_ = np.zeros_like(Delta_yp)
            Delta_yp_[p] = solve_triangular(R, d)
            Delta_yp[free_yp] = Delta_yp_
            return Delta_y, Delta_yp

        case _:
            # case == "rank_deficient"
            _, _, _, _, rank, Q, R, p, S, rankS, QS, RS, pS = factors
            d = -Q.T @ f

    # decompose d
    d1 = d[:rank]
    d2 = d[rank:]

    # compute basic solution of underdetermined system
    # [S21, S22] [w1] = d2
    #            [w2]
    # using column pivoting QR-decomposition
    # [RS11, RS12] [v1] = [c1]
    # [   0,    0] [v2]   [c2]
    # with v2 = 0 this gives
    # RS11 @ v1 = c1
    c = QS.T @ d2
    v = np.zeros(RS.shape[1])
    v[:rankS] = solve_triangular(RS[:rankS, :rankS], c[:rankS])

    # apply permutation
    w = np.zeros_like(v)
    w[pS] = v

    # set w2' = 0 and solve the remaining system
    # [R11] w1' = d1 - [S11, S12] [w1]
    #                             [w2]
    wp = np.zeros(R.shape[1])
    if rank > 0:
        wp_ = np.zeros(R.shape[1])
        wp_[:rank] = solve_triangular(R[:rank, :rank], d1 - S[:rank] @ w)
        wp[p] = wp_

    # store w and wp
    Delta_y[free_y] = w
    Delta_yp[free_yp] = wp

    return Delta_y, Delta_yp
