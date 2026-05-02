import numpy as np
from math import factorial, tau
from warnings import warn
from scipy.integrate._ivp.common import norm, EPS, warn_extraneous
from .base import DAEDenseOutput
from .dae import DaeSolver


NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10


def solve_alpha_system(fun, t, h, yn, ypn, xn, ypn1_predict, 
                       alpha_f, alpha_m, gamma, LU, solve_lu, 
                       scale, tol, newton_max_iter):
    n = ypn.shape
    ypn1 = ypn1_predict.copy()

    # auxiliary derivative and state value
    xn1 = (alpha_f * ypn + (1 - alpha_f) * ypn1 - alpha_m * xn) / (
        1 - alpha_m
    )
    yn1 = yn + h * ((1 - gamma) * xn + gamma * xn1)

    dy_norm_old = None
    dyp = np.empty_like(ypn)
    dx = np.empty_like(xn1)
    dy = np.empty_like(yn1)
    converged = False
    rate = None
    for k in range(newton_max_iter):
        # evaluate DAE function
        F = fun(t, yn1, ypn1)
        if not np.all(np.isfinite(F)):
            break

        # solve linear system and do dependent updates
        dyp = -solve_lu(LU, F)
        dx = (1 - alpha_f) / (1 - alpha_m) * dyp
        dy = h * gamma * dx

        ypn1 += dyp
        xn1 += dx
        yn1 += dy

        dy_norm = norm(dy / scale)
        if dy_norm_old is not None:
            rate = dy_norm / dy_norm_old

        if (rate is not None and (rate >= 1 or rate ** (newton_max_iter - k) / (1 - rate) * dy_norm > tol)):
            break

        if (dy_norm == 0 or rate is not None and rate / (1 - rate) * dy_norm < tol):
            converged = True
            break

        dy_norm_old = dy_norm

    return converged, k + 1, yn1, ypn1, xn1, rate


class AlphaDAE(DaeSolver):
    # TODO: Adapt documentation
    """Implicit method based on generalized-alpha method.

    This is a variable order method with the order varying automatically from
    1 to 5. The general framework of the generalized-alpha algorithm is described 
    in [1]_.

    This class implements a quasi-constant step size as explained in [2]_.
    The error estimation strategy for the constant-step BDF is derived in [3]_.

    Different numerical differentiation formulas (NDF) are implemented. The 
    choice of [5]_ enhances the stability, while [2]_ improves the accuracy 
    of the method. Standard BDF methods are also implemented, although the 
    first and second order formula use the accuracy enhancement of [2]_ and 
    [5]_ since both methods are L-stable.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Function defining the DAE system: ``f(t, y, yp) = 0``. The calling 
        signature is ``fun(t, y, yp)``, where ``t`` is a scalar and 
        ``y, yp`` are ndarrays with 
        ``len(y) = len(yp) = len(y0) = len(yp0)``. ``fun`` must return 
        an array of the same shape as ``y, yp``. See `vectorized` for more
        information.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    yp0 : array_like, shape (n,)
        Initial derivative.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_order : int, optional
        Highest order of the method with 1 <= max_order <= 6, although 
        max_order = 6 should be used with care due to the limited stability 
        propoerties of the corresponding BDF method.
    NDF_strategy : string, optional
        The strategy that is applied for obtaining numerical differentiation 
        formulas (NDF):

            * 'stability' (default): Increase A(alpha) stability without decreasing 
              efficiency too much. This uses the coefficients of [5]_ but also 
              enhances the first order coefficient as proposed in [2]_.
            * 'efficiency': Increase efficiency without decreasing A(alpha) 
              stability too much, see [2]_.
            * otherwise: BDF case with improved efficiency for first and second 
              order method as proposed in [2]_ and [5]_.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : (array_like, array_like), (sparse_matrix, sparse_matrix), callable or None, optional
        Jacobian matrices of the right-hand side of the system with respect
        to y and y'. The Jacobian matrices have shape (n, n) and their 
        elements (i, j) are equal to ``d f_i / d y_j`` and 
        ``d f_i / d y_j'``, respectively.  There are three ways to define 
        the Jacobian:

            * If (array_like, array_like) or (sparse_matrix, sparse_matrix) 
              the Jacobian matrices are assumed to be constant.
            * If callable, the Jacobians are assumed to depend on t, y and y'; 
              it will be called as ``jac(t, y, y')``, as necessary. Additional 
              arguments have to be passed if ``args`` is used (see 
              documentation of ``args`` argument). The return values might be 
              a tuple of sparse matrices.
            * If None (default), the Jacobians will be approximated by finite 
              differences.

        It is generally recommended to provide the Jacobians rather than
        relying on a finite-difference approximation.
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
        elements in *each* row, providing the sparsity structure will greatly
        speed up the computations [4]_. A zero entry means that a corresponding
        element in the Jacobian is always zero. If None (default), the Jacobian
        is assumed to be dense.
    vectorized : bool, optional
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` 
        and ``yp`` of shape ``(n,)``, where ``n = len(y0) = len(yp0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` and ``yp`` of 
        shape ``(n, k)``, where ``k`` is an integer. In this case, `fun` must 
        behave such that ``fun(t, y, yp)[:, i] == fun(t, y[:, i], yp[:, i])``.

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by this method, but may result in slower
        execution overall in some circumstances (e.g. small ``len(y0)``).
        Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    yp : ndarray
        Current derivative.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.

    References
    ----------
    .. [1] G. D. Byrne, A. C. Hindmarsh, "A Polyalgorithm for the Numerical
           Solution of Ordinary Differential Equations", ACM Transactions on
           Mathematical Software, Vol. 1, No. 1, pp. 71-96, March 1975.
    .. [2] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
    .. [3] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations I:
           Nonstiff Problems", Sec. III.2.
    .. [4] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13, pp. 117-120, 1974.
    .. [5] R. W. Klopfenstein, "Numerical differentiation formulas for stiff 
           systems of ordinary differential equations", RCA Review, 32, 
           pp. 447-462, September 1971.
    .. [6] K. Radhakrishnan  and A C. Hindmarsh, "Description and Use of LSODE, 
           the Livermore Solver for Ordinary Differential Equations", NASA 
           Reference Publication, December, 1993.
    """
    def __init__(self, fun, t0, y0, yp0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, jac=None, jac_sparsity=None,
                 vectorized=False, first_step=None, rho_inf=0.5,
                 **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, yp0, t_bound, rtol, atol, 
                         first_step=first_step, max_step=max_step, 
                         vectorized=vectorized, jac=jac, 
                         jac_sparsity=jac_sparsity, support_complex=True)
        
        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))

        # compuate constants for generalized-alpha method
        assert 0 <= rho_inf <= 1, "Ensure that 0 <= rho_inf <= 1."
        self.rho_inf = rho_inf
        self.alpha_f = rho_inf / (rho_inf + 1)
        self.alpha_m = (3 * rho_inf - 1) / (2 * (rho_inf + 1))
        self.gamma = 0.5 + self.alpha_f - self.alpha_m
        self.mu = self.gamma * (1 - self.alpha_f) / (1 - self.alpha_m)

        self.LU = None
        self.x = self.yp.copy() # TODO: Add second-order guess here!

    def _step_impl(self):
        t = self.t
        y = self.y
        yp = self.yp
        x = self.x

        print(f"t: {t}")

        max_step = self.max_step
        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        atol = self.atol
        rtol = self.rtol
        alpha_f = self.alpha_f
        alpha_m = self.alpha_m
        gamma = self.gamma
        mu = self.mu

        Jy = self.Jy
        Jyp = self.Jyp
        LU = self.LU
        current_jac = self.jac is None

        step_accepted = False
        factor = 1.0
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
                LU = None

            h = t_new - t
            h_abs = np.abs(h)

            # TODO: Do prediction with extrapolation of dense output as in Radau IIA?
            ypn1_predict = yp

            scale = atol + rtol * np.abs(y)

            converged = False
            while not converged:
                if LU is None:
                    LU = self.lu(Jyp + h * mu * Jy)

                converged, n_iter, y_new, yp_new, x_new, rate = solve_alpha_system(
                    self.fun, t_new, h, y, yp, x, ypn1_predict, 
                    alpha_f, alpha_m, gamma, LU, self.solve_lu, 
                    scale, self.newton_tol, NEWTON_MAXITER)

                if not converged:
                    if current_jac:
                        break

                    Jy, Jyp = self.jac(t, y, yp)
                    current_jac = True
                    LU = None

            if not converged:
                h_abs *= 0.5
                LU = None
                continue

            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

            # error estimate (w.r.t implicit euler step)
            error = y_new - (y + h * yp_new)
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
            error_norm = norm(error / scale)

            if error_norm > 1:
                # p = 2, p_hat = 1 => p_hat + 1 = 2
                factor = max(MIN_FACTOR,
                             safety * error_norm ** (-1 / 2))
                h_abs *= factor
                # As we didn't have problems with convergence, we don't
                # reset LU here.
                # TODO: Radau resets LU decomposition here
                LU = None
            else:
                step_accepted = True

        # # Step is converged and accepted
        # recompute_jac = (
        #     jac is not None 
        #     and n_iter > 2 
        #     and rate > self.jac_recompute_rate
        # )

        self.t = t_new
        self.y = y_new
        self.yp = yp_new
        self.x = x_new

        self.h_abs = h_abs
        self.Jy = Jy
        self.Jyp = Jyp
        self.LU = LU

        factor = min(MAX_FACTOR, factor)
        self.h_abs *= factor
        self.LU = None

        return True, None

    def _dense_output_impl(self):
        return AlphaDenseOutput(self.t_old, self.t, self.h_abs * self.direction)


# TODO: Implement dense output using Hermite polynomials.
class AlphaDenseOutput(DAEDenseOutput):
    def __init__(self, t_old, t, h):
        super().__init__(t_old, t)
        # self.order = order
        # self.t_shift = self.t - h * np.arange(self.order)
        # self.denom = h * (1 + np.arange(self.order))
        # self.D = D
        self.h = h

    def _call_impl(self, t):
        # return y, yp
        return 0, 0
        raise NotImplementedError("Dense output is not implemented for generalized-alpha method yet.")

        ############################################################
        # 2. naive implementation of P(t) and P'(t) of p. 7 in [2]_.
        ############################################################
        y2 = np.zeros((self.D.shape[1], len(t)), dtype=self.D.dtype)
        y2 += self.D[0, :, None]
        yp2 = np.zeros_like(y2)
        for j in range(1, self.order + 1):
            fac2 = np.ones_like(t)
            dfac2 = np.zeros_like(t)
            for m in range(j):
                fac2 *= t - (self.t - m * self.h)

                dprod2 = np.ones_like(t)
                for i in range(j):
                    if i != m:
                        dprod2 *= t - (self.t - i * self.h)
                dfac2 += dprod2

            denom = factorial(j) * self.h**j
            y2 += self.D[j, :, None] * fac2 / denom
            yp2 += self.D[j, :, None] * dfac2 / denom

        if vector_valued == 0:
            y2 = np.squeeze(y2)
            yp2 = np.squeeze(yp2)

        return y2, yp2
