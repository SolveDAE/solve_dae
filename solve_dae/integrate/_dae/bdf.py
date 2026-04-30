import numpy as np
from math import factorial
from warnings import warn
from scipy.integrate._ivp.common import norm, EPS, warn_extraneous
from .base import DAEDenseOutput
from .dae import DaeSolver


NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10


def compute_R(order, factor):
    """Compute the matrix for changing the differences array."""
    I = np.arange(1, order + 1)[:, None]
    J = np.arange(1, order + 1)
    M = np.zeros((order + 1, order + 1))
    M[1:, 1:] = (I - 1 - factor * J) / I
    M[0] = 1
    return np.cumprod(M, axis=0)


def change_D(D, order, factor):
    """Change differences array in-place when step size is changed."""
    R = compute_R(order, factor)
    U = compute_R(order, 1)
    RU = R.dot(U)
    D[:order + 1] = np.dot(RU.T, D[:order + 1])


def solve_bdf_system(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol):
    """Solve the algebraic system resulting from BDF method."""
    d = np.zeros_like(y_predict)
    y = y_predict.copy()
    dy_norm_old = None
    converged = False
    for k in range(NEWTON_MAXITER):
        yp = c * d + psi
        f = fun(t_new, y, yp)
        if not np.all(np.isfinite(f)):
            break

        dy = solve_lu(LU, -f)
        dy_norm = norm(dy / scale)

        if dy_norm_old is None:
            rate = None
        else:
            rate = dy_norm / dy_norm_old

        if (rate is not None and (rate >= 1 or rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol)):
            break

        y += dy
        d += dy

        if (dy_norm == 0 or rate is not None and rate / (1 - rate) * dy_norm < tol):
            converged = True
            break

        dy_norm_old = dy_norm

    return converged, k + 1, y, yp, d


class BDFDAE(DaeSolver):
    """Implicit method based on backward-differentiation formulas.

    This is a variable order method with the order varying automatically from
    1 to 5. The general framework of the BDF algorithm is described in [1]_.
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
                 vectorized=False, first_step=None, max_order=5,
                 NDF_strategy="stability", **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, yp0, t_bound, rtol, atol, 
                         first_step=first_step, max_step=max_step, 
                         vectorized=vectorized, jac=jac, 
                         jac_sparsity=jac_sparsity, support_complex=True)
        
        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))

        assert 1 <= max_order <= 6, "Ensure that 1 <= max_order <= 6."
        if max_order == 6:
            warn("Choosing `max_order = 6` is not recomended due to its poor stability properties.",
                 stacklevel=3)
        self.max_order = max_order

        if NDF_strategy == "stability":
            kappa = np.array([0, -37 / 200, -1/9, 0.0834, 0.0665, 0.0551, 0.0464])
        elif NDF_strategy == "efficiency":
            kappa = np.array([0, -37 / 200, -1/9, -0.0823, -0.0415, 0, 0])
        else:
            kappa = np.array([0, -37 / 200, -1/9, 0, 0, 0, 0])

        kappa = kappa[:max_order + 1]
        self.gamma = np.hstack((0, np.cumsum(1 / np.arange(1, max_order + 1))))
        self.alpha = (1 - kappa) * self.gamma
        self.error_const = kappa * self.gamma + 1 / np.arange(1, max_order + 2)

        D = np.zeros((max_order + 3, self.n), dtype=self.y.dtype)
        D[0] = self.y
        D[1] = self.yp * self.h_abs * self.direction
        self.D = D

        self.order = 1
        self.n_equal_steps = 0
        self.LU = None

    def _step_impl(self):
        t = self.t
        D = self.D
        y = self.y
        yp = self.yp

        max_step = self.max_step
        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            change_D(D, self.order, max_step / self.h_abs)
            self.n_equal_steps = 0
        elif self.h_abs < min_step:
            h_abs = min_step
            change_D(D, self.order, min_step / self.h_abs)
            self.n_equal_steps = 0
        else:
            h_abs = self.h_abs

        atol = self.atol
        rtol = self.rtol
        order = self.order

        alpha = self.alpha
        gamma = self.gamma
        error_const = self.error_const

        Jy = self.Jy
        Jyp = self.Jyp
        LU = self.LU
        current_jac = self.jac is None

        step_accepted = False
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
                change_D(D, order, np.abs(t_new - t) / h_abs)
                self.n_equal_steps = 0
                LU = None

            h = t_new - t
            h_abs = np.abs(h)

            y_predict = np.sum(D[:order + 1], axis=0)

            scale = atol + rtol * np.abs(y_predict)
            psi = np.dot(D[1: order + 1].T, gamma[1: order + 1]) / h

            converged = False
            c = alpha[order] / h
            while not converged:
                if LU is None:
                    LU = self.lu(Jy + c * Jyp)

                converged, n_iter, y_new, yp_new, d = solve_bdf_system(
                    self.fun, t_new, y_predict, c, psi, 
                    LU, self.solve_lu, scale, self.newton_tol)

                if not converged:
                    if current_jac:
                        break
                    Jy, Jyp = self.jac(t, y, yp)
                    LU = None
                    current_jac = True

            if not converged:
                factor = 0.5
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                LU = None
                continue

            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

            error = error_const[order] * d
            scale = atol + rtol * np.abs(y_new)
            error_norm = norm(error / scale)

            if error_norm > 1:
                factor = max(MIN_FACTOR,
                             safety * error_norm ** (-1 / (order + 1)))
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                # As we didn't have problems with convergence, we don't
                # reset LU here.
            else:
                step_accepted = True

        self.n_equal_steps += 1

        self.t = t_new
        self.y = y_new
        self.yp = yp_new

        self.h_abs = h_abs
        self.Jy = Jy
        self.Jyp = Jyp
        self.LU = LU

        # Update differences. The principal relation here is
        # D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}. Keep in mind that D
        # contained difference for previous interpolating polynomial and
        # d = D^{k + 1} y_n. Thus this elegant code follows.
        D[order + 2] = d - D[order + 1]
        D[order + 1] = d
        for i in reversed(range(order + 1)):
            D[i] += D[i + 1]

        if self.n_equal_steps < order + 1:
            return True, None

        if order > 1:
            error_m = error_const[order - 1] * D[order]
            error_m_norm = norm(error_m / scale)
        else:
            error_m_norm = np.inf

        if order < self.max_order:
            error_p = error_const[order + 1] * D[order + 2]
            error_p_norm = norm(error_p / scale)
        else:
            error_p_norm = np.inf

        error_norms = np.array([error_m_norm, error_norm, error_p_norm])
        with np.errstate(divide='ignore'):
            factors = error_norms ** (-1 / np.arange(order, order + 3))

        # choose order with largest factor
        delta_order = np.argmax(factors) - 1

        # choose order with smallest error
        # TODO: This choice is advertised in Shampine2002 but experiments 
        # indicate it is not worth it
        # delta_order = np.argmin(error_norms) - 1

        order += delta_order
        self.order = order

        factor = min(MAX_FACTOR, safety * np.max(factors))
        self.h_abs *= factor
        change_D(D, order, factor)
        self.n_equal_steps = 0
        self.LU = None

        return True, None

    def _dense_output_impl(self):
        return BdfDenseOutput(self.t_old, self.t, self.h_abs * self.direction,
                              self.order, self.D[:self.order + 1].copy())


class BdfDenseOutput(DAEDenseOutput):
    def __init__(self, t_old, t, h, order, D):
        super().__init__(t_old, t)
        self.order = order
        self.t_shift = self.t - h * np.arange(self.order)
        self.denom = h * (1 + np.arange(self.order))
        self.D = D
        self.h = h

    def _call_impl(self, t):
        # if t.ndim == 0:
        #     dt = t - self.t_shift
        #     x = (t - self.t_shift) / self.denom
        #     p = np.cumprod(x)
        # else:
        #     dt = t - self.t_shift[:, None]
        #     x = (t - self.t_shift[:, None]) / self.denom[:, None]
        #     p = np.cumprod(x, axis=0)

        # with np.errstate(divide="ignore", invalid="ignore"):
        #     dp = p * np.cumsum(1 / dt, axis=0)
        #     mask = np.abs(dt) == 0
        #     mask.nonzero()
        #     # idx = np.where(mask, axis=0)
        #     # dp[np.abs(dt) == 0] = 0

        # y = np.dot(self.D[1:].T, p)
        # yp = np.dot(self.D[1:].T, dp)
        # if y.ndim == 1:
        #     y += self.D[0]
        # else:
        #     y += self.D[0, :, None]

        # # if np.any(mask):
        # #     print(f"hit grid exactly")
        # #     # yp[:, mask[0, :]] = np.dot(self.D[1:].T, mask)
        # #     for i in range(len(t)):
        # #         idx = mask[i]
        # #         yp[:, i] = self.D[0, :, None]
        # #         print(f"")

        # return y, yp

        vector_valued = t.ndim > 0
        t = np.atleast_1d(t)

        # ######################################################
        # # 0. default interpolation for y and yp is set to zero
        # ######################################################
        # x = (t - self.t_shift[:, None]) / self.denom[:, None]
        # p = np.cumprod(x, axis=0)
        # y = self.D[0, :, None] + np.dot(self.D[1:].T, p)
        # return y, 0 * y


        # #######################################################
        # # 1. Vectorized implementation of P(t) as given in 
        # #    reference [2]_. Logarithmic differentiation gives 
        # #    the derivative P'(t).
        # # TODO: Vectorized implementation of yp is not valid 
        # #       when t - t_shift = 0.
        # #######################################################
        # dt = t - self.t_shift[:, None]
        # if not np.all(np.abs(dt) > 0):
        #     print("logarithmix differentiation is not valid")
        # x = dt / self.denom[:, None]
        # p = np.cumprod(x, axis=0)

        # # # dp = p * np.cumsum(1 / dt, axis=0)
        # # with np.errstate(divide="ignore", invalid="ignore"):
        # #     dp = p * np.cumsum(1 / dt, axis=0)
        # #     # dp[np.abs(dt) == 0] = 0

        # # try:
        # #     dp = p * np.cumsum(1 / dt, axis=0)
        # # except:
        # #     print(f"warning")

        # y = self.D[0, :, None] + np.dot(self.D[1:].T, p)
        # # yp = np.dot(self.D[1:].T, dp)

        # if vector_valued:
        #     y = np.squeeze(y)
        #     # yp = np.squeeze(yp)

        # # return y, yp
        # return y, np.zeros_like(y)

        # # # TODO: Compute this derivative efficiently, 
        # # # see https://stackoverflow.com/questions/40916955/how-to-compute-gradient-of-cumprod-safely
        # # dp = np.cumsum((p)[::-1], axis=0)[::-1] / x
        # # yp = np.dot(self.D[1:].T, dp)
        # yp = np.zeros_like(y)

        # if vector_valued:
        #     y = np.squeeze(y)
        #     yp = np.squeeze(yp)

        # return y, yp

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
    
        #########################################################################
        # 3. compute both values by Horner's rule,
        # see see https://orionquest.github.io/Numacom/lectures/interpolation.pdf
        #########################################################################
        # y3 = np.zeros((self.D.shape[1], len(t)), dtype=self.D.dtype)
        # y3 += self.D[n, :, None]
        # yp3 = np.zeros_like(y3)
        # for j in range(n, 0, -1):
        #     x = (t - (self.t - (j - 1) * self.h)) / (j * self.h)
        #     yp3 = y3 + x * yp3 * (j * self.h)
        #     yp3 /= (j * self.h)
        #     y3 = self.D[j - 1, :, None] + x * y3

        # if vector_valued == 0:
        #     y3 = np.squeeze(y3)
        #     yp3 = np.squeeze(yp3)

        # return y3, yp3
