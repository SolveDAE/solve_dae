import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.linalg import eig, cdf2rdf
from scipy.integrate._ivp.common import norm, EPS, warn_extraneous
from .base import DAEDenseOutput
from .dae import DaeSolver


DAMPING_RATIO_ERROR_ESTIMATE = 0.05 # Hairer (8.19) is obtained by the choice 1.0. 
                                    # de Swart proposes 0.067 for s=3.
KAPPA = 1 # factor of the smooth limiter


def butcher_tableau(s):
    """
    Compute the Butcher tableau of an s-stage Radau IIA method as outlined in [1]_ and [2]_.

    Parameters
    ----------
    s : int
       Number of stages.

    References
    ----------
    .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.5.
    .. [2] W. Hoffmann, J. J. B. De Swart, "Approximating Runge-Kutta matrices by triangular matrices"
    """
    # quadrature nodes are given by the zeros of the right Radau polynomial,
    # see Hairer (5.2)
    poly = Poly([0, 1]) ** (s - 1) * Poly([-1, 1]) ** s
    poly_der = poly.deriv(s - 1)
    c = poly_der.roots()

    # the coefficent matrix is computed according to  Hoffmann Section 2 or 
    # Hairer Section IV.5 (5.6) (requires some work to see the equivalence)
    Vc = np.vander(c, increasing=True)
    R = np.diag(1 / np.arange(1, s + 1))
    A = np.diag(c) @ Vc @ R @ np.linalg.inv(Vc)

    # since the method is stiffly accurate we can simply extract the 
    # quadrature weights
    b = A[-1, :]

    # the quadrature order
    p = 2 * s - 1

    return A, b, c, p


def radau_constants(s):
    """Compute all constants required for the integration method, depending on the Radau IIA Butcher tableau.

    References
    ----------
    .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [2] J. de Swart, G. Söderlind, "On the construction of error estimators for 
           implicit Runge-Kutta methods", Journal of Computational and Applied 
           Mathematics, 86, pp. 347-358, 1997.
    """
    # Butcher tableau
    A, b, c, p = butcher_tableau(s)

    # inverse coefficient matrix
    A_inv = np.linalg.inv(A)

    # eigenvalues and corresponding eigenvectors of coefficient matrix
    lambdas, Q = eig(A)

    # sort eigenvalues and permut eigenvectors accordingly
    idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idx]
    Q = Q[:, idx]

    # scale eigenvectors to get a "nice" transformation matrix (used by original 
    # scipy code) and at the same time minimizes the condition number of Q
    for i in range(s):
        Q[:, i] /= Q[-1, i]

    # convert complex eigenvalues and eigenvectors to real eigenvalues 
    # in a block diagonal form and the associated real eigenvectors
    Gammas, T = cdf2rdf(lambdas, Q)
    TI = np.linalg.inv(T)

    # sanity checks
    assert np.allclose(Q @ np.diag(lambdas) @ np.linalg.inv(Q), A)
    assert np.allclose(np.linalg.inv(Q) @ A @ Q, np.diag(lambdas))
    assert np.allclose(T @ Gammas @ TI, A)
    assert np.allclose(TI @ A @ T, Gammas)

    # extract real and complex eigenvalues
    real_idx = lambdas.imag == 0
    complex_idx = ~real_idx
    gammas = lambdas[real_idx].real
    alphas_betas = lambdas[complex_idx]
    alphas = alphas_betas[::2].real
    betas = alphas_betas[::2].imag
    
    # compute embedded method for error estimate
    c_hat = np.array([0, *c])
    Vc_hat = np.vander(c_hat, increasing=True).T

    # explicit embedded method as proposed in Hairer (8.16)
    rhs_explicit = 1 / np.arange(1, s + 1)
    rhs_explicit[0] -= gammas[0]
    b_hat_explicit = np.linalg.solve(Vc_hat[:-1, 1:], rhs_explicit)
    v_explicit = b_hat_explicit - b

    # implicit embedded method as proposed in de Swart (12.1)
    rhs_implicit = 1 / np.arange(1, s + 1)
    b_hats_implicit = gammas[0] # real eigenvalue of A
    b_hat1_implicit = DAMPING_RATIO_ERROR_ESTIMATE * b_hats_implicit
    rhs_implicit[0] -= b_hat1_implicit
    rhs_implicit -= b_hats_implicit
    b_hat_implicit = np.linalg.solve(Vc_hat[:-1, 1:], rhs_implicit)
    v_implicit = b - b_hat_implicit

    # compute the inverse of the Vandermonde matrix to get the inverse 
    # interpolation matrix
    P = np.linalg.inv(Vc_hat[1:, 1:])

    # inverse Vandermonde matrix for collocation error
    Vc = np.vander(c, increasing=True)
    P2 = np.linalg.inv(Vc)

    # these linear combinations are used in the algorithm
    MU_REAL = gammas[0]
    MU_COMPLEX = alphas -1j * betas

    return (
        A, A_inv, c, T, TI, P, P2, b_hat1_implicit, v_implicit, v_explicit, 
        MU_REAL, MU_COMPLEX, b_hat_implicit, b_hat_explicit, p,
    )


def solve_collocation_system(fun, t, y, h, Yp0, scale, tol,
                             LU_real, LU_complex, solve_lu,
                             C, T, TI, A, newton_max_iter):
    s, n = Yp0.shape
    ncs = s // 2
    tau = t + h * C

    Yp = Yp0
    Y = y + h * A.dot(Yp)

    F = np.empty((s, n))

    dY_norm_old = None
    dW_dot = np.empty_like(F)
    converged = False
    rate = None
    for k in range(newton_max_iter):
        for i in range(s):
            F[i] = fun(tau[i], Y[i], Yp[i])

        if not np.all(np.isfinite(F)):
            break

        G = TI @ F
        G_real = -G[0]
        G_complex = np.empty((ncs, n), dtype=complex)
        for i in range(ncs):
            G_complex[i] = -(G[2 * i + 1] + 1j * G[2 * i + 2])

        dW_dot_real = solve_lu(LU_real, G_real)
        dW_dot_complex = np.empty_like(G_complex)
        for i in range(ncs):
            dW_dot_complex[i] = solve_lu(LU_complex[i], G_complex[i])

        dW_dot[0] = dW_dot_real
        for i in range(ncs):
            dW_dot[2 * i + 1] = dW_dot_complex[i].real
            dW_dot[2 * i + 2] = dW_dot_complex[i].imag

        dYp = T.dot(dW_dot)
        dY = h * A.dot(dYp)

        Yp += dYp
        Y += dY

        dY_norm = norm(dY / scale)
        if dY_norm_old is not None:
            rate = dY_norm / dY_norm_old

        if (rate is not None and (rate >= 1 or rate ** (newton_max_iter - k) / (1 - rate) * dY_norm > tol)):
            break

        if (dY_norm == 0 or rate is not None and rate / (1 - rate) * dY_norm < tol):
            converged = True
            break

        dY_norm_old = dY_norm

    return converged, k + 1, Y, Yp, Y - y, rate


def predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s):
    """Predict by which factor to increase/decrease the step size.

    The algorithm is described in [1]_.

    Parameters
    ----------
    h_abs, h_abs_old : float
        Current and previous values of the step size, `h_abs_old` can be None
        (see Notes).
    error_norm, error_norm_old : float
        Current and previous values of the error norm, `error_norm_old` can
        be None (see Notes).
    s : int
        Number of stages of the Radau IIA method.

    Returns
    -------
    factor : float
        Predicted factor.

    Notes
    -----
    If `h_abs_old` and `error_norm_old` are both not None then a two-step
    algorithm is used, otherwise a one-step algorithm is used.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems", Sec. IV.8.
    """
    if error_norm_old is None or h_abs_old is None or error_norm == 0:
        multiplier = 1
    else:
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** (1 / (s + 1))

    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** (-1 / (s + 1))

    # smooth limiter
    factor = 1 + KAPPA * np.arctan((factor - 1) / KAPPA)

    return factor


class RadauDAE(DaeSolver):
    """Implicit Runge-Kutta method of Radau IIA family of order 2s - 1.

    The implementation follows [4]_, where most of the ideas come from [2]_. 
    The embedded formula of [3]_ is applied to implicit differential equations.
    The error is controlled with a (s)th-order accurate embedded formula as 
    introduced in [2]_ and refined in [3]_. The procedure is slightly adapted 
    by [4]_ to cope with implicit differential equations. The embedded error 
    estimate can be mixed with the contunous error of the lower order 
    collocation polynomial as porposed in [5]_ and [6]_.
    
    A cubic polynomial 
    which satisfies the collocation conditions is used for the dense output of 
    both state and derivatives.

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
    stages : int, optional
        Number of used stages. Default is 3, which corresponds to the 
        ``solve_ivp`` method. Only odd number of stages are allowed.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    newton_max_iter : int or None, optional
        Number of allowed (simplified) Newton iterations. Default is ``None`` 
        which uses ``newton_max_iter = 7 + (stages - 3) * 2`` as done in
        Hairer's radaup.f code.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. HHere `rtol` controls a
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
    continuous_error_weight : float, optional
        Weighting of continuous error of the dense output as introduced in 
        [5]_ and [6]_. The embedded error is weighted by (1 - continuous_error_weight). 
        Has to satisfy 0 <= continuous_error_weight <= 1. Default is 0.0, i.e., only 
        the embedded error is considered.
    jac : {None, array_like, sparse_matrix, callable}, optional
        Jacobian matrix of the right-hand side of the system with respect to
        y, required by this method. The Jacobian matrix has shape (n, n) and
        its element (i, j) is equal to ``d f_i / d y_j``.
        There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)`` as necessary.
              For the 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
        elements in *each* row, providing the sparsity structure will greatly
        speed up the computations [2]_. A zero entry means that a corresponding
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
    .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13, pp. 117-120, 1974.
    .. [3] J. de Swart, G. Söderlind, "On the construction of error estimators for 
           implicit Runge-Kutta methods", Journal of Computational and Applied 
           Mathematics, 86, pp. 347-358, 1997.
    .. [4] B. Fabien, "Analytical System Dynamics: Modeling and Simulation", 
           Sec. 5.3.5.
    .. [5] N. Guglielmi, E. Hairer, "Implementing Radau IIA Methods for Stiff 
           Delay Differential Equations", Computing 67, 1-12, 2001.
    .. [6] N. Guglielmi, "Open issues in devising software for the numerical 
           solution of implicit delay differential equations", Journal of 
           Computational and Applied Mathematics 185, 261-277, 2006.
    """
    def __init__(self, fun, t0, y0, yp0, t_bound, stages=3,
                 max_step=np.inf, rtol=1e-3, atol=1e-6, 
                 continuous_error_weight=0.0, jac=None, 
                 jac_sparsity=None, vectorized=False, 
                 first_step=None, newton_max_iter=None,
                 jac_recompute_rate=1e-3, newton_iter_embedded=1,
                 controller_deadband=(1.0, 1.2),
                 **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, yp0, t_bound, rtol, atol, 
                         first_step=first_step, max_step=max_step, 
                         vectorized=vectorized, jac=jac, 
                         jac_sparsity=jac_sparsity)

        assert stages % 2 == 1
        self.stages = stages
        (
            self.A, self.A_inv, self.C, self.T, self.TI, self.P, self.P2, 
            self.b_hat1_implicit, self.v_implicit, self.v_explicit, self.MU_REAL, self.MU_COMPLEX, 
            self.b_hat_implicit, self.b_hat_explicit, self.order,
        ) = radau_constants(stages)

        self.h_abs_old = None
        self.h_abs_oldold = None
        self.error_norm_old = None
        self.error_norm_oldold = None

        # modify tolerances as in radau.f line 824ff and 920ff
        # TODO: Document this rescaling
        EXPMNS = (stages + 1) / (2 * stages)
        # print(f"atol: {atol}")
        # print(f"rtol: {rtol}")
        # rtol = 0.1 * rtol ** EXPMNS
        # quott = atol / rtol
        # atol = rtol * quott
        # print(f"atol: {atol}")
        # print(f"rtol: {rtol}")

        # newton tolerance as in radau.f line 1008ff
        EXPMI = 1 / EXPMNS
        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** (EXPMI - 1)))
        # print(f"newton_tol: {self.newton_tol}")
        # print(f"10 * EPS / rtol: {10 * EPS / rtol}")
        # print(f"0.03")
        # print(f"rtol ** (EXPMI - 1): {rtol ** (EXPMI - 1)}")

        # maximum number of newton terations as in radaup.f line 234
        if newton_max_iter is None:
            newton_max_iter = 7 + int((stages - 3) * 2.5)
        
        assert isinstance(newton_max_iter, int)
        assert newton_max_iter >= 1
        self.newton_max_iter = newton_max_iter

        assert 0 <= continuous_error_weight <= 1
        self.continuous_error_weight = continuous_error_weight
        self.sol = None

        assert 0 < jac_recompute_rate < 1
        self.jac_recompute_rate = jac_recompute_rate

        assert 0 < controller_deadband[0] <= controller_deadband[1]
        self.controller_deadband = controller_deadband

        assert 0 <= newton_iter_embedded
        self.newton_iter_embedded = newton_iter_embedded

        self.current_jac = True
        self.LU_real = None
        self.LU_complex = None
        self.Z = None

    def _step_impl(self):
        t = self.t
        y = self.y
        yp = self.yp

        s = self.stages
        MU_REAL = self.MU_REAL
        MU_COMPLEX = self.MU_COMPLEX
        C = self.C
        T = self.T
        TI = self.TI
        A = self.A
        A_inv = self.A_inv
        v_implicit = self.v_implicit
        v_explicit = self.v_explicit
        b_hat1_implicit = self.b_hat1_implicit
        P2 = self.P2

        max_step = self.max_step
        atol = self.atol
        rtol = self.rtol
        newton_tol = self.newton_tol
        newton_max_iter = self.newton_max_iter

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            h_abs_old = None
            error_norm_old = None
        elif self.h_abs < min_step:
            h_abs = min_step
            h_abs_old = None
            error_norm_old = None
        else:
            h_abs = self.h_abs
            h_abs_old = self.h_abs_old
            error_norm_old = self.error_norm_old

        Jy = self.Jy
        Jyp = self.Jyp
        LU_real = self.LU_real
        LU_complex = self.LU_complex

        current_jac = self.current_jac
        jac = self.jac

        factor = None
        step_accepted = False
        message = None
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            if self.sol is None:
                Yp0 = np.tile(yp, s).reshape(s, -1)
            else:
                # improved guess by extrapolating the collocation polynomial
                Yp0 = self.sol(t + h * C)[1].T
            scale = atol + np.abs(y) * rtol

            converged = False
            while not converged:
                if LU_real is None or LU_complex is None:
                    LU_real = self.lu(Jyp + h * MU_REAL * Jy)
                    LU_complex = [self.lu(Jyp + h * MU * Jy) for MU in MU_COMPLEX]

                converged, n_iter, Y, Yp, Z, rate = solve_collocation_system(
                    self.fun, t, y, h, Yp0, scale, newton_tol,
                    LU_real, LU_complex, self.solve_lu,
                    C, T, TI, A, newton_max_iter)

                if not converged:
                    if current_jac:
                        break

                    Jy, Jyp = self.jac(t, y, yp)
                    current_jac = True
                    LU_real = None
                    LU_complex = None

            if not converged:
                h_abs *= 0.5
                LU_real = None
                LU_complex = None
                continue

            y_new = Y[-1]
            yp_new = Yp[-1]

            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol

            # error of collocation polynomial of order s
            error_collocation = y - P2[0] @ Y

            # embedded error measures
            if self.newton_iter_embedded == 0:
                # explicit embedded method with R(z) = +oo for z -> oo
                error_embedded = h * (MU_REAL * yp + v_explicit @ Yp)
            elif self.newton_iter_embedded == 1:
                # compute implicit embedded method with a single Newton iteration;
                # R(z) = b_hat1 / b_hats2 = DAMPING_RATIO_ERROR_ESTIMATE for z -> oo
                yp_hat_new = (v_implicit @ Yp - b_hat1_implicit * yp) / MU_REAL
                F = self.fun(t_new, y_new, yp_hat_new)
                error_embedded = -h * MU_REAL * self.solve_lu(LU_real, F)
            else:
                # compute implicit embedded method with `newton_iter_embedded` iterations
                yp_hat_new0 = -(y / h + b_hat1_implicit * yp + self.b_hat_implicit @ Yp) / MU_REAL 
                y_hat_new = y_new.copy() # initial guess
                for _ in range(self.newton_iter_embedded):
                    yp_hat_new = yp_hat_new0 + y_hat_new / (h * MU_REAL)
                    F = self.fun(t_new, y_hat_new, yp_hat_new)
                    y_hat_new -= h * MU_REAL * self.solve_lu(LU_real, F)

                error_embedded = y_hat_new - y_new 

            # mix embedded error with collocation error as proposed in Guglielmi2001/ Guglielmi2003
            error = (
                self.continuous_error_weight * np.abs(error_collocation)**((s + 1) / s) 
                + (1 - self.continuous_error_weight) * np.abs(error_embedded)
            )
            error_norm = norm(error / scale)

            safety = 0.9 * (2 * newton_max_iter + 1) / (2 * newton_max_iter + n_iter)

            if error_norm > 1:
                factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s)
                h_abs *= safety * factor

                LU_real = None
                LU_complex = None
            else:
                step_accepted = True

        # Step is converged and accepted
        recompute_jac = (
            jac is not None 
            and n_iter > 2 
            and rate > self.jac_recompute_rate
        )

        if factor is None:
            factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s)
            factor = safety * factor

        # do not alter step-size in deadband
        if (
            not recompute_jac 
            and self.controller_deadband[0] <= factor <= self.controller_deadband[1]
        ):
            factor = 1
        else:
            LU_real = None
            LU_complex = None

        if recompute_jac:
            Jy, Jyp = self.jac(t_new, y_new, yp_new)
            current_jac = True
        elif jac is not None:
            current_jac = False

        self.h_abs_old = self.h_abs
        self.error_norm_old = error_norm

        self.h_abs = h_abs * factor

        self.y_old = y
        self.yp_old = yp

        self.t = t_new
        self.y = y_new
        self.yp = yp_new

        self.Z = Z
        self.Y = Y
        self.Yp = Yp

        self.LU_real = LU_real
        self.LU_complex = LU_complex
        self.current_jac = current_jac
        self.Jy = Jy
        self.Jyp = Jyp

        self.t_old = t
        self.sol = self._compute_dense_output()

        return step_accepted, message

    def _compute_dense_output(self):
        Z = self.Y - self.y_old
        ZP = np.dot(Z.T, self.P)
        Zp = self.Yp - self.yp_old
        ZpP = np.dot(Zp.T, self.P)
        return RadauDenseOutput(self.t_old, self.t, self.y_old, ZP, ZpP)

    def _dense_output_impl(self):
        return self.sol


class RadauDenseOutput(DAEDenseOutput):
    def __init__(self, t_old, t, y_old, ZP, ZpP):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.ZP = ZP
        self.ZpP = ZpP
        self.order = ZP.shape[1] - 1
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        x = np.atleast_1d(x)

        # factors for interpolation polynomial and its derivative
        c = np.arange(1, self.order + 2)[:, None]
        p = x**c
        dp = (c / self.h) * (x**(c - 1))

        # compute derivative of interpolation polynomial for y
        y = np.dot(self.ZP, p)
        y += self.y_old[:, None]
        yp = np.dot(self.ZP, dp)

        if t.ndim == 0:
            y = np.squeeze(y)
            yp = np.squeeze(yp)

        return y, yp
