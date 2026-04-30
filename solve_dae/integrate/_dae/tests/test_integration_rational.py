from itertools import product
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pytest
from scipy.sparse import identity, csc_matrix
from solve_dae.integrate import solve_dae

from scipy.integrate._ivp.tests.test_ivp import fun_rational, fun_rational_vectorized, jac_rational, jac_rational_sparse, sol_rational
from scipy.integrate._ivp.tests.test_ivp import compute_error


def F_rational(t, y, yp):
    return yp - fun_rational(t, y)


def F_rational_vectorized(t, y, yp):
    return np.vstack(yp) - fun_rational_vectorized(t, y)


def J_rational(t, y, yp):
    Jy = -jac_rational(t, y)
    Jyp = np.eye(2)
    return Jy, Jyp


def J_rational_sparse(t, y, yp):
    Jy = -jac_rational_sparse(t, y)
    Jyp = identity(2)
    return Jy, Jyp


sparsity = (
    csc_matrix(np.ones((2, 2))),
    identity(2),
)

parameters_rational = product(
    [None, 0.1], # first_step
    [False, True], # vectorized
    ["BDF", "Radau"], # method
    [[5, 9], [5, 1]], # t_span
    [None, J_rational, J_rational_sparse], # jac
    [None, sparsity], # jac_sparsity
)
@pytest.mark.parametrize("first_step, vectorized, method, t_span, jac, jac_sparsity", parameters_rational)
def test_integration_rational(first_step, vectorized, method, t_span, jac, jac_sparsity):
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    yp0 = fun_rational(5, y0)

    if vectorized:
        fun = F_rational_vectorized
    else:
        fun = F_rational

    res = solve_dae(fun, t_span, y0, yp0, rtol=rtol, atol=atol, 
                    method=method, dense_output=True, jac=jac, 
                    vectorized=vectorized, first_step=first_step,
                    jac_sparsity=jac_sparsity)
    
    assert_equal(res.t[0], t_span[0])
    assert_(res.t_events is None)
    assert_(res.y_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    if method == "BDF":
        assert_(0 < res.nfev < 31)
        assert_(0 < res.njev < 3)
        assert_(0 < res.nlu < 7)
    else: # Radau
        assert_(0 < res.nfev < 46)
        assert_(0 < res.njev < 3)
        assert_(0 < res.nlu < 13)

    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 6))

    tc = np.linspace(*t_span)
    yc_true = sol_rational(tc)
    yc = res.sol(tc)[0]

    e = compute_error(yc, yc_true, rtol, atol)
    assert_(np.all(e < 6))

    tc = (t_span[0] + t_span[-1]) / 2
    yc_true = sol_rational(tc)
    yc = res.sol(tc)[0]

    e = compute_error(yc, yc_true, rtol, atol)
    assert_(np.all(e < 5))

    assert_allclose(res.sol(res.t)[0], res.y, rtol=1e-15, atol=1e-15)

# if __name__ == "__main__":
#     for param in parameters_rational:
#         test_integration_rational(*param)
