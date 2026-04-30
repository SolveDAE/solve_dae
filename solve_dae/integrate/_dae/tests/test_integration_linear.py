from itertools import product
import numpy as np
from numpy.testing import assert_, assert_equal
import pytest
from scipy.sparse import csc_matrix, identity
from solve_dae.integrate import solve_dae
from scipy.integrate._ivp.tests.test_ivp import fun_linear, jac_linear, sol_linear, compute_error


def F_linear(t, y, yp):
    return yp - fun_linear(t, y)


J_linear = (
    -jac_linear(),
    np.eye(2),
)

J_linear_sparse = (
    -csc_matrix(jac_linear()),
    identity(2, format="csc"),
)


parameters_linear = product(
    ["BDF", "Radau"], # method
    [None, J_linear, J_linear_sparse], # jac
)
@pytest.mark.parametrize("method, jac", parameters_linear)
def test_integration_linear(method, jac):
    rtol = 1e-3
    atol = 1e-6
    y0 = [0, 2]
    yp0 = fun_linear(0, y0)
    t_span = [0, 2]

    res = solve_dae(F_linear, t_span, y0, yp0, rtol=rtol, atol=atol, 
                    method=method, dense_output=True, jac=jac)
    
    assert_equal(res.t[0], t_span[0])
    assert_(res.t_events is None)
    assert_(res.y_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    y_true = sol_linear(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))

# if __name__ == "__main__":
#     for param in parameters_linear:
#         test_integration_linear(*param)
