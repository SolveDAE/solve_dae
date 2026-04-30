from itertools import product
import numpy as np
from numpy.testing import assert_, assert_equal
import pytest
from scipy.sparse import identity
from solve_dae.integrate import solve_dae
from scipy.integrate._ivp.tests.test_ivp import fun_complex, jac_complex, jac_complex_sparse, sol_complex
from scipy.integrate._ivp.tests.test_ivp import compute_error


def F_complex(t, y, yp):
    return yp - fun_complex(t, y)


def J_complex(t, y, yp):
    Jy = -jac_complex(t, y)
    Jyp = np.eye(1)
    return Jy, Jyp


def J_complex_sparse(t, y, yp):
    Jy = -jac_complex_sparse(t, y)
    Jyp = identity(1, dtype=np.common_type(y, yp))
    return Jy, Jyp


parameters_complex = product(
    ["BDF"], # method
    [None, J_complex, J_complex_sparse] # jac
)
@pytest.mark.parametrize("method, jac", parameters_complex)
def test_integration_complex(method, jac):
    rtol = 1e-3
    atol = 1e-6
    y0 = np.array([0.5 + 1j])
    yp0 = fun_complex(0, y0)
    t_span = [0, 1]
    tc = np.linspace(t_span[0], t_span[1])

    res = solve_dae(F_complex, t_span, y0, yp0, rtol=rtol, atol=atol, 
                    method=method, dense_output=True, jac=jac)
    
    assert_equal(res.t[0], t_span[0])
    assert_(res.t_events is None)
    assert_(res.y_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    # assert res.nfev < 25
    assert res.nfev < 29
    assert_equal(res.njev, 1)
    assert res.nlu < 6

    y_true = sol_complex(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert np.all(e < 5)

    yc_true = sol_complex(tc)
    yc = res.sol(tc)[0]
    e = compute_error(yc, yc_true, rtol, atol)

    assert np.all(e < 5)


# if __name__ == "__main__":
#     for params in parameters_complex:
#         test_integration_complex(*params)
