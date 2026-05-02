import numpy as np
import pytest
from itertools import product
from scipy.sparse import csc_matrix
from solve_dae.integrate import solve_dae


# initial conditions
y0 = [1]
yp0 = [0.5 * y0[0]]
t_span = (0, 1)


def F(t, y, yp):
    return yp - 0.5 * y

def jac_dense(t, y, yp):
    return np.eye(len(y)), -0.5 * np.eye(len(yp))

def jac_sparse(t, y, yp):
    return csc_matrix(np.eye(len(y))), csc_matrix(-0.5 * np.eye(len(yp)))

def jac_wrong_shape_dense(t, y, yp):
    return np.eye(len(y) + 1), -0.5 * np.eye(len(yp) + 1)

def jac_wrong_shape_sparse(t, y, yp):
    return csc_matrix(np.eye(len(y) + 1)), csc_matrix(-0.5 * np.eye(len(yp) + 1))


parameters_method = ["BDF", "Radau"]
parameters_jac = [jac_dense, jac_sparse, jac_wrong_shape_dense, jac_wrong_shape_sparse]


parameters = product(
    parameters_method,
    parameters_jac,
)


@pytest.mark.parametrize("method, jac", parameters)
def test_jacobian_shape(method, jac):
    if jac in [jac_wrong_shape_dense, jac_wrong_shape_sparse]:
        with pytest.raises(ValueError) as excinfo:
            solve_dae(F, t_span, y0, yp0, method=method, jac=jac)

            Jyp, Jy = jac(t_span[0], y0, yp0)
            message = ("`Jy` is expected to have shape {}, but " +
                    "actually has {}.").format((len(y0), len(y0)), Jy.shape)
            assert (
                message
                in str(excinfo.value)
            )

            message = ("`Jpy` is expected to have shape {}, but " +
                    "actually has {}.").format((len(y0), len(y0)), Jyp.shape)
            assert (
                message
                in str(excinfo.value)
            )
    else:
        solve_dae(F, t_span, y0, yp0, method=method, jac=jac)


# @pytest.mark.parametrize("method,", parameters_method)
# def test_step(method):
#     sol = solve_dae(F, t_span, y0, yp0, method=method)
#     pass


# if __name__ == "__main__":
#     for params in parameters:
#         test_jacobian_shape(*params)
