import numpy as np
import pytest
from itertools import product
from scipy.sparse import csc_matrix
from solve_dae.integrate import solve_dae


y0 = [1]
yp0 = [0.5 * y0[0]]
t_span = (0, 1.2)


def F(t, y, yp):
    return yp - 0.5 * y

def jac_dense(t, y, yp):
    return -0.5 * np.eye(len(y)), np.eye(len(yp))

def jac_sparse(t, y, yp):
    return csc_matrix(-0.5 * np.eye(len(y))), csc_matrix(np.eye(len(yp)))

def jac_wrong_shape_Jyp_dense(t, y, yp):
    return -0.5 * np.eye(len(y)), np.eye(len(yp) + 1), 

def jac_wrong_shape_Jy_dense(t, y, yp):
    return -0.5 * np.eye(len(y) + 1), np.eye(len(yp))

def jac_wrong_shape_Jyp_sparse(t, y, yp):
    return csc_matrix(-0.5 * np.eye(len(y))), csc_matrix(np.eye(len(yp) + 1))

def jac_wrong_shape_Jy_sparse(t, y, yp):
    return csc_matrix(-0.5 * np.eye(len(y) + 1)), csc_matrix(np.eye(len(yp)))

jac_wrong_shape_Jyp_dense_constant = (
    np.eye(2), 
    -0.5 * np.eye(1),
)

jac_wrong_shape_Jy_dense_constant = (
    np.eye(1), 
    -0.5 * np.eye(2),
)

jac_wrong_shape_Jyp_sparse_constant = (
    csc_matrix(np.eye(2)), 
    csc_matrix(-0.5 * np.eye(1)),
)

jac_wrong_shape_Jy_sparse_constant = (
    csc_matrix(np.eye(1)), 
    csc_matrix(-0.5 * np.eye(2)),
)


parameters_method = ["BDF", "Radau"]
parameters_jac_correct_shape = [jac_dense, jac_sparse]
parameters_jac_wrong_shape = [
    jac_wrong_shape_Jyp_dense, 
    jac_wrong_shape_Jy_dense, 
    jac_wrong_shape_Jyp_sparse, 
    jac_wrong_shape_Jy_sparse,
    jac_wrong_shape_Jyp_dense_constant,
    jac_wrong_shape_Jy_dense_constant,
    jac_wrong_shape_Jyp_sparse_constant,
    jac_wrong_shape_Jy_sparse_constant,
]
parameters_jac = parameters_jac_correct_shape + parameters_jac_wrong_shape


parameters = product(
    parameters_method,
    parameters_jac,
)


@pytest.mark.parametrize("method, jac", parameters)
def test_jacobian_shape(method, jac):
    if not callable(jac) or jac in parameters_jac_wrong_shape:
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


@pytest.mark.parametrize("method", parameters_method)
def test_small_max_step(method):
    solve_dae(F, t_span, y0, yp0, method=method, max_step=1e-2)


def F(t, y, yp):
    return yp - 1 / (1 - t)

y0 = [0]
yp0 = [1]
t_span = (0, 1.2)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in scalar divide")
@pytest.mark.parametrize("method", parameters_method)
def test_overflow(method):
    sol = solve_dae(F, t_span, y0, yp0, method=method)
    assert sol.status == -1
    assert sol.success == False
    assert sol.message == "Required step size is less than spacing between numbers."


# if __name__ == "__main__":
#     for params in parameters:
#         test_jacobian_shape(*params)

#     for params in parameters_method:
#         test_small_max_step(params)

#     for params in parameters_method:
#         test_overflow(params)
