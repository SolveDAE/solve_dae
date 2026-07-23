import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose
from solve_dae.integrate import solve_dae


parameters_method = ["BDF", "Radau"]


def F(t, y, yp, scale=1.0):
    return yp - scale * y


def y_true(t, y0, t0, scale=1.0):
    return y0 * np.exp(scale * (t - t0))


@pytest.mark.parametrize("method", parameters_method)
@pytest.mark.parametrize("dense_output", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_t_eval(method, dense_output, backward):
    y0 = np.array([1.0])
    yp0 = np.array([-0.5])
    t_span = (0, 5) if not backward else (5, 0)

    t_eval = np.linspace(*t_span, 11)

    res = solve_dae(F, t_span, y0, yp0, method=method, args=(-0.5,),
                    t_eval=t_eval, dense_output=dense_output,
                    rtol=1e-8, atol=1e-10)
    assert res.success
    assert_allclose(res.t, t_eval)
    assert_allclose(res.y[0], y_true(t_eval, y0[0], t_span[0], -0.5), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("method", parameters_method)
def test_t_eval_not_1d(method):
    y0 = np.array([1.0])
    yp0 = np.array([-0.5])
    with pytest.raises(ValueError, match="`t_eval` must be 1-dimensional"):
        solve_dae(F, (0, 5), y0, yp0, method=method, t_eval=np.zeros((2, 2)))


@pytest.mark.parametrize("method", parameters_method)
def test_t_eval_out_of_bounds(method):
    y0 = np.array([1.0])
    yp0 = np.array([-0.5])
    with pytest.raises(ValueError, match="not within `t_span`"):
        solve_dae(F, (0, 5), y0, yp0, method=method, t_eval=[0, 6])


@pytest.mark.parametrize("method", parameters_method)
def test_t_eval_not_sorted(method):
    y0 = np.array([1.0])
    yp0 = np.array([-0.5])
    with pytest.raises(ValueError, match="not properly sorted"):
        solve_dae(F, (0, 5), y0, yp0, method=method, t_eval=[0, 3, 2])


@pytest.mark.parametrize("method", parameters_method)
def test_args_passed_to_fun(method):
    y0 = np.array([1.0])
    yp0 = np.array([-0.5])
    t_span = (0, 2)

    res = solve_dae(F, t_span, y0, yp0, method=method, args=(-0.5,),
                    rtol=1e-8, atol=1e-10)
    assert res.success
    assert_allclose(res.y[:, -1], y_true(2, y0[0], t_span[0], -0.5), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("method", parameters_method)
def test_args_passed_to_jac_and_events(method):
    def jac(t, y, yp, scale=1.0):
        return -scale * np.eye(1), np.eye(1)

    def event(t, y, yp, scale=1.0):
        return y - 0.5
    event.terminal = True

    y0 = np.array([1.0])
    yp0 = np.array([-0.5])
    t_span = (0, 10)

    res = solve_dae(F, t_span, y0, yp0, method=method, args=(-0.5,),
                    jac=jac, events=[event], rtol=1e-8, atol=1e-10)
    assert res.success
    assert res.status == 1
    assert_allclose(res.y_events[0][0], 0.5, atol=1e-6)


@pytest.mark.parametrize("method", parameters_method)
def test_args_not_a_tuple_raises_helpful_error(method):
    y0 = np.array([1.0])
    yp0 = np.array([-0.5])
    with pytest.raises(TypeError, match="Supplied 'args' cannot be unpacked"):
        solve_dae(F, (0, 2), y0, yp0, method=method, args=-0.5)


# if __name__ == "__main__":
#     for method in parameters_method:
#         test_args_passed_to_jac_and_events(method)
