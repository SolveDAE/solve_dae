import numpy as np
import pytest
from solve_dae.integrate import solve_dae


parameters_check_arguments = ["BDF", "Radau"] # method
@pytest.mark.parametrize("method,", parameters_check_arguments)
def test_check_arguments(method):
    F = lambda t, y, yp: yp + 0.5 * y

    # support_complex
    y0 = np.arange(3) * 1j
    yp0 = np.arange(-1, 2) * 1j
    t_span = (0, 1)
    if method == "Radau":
        with pytest.raises(ValueError) as excinfo:
            solve_dae(F, t_span, y0, yp0, method=method)
        assert (
            "`y0` or `yp0` is complex, but the chosen "
            "solver does not support integration in a "
            "complex domain."
            in str(excinfo.value)
        )
    else:
        solve_dae(F, t_span, y0, yp0, method=method)

    # `y0` must be 1-dimensional.
    y0 = 1
    yp0 = np.arange(2)
    t_span = (0, 1)
    with pytest.raises(ValueError) as excinfo:
        solve_dae(F, t_span, y0, yp0, method=method)
    assert (
        "`y0` must be 1-dimensional."
        in str(excinfo.value)
    )

    # `yp0` must be 1-dimensional.
    y0 = np.arange(2)
    yp0 = 1
    t_span = (0, 1)
    with pytest.raises(ValueError) as excinfo:
        solve_dae(F, t_span, y0, yp0, method=method)
    assert (
        "`yp0` must be 1-dimensional."
        in str(excinfo.value)
    )

    # y0 and yp0 must be of same shape.
    y0 = np.arange(2)
    yp0 = np.arange(3)
    t_span = (0, 1)
    with pytest.raises(ValueError) as excinfo:
        solve_dae(F, t_span, y0, yp0, method=method)
    assert (
        "`y0` and `yp0` must be of same shape."
        in str(excinfo.value)
    )

    # All components of the initial state `y0` must be finite.
    y0 = np.arange(1, 3) * np.inf
    yp0 = np.arange(2)
    t_span = (0, 1)
    with pytest.raises(ValueError) as excinfo:
        solve_dae(F, t_span, y0, yp0, method=method)
    assert (
        "All components of the initial state `y0` must be finite."
        in str(excinfo.value)
    )
    
    # All components of the initial state `yp0` must be finite.
    y0 = np.arange(2)
    yp0 = np.arange(1, 3) * np.inf
    t_span = (0, 1)
    with pytest.raises(ValueError) as excinfo:
        solve_dae(F, t_span, y0, yp0, method=method)
    assert (
        "All components of the initial state `yp0` must be finite."
        in str(excinfo.value)
    )


# if __name__ == "__main__":
#     for params in parameters_check_arguments:
#         test_check_arguments(params)
