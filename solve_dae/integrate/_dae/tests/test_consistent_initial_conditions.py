# see https://www.mathworks.com/help/matlab/ref/decic.html
import pytest
from itertools import product
import numpy as np
from solve_dae.integrate import consistent_initial_conditions

rtol = 1e-5
atol = 1e-5


def fun_implicit(t, y, yp):
    return np.array([
        2 * yp[0] - y[1],
        y[0] + y[1],
    ])

def jac_implicit(t, y, yp):
    Jy = np.array([
        [0, -1],
        [1,  1],
    ])
    Jyp = np.array([
        [2, 0],
        [0, 0],
    ])
    return Jy, Jyp

parameters_implicit = product(
    [([], []), ([0], []), ([], [0])], # fixed_y0, fixed_yp0
    [None, jac_implicit], # jac
)
@pytest.mark.parametrize("fixed_y0_and_fixed_yp0, jac", parameters_implicit)
def test_implicit(fixed_y0_and_fixed_yp0, jac):
    fixed_y0, fixed_yp0 = fixed_y0_and_fixed_yp0
    t0 = 0
    y0 = [1, 0]
    yp0 = [0, 0]

    f0 = fun_implicit(t0, y0, yp0)
    assert not np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)

    y0, yp0, f0 = consistent_initial_conditions(
        fun_implicit, t0, y0, yp0, jac, 
        fixed_y0=fixed_y0, fixed_yp0=fixed_yp0,
        rtol=rtol, atol=atol)
    assert np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)


def fun_algebraic(t, y, yp):
    return np.array([
        2 * y[0] * y[1] - y[0] + 1,
        y[0] + y[1] * y[1] + 2,
    ])

def jac_algebraic(t, y, yp):
    Jy = np.array([
        [2 * y[1] - 1,  2 * y[0]],
        [           1,  2 * y[1]],
    ])
    Jyp = np.zeros((2, 2))
    return Jy, Jyp

parameters_algebraic = product(
    [([], []), ([], [0]), ([], [1]), ([], [0, 1])], # fixed_y0, fixed_yp0
    [None, jac_algebraic], # jac
)
@pytest.mark.parametrize("fixed_y0_and_fixed_yp0, jac", parameters_algebraic)
def test_algebraic(fixed_y0_and_fixed_yp0, jac):
    fixed_y0, fixed_yp0 = fixed_y0_and_fixed_yp0
    t0 = 0
    y0 = [-2, 0.5]
    yp0 = np.random.rand(2)

    f0 = fun_algebraic(t0, y0, yp0)
    assert not np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)

    y0, yp0, f0 = consistent_initial_conditions(
        fun_algebraic, t0, y0, yp0, jac, 
        fixed_y0=fixed_y0, fixed_yp0=fixed_yp0,
        rtol=rtol, atol=atol)
    assert np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)


def fun_differential(t, y, yp):
    return np.array([
        2 * yp[0] * yp[1] - yp[0] + 1,
        yp[0] + yp[1] * yp[1] + 2,
    ])

def jac_differential(t, y, yp):
    Jy = np.zeros((2, 2))
    Jyp = np.array([
        [2 * yp[1] - 1,  2 * yp[0]],
        [            1,  2 * yp[1]],
    ])
    return Jy, Jyp

parameters_differential = product(
    [([], []), ([0], []), ([1], []), ([0, 1], [])], # fixed_y0, fixed_yp0
    [None, jac_differential], # jac
)
@pytest.mark.parametrize("fixed_y0_and_fixed_yp0, jac", parameters_differential)
def test_differential(fixed_y0_and_fixed_yp0, jac):
    fixed_y0, fixed_yp0 = fixed_y0_and_fixed_yp0
    t0 = 0
    y0 = np.random.rand(2)
    yp0 = [-2, 0.75]

    f0 = fun_differential(t0, y0, yp0)
    assert not np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)

    y0, yp0, f0 = consistent_initial_conditions(
        fun_differential, t0, y0, yp0, jac,
        fixed_y0=fixed_y0, fixed_yp0=fixed_yp0,
        rtol=rtol, atol=atol)
    assert np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)


def fun_weissinger(t, y, yp):
    return (
        t * y**2 * yp**3 
        - y**3 * yp**2 
        + t * (t**2 + 1) * yp 
        - t**2 * y
    )

def jac_weissinger(t, y, yp):
    Jy = np.array([
        2 * t * y * yp**3
        - 3 * y**2 * yp**2  
        - t**2 * y,
    ])
    Jyp = np.array([
        3 * t * y**2 * yp**2 
        - 2 * y**3 * yp 
        + t * (t**2 + 1)
    ])
    return Jy, Jyp

parameters_weissinger = product(
    [
        (np.sqrt(3 / 2), 0.5, [], []), 
        (np.sqrt(3 / 2), 0.5, [0], []), 
        (1.2, np.sqrt(6) / 3, [], [0]),
    ], # y0, yp0, fixed_y0, fixed_yp0
    [None, jac_weissinger], # jac
)

@pytest.mark.parametrize("y0_and_yp0_and_fixed_y0_and_fixed_yp0, jac", parameters_weissinger)
def test_weissinger(y0_and_yp0_and_fixed_y0_and_fixed_yp0, jac):
    y0, yp0, fixed_y0, fixed_yp0 = y0_and_yp0_and_fixed_y0_and_fixed_yp0
    t0 = 1.0
    y0 = np.array([y0])
    yp0 = np.array([yp0])

    f0 = fun_weissinger(t0, y0, yp0)
    assert not np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)

    y0, yp0, f0 = consistent_initial_conditions(
        fun_weissinger, t0, y0, yp0, jac,
        fixed_y0=fixed_y0, fixed_yp0=fixed_yp0,
        rtol=rtol, atol=atol)
    assert np.allclose(y0, np.array([np.sqrt(3 / 2)]), rtol=rtol, atol=atol)
    assert np.allclose(yp0, np.array([np.sqrt(6) / 3]), rtol=rtol, atol=atol)
    assert np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)


# if __name__ == "__main__":
#     for params in parameters_implicit:
#         test_implicit(*params)

#     for params in parameters_algebraic:
#         test_algebraic(*params)

#     for params in parameters_weissinger:
#         test_weissinger(*params)
