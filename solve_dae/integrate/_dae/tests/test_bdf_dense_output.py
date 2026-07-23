"""Verify `BdfDenseOutput` (the Newton backward-difference dense-output
interpolant) against an independent, differently-implemented oracle:
`scipy.interpolate.KroghInterpolator`, which builds and differentiates the
same mathematical object (a Newton divided-difference polynomial) through a
completely separate code path.
"""
from math import comb
import numpy as np
import pytest
from scipy.interpolate import KroghInterpolator
from solve_dae.integrate._dae.bdf import BdfDenseOutput


def reconstruct_grid_values(D, order):
    """Invert backward differences: D[j] = nabla^j y_n implies
    y_{n-k} = sum_{j=0}^{k} (-1)^j * C(k, j) * D[j].
    """
    n = D.shape[1]
    y = np.zeros((order + 1, n), dtype=D.dtype)
    for k in range(order + 1):
        for j in range(k + 1):
            y[k] += (-1)**j * comb(k, j) * D[j]
    return y  # y[k] == y_{t_old - k*h}


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("n", [1, 3])
def test_matches_krogh_interpolator(order, n):
    rng = np.random.default_rng(order * 100 + n)
    h = rng.uniform(0.1, 2.0)
    t_old = rng.uniform(-5, 5)
    t = t_old + h  # BdfDenseOutput's "t" is the right end of the window
    D = rng.normal(scale=rng.uniform(0.1, 5), size=(order + 1, n))

    dense_output = BdfDenseOutput(t_old, t, h, order, D)

    y_grid = reconstruct_grid_values(D, order)  # y_grid[k] = y at t - k*h
    t_nodes = t - h * np.arange(order + 1)

    # points inside, on, and outside (extrapolated) the interpolation window
    t_test = np.concatenate([
        t_nodes,
        t + h * rng.uniform(-3, 3, size=5),
    ])

    y, yp = dense_output(t_test)

    for comp in range(n):
        krogh = KroghInterpolator(t_nodes, y_grid[:, comp])
        y_ref = krogh(t_test)
        yp_ref = krogh.derivative(t_test, der=1)
        np.testing.assert_allclose(y[comp], y_ref, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(yp[comp], yp_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("order", [1, 3, 5])
def test_matches_krogh_interpolator_complex(order):
    # BDFDAE supports the complex domain (support_complex=True); the
    # interpolant must handle complex D transparently.
    rng = np.random.default_rng(order)
    h = 0.3
    t_old = 1.0
    t = t_old + h
    D = (rng.normal(size=(order + 1, 2)) + 1j * rng.normal(size=(order + 1, 2)))

    dense_output = BdfDenseOutput(t_old, t, h, order, D)
    y_grid = reconstruct_grid_values(D, order)
    t_nodes = t - h * np.arange(order + 1)
    t_test = t + h * np.array([-2.5, -0.5, 0.5, 2.5])

    y, yp = dense_output(t_test)
    for comp in range(2):
        krogh = KroghInterpolator(t_nodes, y_grid[:, comp])
        np.testing.assert_allclose(y[comp], krogh(t_test), atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(yp[comp], krogh.derivative(t_test, der=1), atol=1e-6, rtol=1e-6)


def test_scalar_input_returns_1d_arrays():
    D = np.arange(3 * 2, dtype=float).reshape(3, 2)
    dense_output = BdfDenseOutput(0.0, 1.0, 1.0, 2, D)
    y, yp = dense_output(np.array(0.5))
    assert y.shape == (2,)
    assert yp.shape == (2,)


def test_vector_input_returns_2d_arrays():
    D = np.arange(3 * 2, dtype=float).reshape(3, 2)
    dense_output = BdfDenseOutput(0.0, 1.0, 1.0, 2, D)
    y, yp = dense_output(np.array([0.2, 0.5, 0.8]))
    assert y.shape == (2, 3)
    assert yp.shape == (2, 3)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_yp_loses_one_order_of_accuracy_relative_to_y(order):
    # Structural property documented in BdfDenseOutput's docstring:
    # differentiating a degree-`order` interpolant of y (accurate to
    # O(h**(order+1))) generically only approximates y' to O(h**order).
    t_n = 1.0
    errs_y, errs_yp = [], []
    hs = [0.2, 0.1, 0.05, 0.025]
    for h in hs:
        t_nodes = t_n - h * np.arange(order + 1)
        y_vals = np.sin(t_nodes)
        # backward differences: D[j] = nabla^j y_n, built from y_vals via
        # the inverse of reconstruct_grid_values (repeated first differences)
        D = np.zeros((order + 1, 1))
        col = y_vals.copy()
        D[0, 0] = col[0]
        for j in range(1, order + 1):
            col = col[:-1] - col[1:]
            D[j, 0] = col[0]

        dense_output = BdfDenseOutput(t_n - h, t_n, h, order, D)
        t_eval = np.array([t_n + 0.3 * h])
        y, yp = dense_output(t_eval)
        errs_y.append(abs(y[0, 0] - np.sin(t_eval[0])))
        errs_yp.append(abs(yp[0, 0] - np.cos(t_eval[0])))

    errs_y = np.array(errs_y)
    errs_yp = np.array(errs_yp)
    rate_y = np.mean(np.log(errs_y[:-1] / errs_y[1:]) / np.log(2))
    rate_yp = np.mean(np.log(errs_yp[:-1] / errs_yp[1:]) / np.log(2))

    assert rate_y == pytest.approx(order + 1, abs=0.3)
    assert rate_yp == pytest.approx(order, abs=0.3)
