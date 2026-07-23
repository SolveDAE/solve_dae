import numpy as np
import pytest
from scipy.linalg import eig, cdf2rdf
from solve_dae.integrate._dae.radau import butcher_tableau, radau_constants


@pytest.mark.parametrize("s", [1, 3, 5, 7, 9])
def test_eigendecomposition_reconstructs_A(s):
    """The real block-diagonal decomposition Gammas/T computed inside
    `radau_constants` must reconstruct the Butcher matrix A exactly, since
    the collocation system is solved entirely in the transformed basis.
    This used to be asserted on every `RadauDAE` instantiation; it is a
    property of the (s-dependent) constants, so a parametrized test is a
    better home for it.
    """
    A, b, c, p = butcher_tableau(s)

    lambdas, Q = eig(A)
    idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idx]
    Q = Q[:, idx]
    for i in range(s):
        Q[:, i] /= Q[-1, i]

    Gammas, T = cdf2rdf(lambdas, Q)
    TI = np.linalg.inv(T)

    assert_allclose = np.testing.assert_allclose
    assert_allclose(Q @ np.diag(lambdas) @ np.linalg.inv(Q), A, atol=1e-12)
    assert_allclose(np.linalg.inv(Q) @ A @ Q, np.diag(lambdas), atol=1e-12)
    assert_allclose(T @ Gammas @ TI, A, atol=1e-12)
    assert_allclose(TI @ A @ T, Gammas, atol=1e-12)


@pytest.mark.parametrize("s", [1, 3, 5, 7, 9])
def test_radau_constants_runs(s):
    # smoke test that radau_constants itself still succeeds for a range of
    # (odd) stage counts now that the reconstruction is only checked above
    radau_constants(s)
