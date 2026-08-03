# see https://www.mathworks.com/help/matlab/ref/decic.html
import pytest
from itertools import product
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from solve_dae.integrate import consistent_initial_conditions
from solve_dae.integrate._dae.common import (
    factorize_underdetermined_system, solve_underdetermined_system, norm,
)

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


def fun_implicit_with_args(t, y, yp, scale):
    return np.array([
        2 * yp[0] - scale * y[1],
        y[0] + y[1],
    ])


def test_implicit_with_args_and_default_jacobian():
    # Regression test: the default (finite-difference) Jacobian used to drop
    # `*args` when evaluating `fun` internally, so any caller relying on the
    # default Jacobian together with a non-empty `args` would either hit a
    # TypeError or get a Jacobian for the wrong arguments.
    t0 = 0
    y0 = [1, 0]
    yp0 = [0, 0]
    scale = 2.0

    f0 = fun_implicit_with_args(t0, y0, yp0, scale)
    assert not np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)

    y0, yp0, f0 = consistent_initial_conditions(
        fun_implicit_with_args, t0, y0, yp0, None, None, None,
        rtol, atol, 10, 3, 0.5, scale)
    assert np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)


def jac_implicit_sparse(t, y, yp, sparse_Jy, sparse_Jyp):
    Jy = np.array([[0, -1], [1, 1]], dtype=float)
    Jyp = np.array([[2, 0], [0, 0]], dtype=float)
    if sparse_Jy:
        Jy = csr_matrix(Jy)
    if sparse_Jyp:
        Jyp = csc_matrix(Jyp)
    return Jy, Jyp


@pytest.mark.parametrize("sparse_Jy, sparse_Jyp", [
    (True, True), (True, False), (False, True),
])
def test_sparse_jacobian(sparse_Jy, sparse_Jyp):
    # `jac` may return Jy and/or Jyp as sparse matrices (the same convention
    # used by `solve_dae`/`BDF`/`Radau`); they must be densified internally
    # before the dense, pivoted-QR-based solve.
    t0 = 0
    y0 = [1, 0]
    yp0 = [0, 0]

    def jac(t, y, yp):
        return jac_implicit_sparse(t, y, yp, sparse_Jy, sparse_Jyp)

    f0 = fun_implicit(t0, y0, yp0)
    assert not np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)

    y0, yp0, f0 = consistent_initial_conditions(
        fun_implicit, t0, y0, yp0, jac, rtol=rtol, atol=atol)
    assert np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)


def test_trust_region_triggered():
    # A single scalar equation yp = 1e6 with an initial guess of yp = 0: the
    # raw (undamped) chord correction jumps straight to 1e6, which is far
    # more than twice the current, near-zero state. This must engage the
    # trust region inside `consistent_initial_conditions` instead of taking
    # the raw step directly.
    def fun_trust(t, y, yp):
        return np.array([yp[0] - 1e6])

    def jac_trust(t, y, yp):
        return np.zeros((1, 1)), np.array([[1.0]])

    t0 = 0.0
    y0 = np.array([0.0])
    yp0 = np.array([0.0])

    # Confirm the precondition directly: the undamped correction is huge
    # relative to the current state, so the trust region must trigger.
    f = fun_trust(t0, y0, yp0)
    Jy, Jyp = jac_trust(t0, y0, yp0)
    factors = factorize_underdetermined_system(
        Jy, Jyp, np.array([], dtype=int), np.array([0]))
    dy, dyp = solve_underdetermined_system(f, factors)
    nrm_state = max(norm(np.concatenate((y0, yp0))), norm(np.array([atol])))
    nrm_step = norm(np.concatenate((dy, dyp)))
    assert nrm_step > 2 * nrm_state

    y0, yp0, f0 = consistent_initial_conditions(
        fun_trust, t0, y0, yp0, jac_trust, fixed_y0=[0],
        rtol=rtol, atol=atol)
    assert np.allclose(yp0, [1e6], rtol=rtol, atol=atol)
    assert np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)


def test_too_many_components_fixed():
    # Fixing every component of both y0 and yp0 leaves nothing for the
    # solver to adjust, which must be rejected before any linear algebra is
    # attempted.
    t0 = 0
    y0 = [1, 0]
    yp0 = [0, 0]
    with pytest.raises(ValueError, match="Too many components fixed"):
        consistent_initial_conditions(
            fun_implicit, t0, y0, yp0, jac_implicit,
            fixed_y0=[0, 1], fixed_yp0=[0, 1], rtol=rtol, atol=atol)


def test_too_many_fixed_rank_deficient():
    # `fun_algebraic` has Jyp identically zero (it is a pure algebraic
    # constraint). Fixing both components of y0 leaves only the (rank-zero)
    # Jyp block to solve, which is a distinct, finer-grained "too many
    # fixed" check than the coarse free-variable count in
    # `consistent_initial_conditions` itself -- this one is only detected
    # once `factorize_underdetermined_system` inspects the actual rank of
    # the Jacobian pencil.
    t0 = 0
    y0 = [-2, 0.5]
    yp0 = [0.1, 0.2]
    with pytest.raises(ValueError, match="Too many fixed components"):
        consistent_initial_conditions(
            fun_algebraic, t0, y0, yp0, jac_algebraic,
            fixed_y0=[0, 1], fixed_yp0=None, rtol=rtol, atol=atol)


def test_index_greater_than_one():
    # A Jacobian pencil that is identically singular in both Jy and Jyp,
    # with nothing fixed, cannot be resolved by fixing any component --
    # `factorize_underdetermined_system` must report this as an index > 1
    # problem rather than a fixable "too many fixed" case.
    def fun_zero(t, y, yp):
        return np.zeros(2)

    def jac_zero(t, y, yp):
        return np.zeros((2, 2)), np.zeros((2, 2))

    t0 = 0
    y0 = [1.0, 1.0]
    yp0 = [1.0, 1.0]
    with pytest.raises(ValueError, match="Index greater than one"):
        consistent_initial_conditions(
            fun_zero, t0, y0, yp0, jac_zero, rtol=rtol, atol=atol)


def test_invalid_rtol():
    with pytest.raises(ValueError, match="Relative tolerance must be a positive scalar"):
        consistent_initial_conditions(
            fun_implicit, 0, [1, 0], [0, 0], jac_implicit, rtol=-1e-5, atol=atol)


def test_rtol_below_eps_is_increased(capsys):
    # Too-tight a `rtol` is silently floored to 100 * eps (with a message),
    # rather than being rejected or causing spurious non-convergence.
    t0 = 0
    y0 = [1, 0]
    yp0 = [0, 0]
    y0, yp0, f0 = consistent_initial_conditions(
        fun_implicit, t0, y0, yp0, jac_implicit, rtol=1e-20, atol=atol)
    assert "Relative tolerance increased" in capsys.readouterr().out
    assert np.allclose(f0, np.zeros_like(f0), rtol=rtol, atol=atol)


def test_invalid_atol():
    with pytest.raises(ValueError, match="Absolute tolerance must be positive"):
        consistent_initial_conditions(
            fun_implicit, 0, [1, 0], [0, 0], jac_implicit, rtol=rtol, atol=-1e-5)


def test_convergence_failure():
    # With only a single, undamped chord iteration allowed, a genuinely
    # nonlinear problem started far from its solution cannot converge, and
    # `consistent_initial_conditions` must report this rather than silently
    # returning an inconsistent point.
    t0 = 1.0
    y0 = np.array([1.2])
    yp0 = np.array([1.5])
    with pytest.raises(RuntimeError, match="Convergence failed"):
        consistent_initial_conditions(
            fun_weissinger, t0, y0, yp0, None, None, None,
            rtol, atol, 1, 1, 0.5)


# if __name__ == "__main__":
#     for params in parameters_implicit:
#         test_implicit(*params)

#     for params in parameters_algebraic:
#         test_algebraic(*params)

#     for params in parameters_weissinger:
#         test_weissinger(*params)
