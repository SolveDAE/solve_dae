from itertools import product
from numpy.testing import suppress_warnings
import pytest
from solve_dae.integrate import solve_dae


parameters_stiff = ["BDF", "Radau"]


@pytest.mark.slow
@pytest.mark.parametrize("method", parameters_stiff)
def test_integration_robertson_ode(method):
    def fun_robertson(t, state):
        x, y, z = state
        return [
            -0.04 * x + 1e4 * y * z,
            0.04 * x - 1e4 * y * z - 3e7 * y * y,
            3e7 * y * y,
        ]
    
    def F_robertson(t, state, statep):
        return statep - fun_robertson(t, state)
    
    rtol = 1e-6
    atol = 1e-6
    y0 = [1e4, 0, 0]
    yp0 = fun_robertson(0, y0)
    tspan = [0, 1e8]

    if method == "BDF":
        for NDF_strategy, max_order in product(
            ["stability", "efficiency", None], # NDF_strategy
            [1, 2, 3, 4, 5, 6], # max_order
        ):
            with suppress_warnings() as sup:
                sup.filter(UserWarning,
                        "Choosing `max_order = 6` is not recommended due to its "
                        "poor stability properties.")
                res = solve_dae(F_robertson, tspan, y0, yp0, rtol=rtol,
                                atol=atol, method=method, max_order=max_order,
                                NDF_strategy=NDF_strategy)
                
                # If the stiff mode is not activated correctly, these numbers will be much
                # bigger (see max_order=1 case)
                if max_order == 1:
                    # `max_order=1` forces plain adaptive implicit Euler for
                    # the ~7500 steps this run needs, which is the one
                    # regime where BDF's controller_deadband (see bdf.py)
                    # trades a modest increase in Newton solves (~15200 vs
                    # ~14000 nfev here) for a much larger cut in LU
                    # factorizations (~160 vs ~3500 nlu) -- a good trade for
                    # any problem where factorization cost isn't negligible,
                    # which is not the case for this 3-variable system.
                    assert res.nfev < 15500
                    assert res.nlu < 3500
                else:
                    # `max_order=2` with the geometric-mean-smoothed Newton
                    # rate estimate (mirroring RadauDAE, see solve_bdf_system
                    # in bdf.py) needs marginally more steps here (~2670 vs
                    # ~2624 nfev before smoothing was added) in exchange for
                    # fewer/no-worse `njev`/`nlu`; not worth chasing away.
                    assert res.nfev < 2700
                    assert res.nlu < 430
                assert res.njev < 50

    else: # Radau
        for stages, continuous_error_weight, newton_iter_embedded in product(
            [3, 5, 7], # stages
            [0.0, 0.5, 1.0], # continuous_error_weight
            [0, 1, 2], # newton_iter_embedded
        ):
            res = solve_dae(F_robertson, tspan, y0, yp0, rtol=rtol,
                            atol=atol, method=method, stages=stages,
                            continuous_error_weight=continuous_error_weight,
                            newton_iter_embedded=newton_iter_embedded)

            # If the stiff mode is not activated correctly, these numbers will be much bigger
            assert res.nfev < 3300
            assert res.njev < 150
            assert res.nlu < 370


@pytest.mark.slow
@pytest.mark.parametrize("method", parameters_stiff)
def test_integration_robertson_dae(method):
    def F_robertson(t, y, yp):
        y1, y2, y3 = y
        y1p, y2p, y3p = yp

        return [
            y1p - (-0.04 * y1 + 1e4 * y2 * y3),
            y2p - (0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2),
            y1 + y2 + y3 - 1,
        ]
    
    rtol = 1e-6
    atol = 1e-6
    y0 = [1, 0, 0]
    yp0 = [-0.04, 0.04, 0]
    tspan = [0, 1e8]

    if method == "BDF":
        for NDF_strategy, max_order in product(
            ["stability", "efficiency", None], # NDF_strategy
            [1, 2, 3, 4, 5, 6], # max_order
        ):
            with suppress_warnings() as sup:
                sup.filter(UserWarning,
                        "Choosing `max_order = 6` is not recommended due to its "
                        "poor stability properties.")
                res = solve_dae(F_robertson, tspan, y0, yp0, rtol=rtol,
                                atol=atol, method=method, max_order=max_order,
                                NDF_strategy=NDF_strategy)
                
                # If the stiff mode is not activated correctly, these numbers will be much 
                # bigger (see max_order=1 case)
                if method == "BDF" and max_order == 1:
                    assert res.nfev < 5000
                    assert res.nlu < 1100
                else:
                    assert res.nfev < 1610
                    assert res.nlu < 210
                assert res.njev < 32

    else: # Radau
        for stages, continuous_error_weight, newton_iter_embedded in product(
            [3, 5, 7], # stages
            [0.0, 0.5, 1.0], # continuous_error_weight
            [0, 1, 2], # newton_iter_embedded
        ):
            res = solve_dae(F_robertson, tspan, y0, yp0, rtol=rtol,
                            atol=atol, method=method, stages=stages,
                            continuous_error_weight=continuous_error_weight,
                            newton_iter_embedded=newton_iter_embedded)

            # If the stiff mode is not activated correctly, these numbers will be much bigger
            assert res.nfev < 2600
            assert res.njev < 90
            assert res.nlu < 450


# if __name__ == "__main__":
#     for param in parameters_stiff:
#         test_integration_robertson_ode(param)
#         test_integration_robertson_dae(param)
