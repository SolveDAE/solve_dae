import numpy as np
import pytest
from itertools import product
from numpy.testing import assert_allclose
from solve_dae.integrate import solve_dae


def f(t, y, yp):
    return yp - np.cos(t)

def event1(t, y, yp):
    return y

def event2(t, y, yp):
    return yp

def event3(t, y, yp):
    return y
event3.terminal = True

def event4(t, y, yp):
    return yp
event4.terminal = True

def solution(t):
    return (
        np.atleast_1d(np.sin(t)),
        np.atleast_1d(np.cos(t)),
    )

events = [
    (
        event1, 
        [np.array([0, np.pi])], 
        [np.array([0, 0])], 
        [np.array([1, -1])], 
    ),
    (
        event2, 
        [np.array([np.pi / 2, np.pi * 3 / 2])], 
        [np.array([1, -1])], 
        [np.array([0, 0])], 
    ),
    (
        [event1], 
        [np.array([0, np.pi])], 
        [np.array([0, 0])], 
        [np.array([1, -1])], 
    ),
    (
        [event2], 
        [np.array([np.pi / 2, np.pi * 3 / 2])], 
        [np.array([1, -1])], 
        [np.array([0, 0])], 
    ),
    (
        [event1, event2],
        [
            np.array([0, 0, np.pi, np.pi]),
            np.array([np.pi / 2, np.pi / 2, np.pi * 3 / 2, np.pi * 3 / 2]),
        ],
        [
            np.array([0, 0, 0, 0]),
            np.array([1, 1, -1, -1]),
        ],
        [
            np.array([1, 1, -1, -1]),
            np.array([0, 0, 0, 0]),
        ],
    ),
    (
        event3, 
        [np.array([0])], 
        [np.array([0])], 
        [np.array([1])], 
    ),
    (
        event4, 
        [np.array([np.pi / 2])], 
        [np.array([1])], 
        [np.array([0])], 
    ),
]

parameters = product(
    ["BDF", "Radau"], # method
    events, # event_options
)

@pytest.mark.parametrize("method, event_options", parameters)
def test_events(method, event_options):
    events, t_events, y_events, yp_events = event_options

    t0 = -0.1
    Dt = 2 * np.pi
    t0, t1 = t0, t0 + Dt
    t_span = (t0, t1)
    y0, yp0 = solution(t0)
    rtol = atol = 1e-12

    sol = solve_dae(f, t_span, y0, yp0, method=method, rtol=rtol, atol=atol, events=events)

    assert_allclose(sol.t_events, t_events, rtol=1e-7, atol=1e-7)
    assert_allclose(sol.y_events, y_events, rtol=1e-7, atol=1e-7)
    assert_allclose(sol.yp_events, yp_events, rtol=1e-7, atol=1e-7)

# if __name__ == "__main__":
#     for param in parameters:
#         test_events(*param)
