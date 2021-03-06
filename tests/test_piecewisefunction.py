# -*- coding: utf-8 -*-

""" Tests for the `piecewiselinear` module."""

import pytest
from pytest import approx
import numpy as np
from scipy.integrate import quad
from ufoldy.piecewisefunction import PLF, PCF


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


@pytest.fixture
def small():
    x = [0, 1, 3]
    y = [1, 2, -1]

    norm = 2.5
    integral13 = 1.0
    integral15 = 1.0

    return x, y, norm, integral13, integral15


def test_PLF_init_from_sorted_list(small):
    x, y, *r = small

    p = PLF(x, y)

    for i in range(len(x)):
        assert x[i] == p.x[i]
        assert y[i] == p.y[i]


def test_PLF_init_from_unsorted_list(small):
    x, y, *r = small

    x = x[::-1]
    y = y[::-1]

    p = PLF(x, y)

    xx = x.copy()
    yy = y.copy()

    idx = argsort(xx)

    for i in range(len(x)):
        assert xx[idx[i]] == p.x[i]
        assert yy[idx[i]] == p.y[i]


def test_PLF_init_wrong_xlist():
    x = []
    y = [1, 2, 3]

    with pytest.raises(ValueError):
        p = PLF(x, y)


def test_PLF_init_wrong_ylist():
    x = [1, 2, 3]
    y = []

    with pytest.raises(ValueError):
        p = PLF(x, y)


def test_PLF_init_noncompatlist():
    x = [1, 2]
    y = [1, 2, 3]

    with pytest.raises(ValueError):
        p = PLF(x, y)


def test_PLF_init_nparray(small):
    x, y, *r = small

    xa = np.array(x)
    ya = np.array(y)

    p = PLF(xa, ya)

    for i in range(len(x)):
        assert xa[i] == p.x[i]
        assert ya[i] == p.y[i]


def test_PLF_init_x2darray():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        p = PLF(x, y)


def test_PLF_init_y2darray():
    x = np.array([1, 2, 3, 4])
    y = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        p = PLF(x, y)


def test_PLF_evaluate(small):
    x, y, *r = small

    p = PLF(x, y)

    assert p(-1) == approx(0.0)
    assert p(0) == approx(1.0)
    assert p(0.5) == approx(1.5)
    assert p(1) == approx(2)
    assert p(2) == approx(0.5)
    assert p(7 / 3) == approx(0.0)
    assert p(3) == approx(-1.0)
    assert p(4) == approx(0.0)

    assert p([-1, 0, 0.5, 1, 2, 7 / 3, 3, 4]) == approx(
        np.array([0.0, 1.0, 1.5, 2.0, 0.5, 0.0, -1.0, 0.0])
    )


def test_PLF_init_normalized(small):
    x, y, *r = small

    p = PLF(x, y, normalize=True)

    assert p.norm() == approx(1.0)


def test_PLF_init_normalized_to_number(small):
    x, y, *r = small

    p = PLF(x, y, normalize=True, normvalue=8.0)

    assert p.norm() == approx(8.0)


def test_PLF_init_emptyxy():
    p = PLF([], [])

    assert p.x[0] == approx(0.0)
    assert p.x[1] == approx(1.0)
    assert p.y[0] == approx(0.0)
    assert p.y[1] == approx(0.0)


def test_PLF_xsetter(small):
    x, y, *r = small

    p = PLF(x, y)

    with pytest.raises(ValueError):
        p.x = [-1]


def test_PLF_ysetter(small):
    x, y, *r = small

    p = PLF(x, y)

    p.y = [1, 1, 3]

    assert p.y[0] == approx(1.0)

    assert p(0.5) == approx(1.0)


def test_PLF_ysetter_wrongshape(small):
    x, y, *r = small

    p = PLF(x, y)

    newy = [1, 2]

    with pytest.raises(ValueError):
        p.y = newy


def test_PLF_slopes_setter(small):
    x, y, *r = small

    p = PLF(x, y)

    with pytest.raises(ValueError):
        p.slopes = 1.0


def test_PLF_str(small):
    x, y, *r = small

    p = PLF(x, y)

    strval = "[+0.0000e+00, +1.0000e+00, +3.0000e+00]\n[+1.0000e+00, +2.0000e+00, -1.0000e+00]"

    assert f"{p}" == strval


def test_PLF_norm(small):
    x, y, norm, *r = small

    p = PLF(x, y)

    assert norm == approx(p.norm())


def test_PLF_norm_partial1(small):
    x, y, *r = small

    p = PLF(x, y)

    assert p.norm(1) == approx(1.0)


def test_PLF_norm_partial2(small):
    x, y, *r = small

    p = PLF(x, y)

    assert p.norm(0, 1) == approx(1.5)


def test_PLF_insert_node_before_left(small):
    x, y, *r = small

    p = PLF(x, y)

    p.insert_nodes(-1, -1)

    assert p(-0.5) == approx(0.0)


def test_PLF_insert_node_after_right(small):
    x, y, *r = small

    p = PLF(x, y)

    p.insert_nodes(4, 1)

    assert p(3.5) == approx(0.0)


def test_PLF_insert_node(small):
    x, y, *r = small

    p = PLF(x, y)

    p.insert_nodes(0.5, 1)

    assert p(0.25) == approx(1.0)
    assert p(0.5) == approx(1.0)
    assert p(1.0) == approx(2.0)


def test_PLF_insert_node_noy(small):
    x, y, *r = small

    p = PLF(x, y)

    p.insert_nodes(7 / 3)

    assert p(-1) == approx(0.0)
    assert p(0) == approx(1.0)
    assert p(0.5) == approx(1.5)
    assert p(1) == approx(2)
    assert p(2) == approx(0.5)
    assert p(7 / 3) == approx(0.0)
    assert p(3) == approx(-1.0)
    assert p(4) == approx(0.0)


def test_PLF_insert_nodes(small):
    x, y, *r = small

    p = PLF(x, y)
    q = PLF(x, y)

    newx = [(p.x[i] + p.x[i + 1]) / 2 for i in range(2)]
    newy = p(newx)

    q.insert_nodes(newx, newy)

    evalx = np.linspace(x[0], x[-1], 11)

    for xx in evalx:
        assert p(xx) == approx(q(xx))


def test_PLF_insert_nodes_mixed(small):
    x, y, *r = small

    p = PLF(x, y)

    p.insert_nodes([0.5, 1], [0, 0])

    assert p(-1) == approx(0.0)
    assert p(0) == approx(1.0)
    assert p(0.5) == approx(0.0)
    assert p(1) == approx(0)
    assert p(2) == approx(-0.5)
    assert p(3) == approx(-1.0)
    assert p(4) == approx(0.0)


def test_PLF_insert_nodes_mixed2(small):
    x, y, *r = small

    p = PLF(x, y)

    p.insert_nodes([0.5, 1], [1.5, 2])

    assert p(-1) == approx(0.0)
    assert p(0) == approx(1.0)
    assert p(0.5) == approx(1.5)
    assert p(1) == approx(2)
    assert p(2) == approx(0.5)
    assert p(7 / 3) == approx(0.0)
    assert p(3) == approx(-1.0)
    assert p(4) == approx(0.0)


def test_PLF_insert_nodes_mixed_noy(small):
    x, y, *r = small

    p = PLF(x, y)

    p.insert_nodes([0.5, 1])

    assert p(-1) == approx(0.0)
    assert p(0) == approx(1.0)
    assert p(0.5) == approx(1.5)
    assert p(1) == approx(2)
    assert p(2) == approx(0.5)
    assert p(7 / 3) == approx(0.0)
    assert p(3) == approx(-1.0)
    assert p(4) == approx(0.0)


def test_PCF_insert_nodes_sim(small):
    x, y, *r = small

    p = PCF(x, y)

    x = p.x
    y = p.y

    p.insert_nodes(x)

    assert p.x == approx(x)
    assert p.y == approx(y)


def test_PLF_refine_lin():
    x = np.array([-2, 0, 2])
    y = np.array([1, 2, 1])

    p = PLF(x, y)

    p.refine_lin()

    assert (p.x) == approx(np.array([-2, -1, 0, 1, 2]))


def test_PLF_refine_log():
    x = np.power(10.0, [-2, 0, 2])
    y = np.array([1, 2, 1])

    p = PLF(x, y)

    p.refine_log()

    assert (np.log10(p.x)) == approx(np.array([-2, -1, 0, 1, 2]))


def test_PLF_copy(small):
    x, y, *r = small

    a = PLF(x, y)

    b = a.copy()

    assert np.any(a.x == b.x) and not a.x is b.x
    assert np.any(a.y == b.y) and not a.y is b.y


def test_PLF_flat_noy():
    a = PLF.flat(-1, 1)

    assert a.x[0] == approx(-1.0)
    assert a.x[1] == approx(+1.0)
    assert a.y[0] == approx(0.5)
    assert a.y[1] == approx(0.5)


def test_PLF_flat_y():
    a = PLF.flat(-1, 1, 10)

    assert a.x[0] == approx(-1.0)
    assert a.x[1] == approx(+1.0)
    assert a.y[0] == approx(10.0)
    assert a.y[1] == approx(10.0)


def test_PLF_flat_swapx():
    a = PLF.flat(+1, -1)

    assert a.x[0] == approx(-1.0)
    assert a.x[1] == approx(+1.0)
    assert a.y[0] == approx(0.5)
    assert a.y[1] == approx(0.5)


def test_PCF_eval(small):
    x, y, *r = small

    p = PCF(x, y)

    assert p.y[-1] == p.y[-2]


def test_PCF_eval(small):
    x, y, *r = small

    p = PCF(x, y)

    xeval = [-1.0, +0.0, +0.5, +1.0, +2.0, +3.0, +4.0]
    yeval = [0.0, +1.0, +1.0, +2.0, +2.0, +0.0, 0.0]

    for xe, ye in zip(xeval, yeval):
        assert p(xe) == approx(ye)


def test_PCF_slopes(small):
    x, y, *r = small

    p = PCF(x, y)

    assert p.slopes == approx(np.ones(2))


def test_PCF_norm(small):
    x, y, *r = small

    p = PCF(x, y)

    assert p.norm() == approx(5.0)


def test_PCF_copy(small):
    x, y, *r = small

    a = PCF(x, y)

    b = a.copy()

    assert np.any(a.x == b.x) and not a.x is b.x
    assert np.any(a.y == b.y) and not a.y is b.y


def test_plf_partial_integrals(small):
    x, y, *r = small

    a = PLF(x, y)

    b = PCF([0, 1, 3], [1, 1, 1])

    cnodes = np.array(
        [[0, 1, 3], [-1, 1, 3], [0.5, 1, 3], [-1, 1, 4], [-1, 1, 2], [0.5, 1, 2]]
    )

    cvals = np.array([0.5, 1, np.nan])

    convs = np.array(
        [[1.5, 1.0], [1.5, 1.0], [0.875, 1.0], [1.5, 1.0], [1.5, 1.25], [0.875, 1.25]]
    )

    for c, cint in zip(cnodes, convs):
        b = PCF(c, cvals)
        res = a.partial_integrals(b)
        assert res == approx(cint)


def test_convolute(small):
    x, y, *r = small

    a = PLF(x, y)

    cnodes = np.array(
        [[0, 1, 3], [-1, 1, 3], [0.5, 1, 3], [-1, 1, 4], [-1, 1, 2], [0.5, 1, 2]]
    )
    cvals = np.array([0.5, 1, np.nan])

    convs = np.array([1.75, 1.75, 1.4375, 1.75, 2, 1.6875])

    for c, cint in zip(cnodes, convs):
        b = PCF(c, cvals)
        res = a.convolute(b)
        assert res == approx(cint)
