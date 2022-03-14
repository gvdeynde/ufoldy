# -*- coding: utf-8 -*-

""" Tests for the `piecewiselinear` module."""

import pytest
from pytest import approx
import numpy as np
from scipy.integrate import quad
from ufoldy.piecewiselinear import PiecewiseLinearFunction as PLF


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


def test_init_from_sorted_list(small):
    x, y, *r = small

    p = PLF(x, y)

    for i in range(len(x)):
        assert x[i] == p.x[i]
        assert y[i] == p.y[i]


def test_init_from_unsorted_list(small):
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


def test_init_wrong_xlist():
    x = []
    y = [1, 2, 3]

    with pytest.raises(ValueError):
        p = PLF(x, y)


def test_init_wrong_ylist():
    x = [1, 2, 3]
    y = []

    with pytest.raises(ValueError):
        p = PLF(x, y)


def test_init_noncompatlist():
    x = [1, 2]
    y = [1, 2, 3]

    with pytest.raises(ValueError):
        p = PLF(x, y)


def test_init_nparray(small):
    x, y, *r = small

    xa = np.array(x)
    ya = np.array(y)

    p = PLF(xa, ya)

    for i in range(len(x)):
        assert xa[i] == p.x[i]
        assert ya[i] == p.y[i]


def test_init_x2darray():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        p = PLF(x, y)


def test_init_y2darray():
    x = np.array([1, 2, 3, 4])
    y = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        p = PLF(x, y)


def test_evaluate(small):
    x, y, *r = small

    p = PLF(x, y)

    assert p(0) == approx(1.0)
    assert p(0.5) == approx(1.5)
    assert p(1) == approx(2)
    assert p(2) == approx(0.5)
    assert p(7 / 3) == approx(0.0)
    assert p(3) == approx(-1.0)


def test_init_normalized(small):
    x, y, *r = small

    p = PLF(x, y, normalize=True)

    assert p.norm == approx(1.0)


def test_init_normalized_to_number(small):
    x, y, *r = small

    p = PLF(x, y, normalize=True, normvalue=8.0)

    assert p.norm == approx(8.0)


def test_init_emptyxy():
    p = PLF([], [])

    assert p.x[0] == approx(0.0)
    assert p.x[1] == approx(1.0)
    assert p.y[0] == approx(0.0)
    assert p.y[1] == approx(0.0)


def test_xsetter(small):
    x, y, *r = small

    p = PLF(x, y)

    with pytest.raises(ValueError):
        p.x = [-1]


def test_ysetter(small):
    x, y, *r = small

    p = PLF(x, y)

    p.y = [1, 1, 3]

    assert p.y[0] == approx(1.0)

    assert p(0.5) == approx(1.0)


def test_ysetter_wrongshape(small):
    x, y, *r = small

    p = PLF(x, y)

    newy = [1, 2]

    with pytest.raises(ValueError):
        p.y = newy


def test_str(small):
    x, y, *r = small

    p = PLF(x, y)

    strval = "[+0., +1., +3.]\n[+1., +2., -1.]"

    assert f"{p}" == strval


def test_norm(small):
    x, y, norm, *r = small

    p = PLF(x, y)

    assert norm == approx(p.norm)


def test_insert_point(small):
    x, y, *r = small

    p = PLF(x, y)

    p.insert_points(0.5, 1)

    assert p(0.25) == approx(1.0)
    assert p(0.5) == approx(1.0)
    assert p(1.0) == approx(2.0)


def test_insert_points(small):
    x, y, *r = small

    p = PLF(x, y)
    q = PLF(x, y)

    newx = [(p.x[i] + p.x[i + 1]) / 2 for i in range(2)]
    newy = p(newx)

    q.insert_points(newx, newy)

    evalx = np.linspace(x[0], x[-1], 11)

    for xx in evalx:
        assert p(xx) == approx(q(xx))


def test_convolute_same_domain(small):
    x, y, *r = small

    p = PLF(x, y)

    qx = [0, 1, 2, 3]
    qy = [1, 2, -1, -2]

    q = PLF(qx, qy)

    conv, errest = quad(lambda x: p(x) * q(x), 0, 3)

    assert conv == approx(p.convolute(q))

    assert conv == approx(q.convolute(p))


def test_convolute_different_domain(small):
    x, y, *r = small

    p = PLF(x, y)

    qx = [-1, 1, 3, 5]
    qy = [1, 2, -1, -2]

    q = PLF(qx, qy)

    conv, errest = quad(lambda x: p(x) * q(x), -1, 5)

    assert conv == approx(p.convolute(q))

    assert conv == approx(q.convolute(p))
