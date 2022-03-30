# -*- coding: utf-8 -*-

""" Tests for the `LLLLPLF` module."""

import pytest
from pytest import approx
import numpy as np
from scipy.integrate import quad
from ufoldy.llplf import LLPLF


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


@pytest.fixture
def llplf_a():
    x = np.power(10.0, np.array([-4.0, -3.0, -2.0, -1.0]))
    y = np.power(10.0, np.array([1.0, 2.0, 1.0, 2.0]))

    norm = 5.229758509299405

    return x, y, norm


@pytest.fixture
def llplf_b():
    x = np.power(10.0, np.array([-5.0, -3.0, -2.0]))
    y = np.power(10.0, np.array([1.0, -1.0, 2.0]))

    norm = 0.25043551701859884

    return x, y, norm


def test_init_from_sorted_list(llplf_a):
    x, y, *r = llplf_a

    p = LLPLF(x, y)

    for i in range(len(x)):
        assert x[i] == approx(p.x[i])
        assert y[i] == approx(p.y[i])


def test_init_from_unsorted_list(llplf_a):
    x, y, *r = llplf_a

    x = x[::-1]
    y = y[::-1]

    p = LLPLF(x, y)

    xx = x.copy()
    yy = y.copy()

    idx = argsort(xx)

    for i in range(len(x)):
        assert xx[idx[i]] == approx(p.x[i])
        assert yy[idx[i]] == approx(p.y[i])


def test_init_wrong_xlist():
    x = []
    y = [1, 2, 3]

    with pytest.raises(ValueError):
        p = LLPLF(x, y)


def test_init_wrong_ylist():
    x = [1, 2, 3]
    y = []

    with pytest.raises(ValueError):
        p = LLPLF(x, y)


def test_init_noncompatlist():
    x = [1, 2]
    y = [1, 2, 3]

    with pytest.raises(ValueError):
        p = LLPLF(x, y)


def test_init_nparray(llplf_a):
    x, y, *r = llplf_a

    xa = np.array(x)
    ya = np.array(y)

    p = LLPLF(xa, ya)

    for i in range(len(x)):
        assert xa[i] == approx(p.x[i])
        assert ya[i] == approx(p.y[i])


def test_init_x2darray():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        p = LLPLF(x, y)


def test_init_y2darray():
    x = np.array([1, 2, 3, 4])
    y = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        p = LLPLF(x, y)


def test_init_scalar():
    with pytest.raises(ValueError):
        p = LLPLF(1, 1)


def test_init_single_element():
    with pytest.raises(ValueError):
        p = LLPLF([1], [1])

def test_init_nonpositive_nodes():

    with pytest.raises(ValueError):
        p = LLPLF([0, 1],[1, 2])
        p = LLPLF([-1, 1],[1, 2])
        p = LLPLF([1, -1],[1, 2])

def test_init_nonpositive_fvals():

    with pytest.raises(ValueError):
        p = LLPLF([1, 2],[0, 2])
        p = LLPLF([1, 2],[-1, 2])
        p = LLPLF([1, 2],[1, -2])

def test_evaluate(llplf_a):
    x, y, *r = llplf_a

    p = LLPLF(x, y)

    for i in range(len(x)):
        assert y[i] == approx(p(x[i]))


def test_norm(llplf_a):
    x, y, norm = llplf_a

    p = LLPLF(x, y)

    assert norm == approx(p.norm)


def test_init_normalized(llplf_a):
    x, y, norm = llplf_a

    p = LLPLF(x, y, normalize=True)

    assert p.norm == approx(1.0)


def test_init_normalized_to_number(llplf_a):
    x, y, *r = llplf_a

    p = LLPLF(x, y, normalize=True, normvalue=8.0)

    assert p.norm == approx(8.0)


def test_xsetter(llplf_a):
    x, y, *r = llplf_a

    p = LLPLF(x, y)

    with pytest.raises(ValueError):
        p.x = [-1]


def test_ysetter(llplf_a):
    x, y, *r = llplf_a

    p = LLPLF(x, y)

    ynew = [10.0, 1.0, 10.0, 1.0]
    p.y = ynew

    for i in range(len(ynew)):
        assert p.y[i] == approx(ynew[i])


def test_ysetter_wrongshape(llplf_a):
    x, y, *r = llplf_a

    p = LLPLF(x, y)

    ynew = [1, 2]

    with pytest.raises(ValueError):
        p.y = ynew


def test_str(llplf_a):
    x, y, *r = llplf_a

    p = LLPLF(x, y)

    strval = "[+1.0000e-04, +1.0000e-03, +1.0000e-02, +1.0000e-01]\n[+1.0000e+01, +1.0000e+02, +1.0000e+01, +1.0000e+02]"

    assert f"{p}" == strval


def test_insert_point(llplf_a):
    x, y, *r = llplf_a

    p = LLPLF(x, y)

    p.insert_points(5e-4, 1e3)

    for i in range(len(x)):
        assert y[i] == approx(p(x[i]))

    assert p(5e-4) == approx(1e3)


def test_insert_points(llplf_a):
    x, y, *r = llplf_a

    p = LLPLF(x, y)

    p.insert_points([5e-4], [1e3])

    for i in range(len(x)):
        assert y[i] == approx(p(x[i]))

    assert p(5e-4) == approx(1e3)


def test_convolute_same_domain(llplf_a):
    xa, ya, norma = llplf_a

    a = LLPLF(xa, ya)
    b = a * a

    assert b.norm == approx(345.33)


def test_convolute_different_domain(llplf_a, llplf_b):
    xa, ya, norma = llplf_a
    xb, yb, normb = llplf_b

    a = LLPLF(xa, ya)
    b = LLPLF(xb, yb)

    assert (a * b).norm == approx(3.339)


def test_copy(llplf_a):
    x, y, *r = llplf_a

    a = LLPLF(x, y)

    b = a.copy()

    assert np.any(a.x == b.x) and not a.x is b.x
    assert np.any(a.y == b.y) and not a.y is b.y


def test_refine(llplf_a):
    x, y, norm = llplf_a

    xnew = np.power(10.0, np.array([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0]))

    a = LLPLF(x, y)
    a.refine()

    for xi, xin in zip(a.x, xnew):
        assert xi == approx(xin)


def test_flat_noy():
    a = LLPLF.flat(1, 10)

    assert a.x[0] == approx(1.0)
    assert a.x[1] == approx(+10.0)
    assert a.y[0] == approx(1./9)
    assert a.y[1] == approx(1./9)


def test_flat_y():
    a = LLPLF.flat(1, 10, 10)

    assert a.x[0] == approx(1.0)
    assert a.x[1] == approx(+10.0)
    assert a.y[0] == approx(10.0)
    assert a.y[1] == approx(10.0)

    assert a(1) == approx(10.0)
    assert a(4) == approx(10.0)
    assert a(10) == approx(10.0)


def test_flat_swapx():
    a = LLPLF.flat(+10, 1)

    assert a.x[0] == approx(1.0)
    assert a.x[1] == approx(+10.0)
    assert a.y[0] == approx(1./9)
    assert a.y[1] == approx(1./9)
