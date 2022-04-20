# -*- coding: utf-8 -*-

""" Tests for the `reactionrate` module."""

import pytest
from pytest import approx
import numpy as np

from ufoldy.piecewisefunction import PCF, PLF
from ufoldy.reactionrate import ReactionRate


@pytest.fixture
def plf_a():
    x = np.power(10.0, np.array([-4.0, -3.0, -2.0, -1.0]))
    y = np.power(10.0, np.array([1.0, 2.0, 1.0, 2.0]))

    norm = 5.229758509299405

    return x, y, norm

def test_init(plf_a):
    x, y, norm = plf_a

    rr = ReactionRate("test", x, y, 1.23, 0.98)

    assert rr.name == "test"
    assert isinstance(rr.cross_section, PLF)
    assert rr.reaction_rate == approx(1.23)
    assert rr.reaction_rate_error == approx(0.98)
