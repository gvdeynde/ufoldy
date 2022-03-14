# -*- coding: utf-8 -*-

""" Tests for the `reactionrate` module."""

import pytest
from pytest import approx
import numpy as np

from ufoldy.piecewiselinear import PiecewiseLinearFunction as PLF
from ufoldy.reactionrate import ReactionRate


@pytest.fixture
def small():
    x = [0, 1, 3]
    y = [1, 2, -1]

    return x, y


def test_init(small):
    x, y = small

    rr = ReactionRate("test", x, y, 1.23, 0.98)

    assert rr.name == "test"
    assert isinstance(rr.cross_section, PLF)
    assert rr.reaction_rate == approx(1.23)
    assert rr.reaction_rate_error == approx(0.98)
