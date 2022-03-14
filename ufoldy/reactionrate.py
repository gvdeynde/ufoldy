# -*- coding: utf-8 -*-
"""
This module provides a class to work with piecewiselinear cross sections.
"""
import numpy as np
from dataclasses import dataclass, field, InitVar
from ufoldy.piecewiselinear import PiecewiseLinearFunction


@dataclass
class ReactionRate:
    """Dataclass to hold reaction rates."""

    name: str = "unknown reaction rate"

    x: InitVar[np.ndarray] = field(default=np.array([]))
    y: InitVar[np.ndarray] = field(default=np.array([]))

    cross_section: PiecewiseLinearFunction = field(init=False)
    reaction_rate: float = 0.0
    reaction_rate_error: float = 0.0

    def __post_init__(self, x, y):
        self.cross_section = PiecewiseLinearFunction(x, y)
