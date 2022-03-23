# -*- coding: utf-8 -*-
"""
This module provides a class to work with piecewiselinear functions
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import roots_legendre


def refine_log(x):
    """
    Function that refines a vector by inserting a new point between two
    existing points in log-equidistant way. This function returns the extra
    nodes, not the whole set.

    Args:
        x (obj:`np.ndarray`): the nodes

    Returns:
        (obj:`np.ndarray`): the extra nodes
    """

    logx = np.log10(x)
    newlogx = (logx[1:] + logx[:-1]) / 2

    return np.power(10, newlogx)


def refine_lin(x):
    """
    Function that refines a vector by inserting a new point between two
    existing points in equidistant way. This function returns the extra nodes,
    not the whole set.

    Args:
        x (obj:`np.ndarray`): the nodes

    Returns:
        (obj:`np.ndarray`): the extra nodes
    """

    return (x[1:] + x[:-1]) / 2


class PiecewiseLinearFunction:
    """Class for piecewise linear functions, defined by x positions and corresponding y values"""

    # Since we work with piecewise linear functions, a Gauss-Legendre quadrature of order 2 is correct
    # All quadrature is done based on this rule.
    nodes, weights = roots_legendre(2)

    @staticmethod
    def interpolate(x, y):
        return interp1d(x, y, kind="linear", bounds_error=False, fill_value=0.0)

    def __init__(self, x, y, normalize=False, normvalue=1.0):
        """
        Args:
            x (:obj: `list` of :obj:`float` or :obj:`np.ndarray`): the nodes
            y (:obj: `list` of :obj:`float` or :obj:`np.ndarray`): the values
            at the nodes
            normalize (bool): if `True`, the y values are normalized such that
            :math:`\int f(x) dx = C` where `C` is the normalisation value. Default value is `False`.
            normvalue (float): the normalisation value `C` to be used.
        """

        self._x = np.copy(np.asarray(x, dtype=float))
        self._y = np.copy(np.asarray(y, dtype=float))

        shapex = self._x.shape
        shapey = self._y.shape

        if len(shapex) != 1:
            raise ValueError("x needs to be a one-dimensional array")

        if len(shapey) != 1:
            raise ValueError("y needs to be a one-dimensional array")

        if shapex[0] != shapey[0]:
            raise ValueError("x and y should have the same length")

        if shapex[0] == 0:
            # Default PLF using [0, 1] and [0, 0]
            self._x = np.array([0, 1])
            self._y = np.array([0, 0])
            normalize = False

        sortindices = np.argsort(self._x)
        self._x = self._x[sortindices]
        self._y = self._y[sortindices]

        self._fun = self.interpolate(self._x, self._y)

        if normalize:
            self._y /= self.norm
            self._y *= normvalue
            self._fun = self.interpolate(self._x, self._y)

    def __call__(self, x):
        return self._fun(x)

    @property
    def norm(self):
        """Calculates the integral of the piecewise function over its domain
        using a second order Gauss-Legendre quadrature which is fully correct.
        """
        z = (
            np.outer((self._x[1:] - self._x[:-1]), (self.nodes + 1) / 2)
        ).flatten() + np.outer(self._x[:-1], [1, 1]).flatten()
        funz = self._fun(z)
        return np.dot(
            (self._x[1:] - self._x[:-1]) / 2,
            funz.reshape((int(funz.shape[0] / 2), 2)).sum(axis=1),
        )

    @property
    def x(self):
        """Returns the nodes"""
        return self._x

    @x.setter
    def x(self, x):
        """No self-manipulation of the nodes. Makes things complicated.
        Use the insert_points function to add nodes. Deleting nodes is not
        permitted."""
        raise ValueError("Shouldn't set 'x'")

    @property
    def y(self):
        """Returns the function values at the nodes."""
        return self._y

    @y.setter
    def y(self, y):
        """Sets new function values."""
        self._y = np.asarray(y)
        if self._x.shape != self._y.shape:
            raise ValueError("y has wrong shape")
        else:
            self._y = y
            self._fun = self.interpolate(self._x, self._y)

    def __str__(self):
        """String representation for a piecewiselinear function."""
        retval = ""

        retval += np.array2string(self._x, separator=", ", sign="+")
        retval += "\n"
        retval += np.array2string(self._y, separator=", ", sign="+")
        return retval

    def insert_points(self, newx, newy):
        """Insert new nodes and corresponding function values. Important: the
        function does not re-normalize the piecewiselinear function!

        Args:
            newx (:obj: `list` of :obj:`float` or :obj:`np.ndarray`): the nodes
            to be added
            newy (:obj: `list` of :obj:`float` or :obj:`np.ndarray`): the
            corresponding function values.

        Returns:
            (:obj: `PiecewiseLinearFunction`): self
        """
        indices = np.searchsorted(self._x, newx)

        self._x = np.insert(self._x, indices, newx)
        self._y = np.insert(self._y, indices, newy)

        self._fun = self.interpolate(self._x, self._y)

        return self

    def refine(self, refiner=refine_log):
        """Insert new nodes according to the Strategy set in `refiner`

        Args:
            refiner: (:fun:): a function that takes a np.ndarray of sorted
            nodes and returns the extra nodes according to some strategy.
            Default: `refine_log`: logarithmic refining.

        Returns:
            (:obj: `PiecewiseLinearFunction`): self
        """

        newx = refiner(self._x)
        newy = self(newx)

        self.insert_points(newx, newy)

        return self

    def convolute(self, pwlf2):
        """Calculates a convolution of two piecewiselinearfunctions
        Args:
            pwlf2: (:obj:`PiecewiseLinearFunction`) the second
            PiecewiseLinearFunction

        Returns:
            float: the result of the convolution integral
        """
        # First unionize the nodes
        unionized_x = np.unique(np.concatenate((self._x, pwlf2.x)))

        # Get upper and lower integration boundary: only integrate where both functions are defined
        min_x = np.max([self._x[0], pwlf2.x[0]])
        max_x = np.min([self._x[-1], pwlf2.x[-1]])

        unionized_x = unionized_x[unionized_x >= min_x]
        unionized_x = unionized_x[unionized_x <= max_x]

        # Calculate the integral using second order Gauss-Legendre quadrature
        # (which is exact for a piecewise quadratic function (product of two
        # piecewise linear functions)

        # Get the Gauss-Legendre nodes per interval
        z = (
            np.outer((unionized_x[1:] - unionized_x[:-1]), (self.nodes + 1) / 2)
        ).flatten() + np.outer(unionized_x[:-1], [1, 1]).flatten()

        # Evaluate the product of the two piecewise linear functions on these
        # nodes
        funz = self._fun(z) * pwlf2(z)

        # Evaluate the Gauss-Legendre weighted sum
        return np.dot(
            (unionized_x[1:] - unionized_x[:-1]) / 2,
            funz.reshape((int(funz.shape[0] / 2), 2)).sum(axis=1),
        )

    def copy(self):
        """
        Return a full, deep copy of PLF.

        Args:
            none

        Returns:
            (:obj: `PiecewiseLinearFunction`): a deep copy of itself
        """
        return PiecewiseLinearFunction(
            self._x, self._y, normalize=True, normvalue=self.norm
        )
