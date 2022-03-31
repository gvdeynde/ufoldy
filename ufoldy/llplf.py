# -*- coding: utf-8 -*-
"""
This module provides a class to work with loglog piecewiselinear functions
"""

import numpy as np
import numpy.ma as ma


class LLPLF:
    """Class for piecewise linear functions in a loglog scale,
    defined by u=log10(x) positions and corresponding z=log10(y) values
    """

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

        # We store the log10 of the nodes and function values
        _x = np.atleast_1d(np.asarray(x, dtype=float))
        _y = np.atleast_1d(np.asarray(y, dtype=float))

        shapex = _x.shape
        shapey = _y.shape

        if len(shapex) != 1:
            raise ValueError("x needs to be a one-dimensional array")

        if len(shapey) != 1:
            raise ValueError("y needs to be a one-dimensional array")

        if shapex[0] != shapey[0]:
            raise ValueError("x and y should have the same length")
        elif shapex[0] == 1:
            raise ValueError("x and y should have at least length 2")

        # Check for non-positive nodes or function values
        if np.any(_x <= 0):
            raise ValueError("Nodes should be positive!")

        if np.any(_y <= 0):
            raise ValueError("Function values should be positive!")

        # Check for unique nodes
        t, counts = np.unique(_x.round(decimals=15), return_counts=True)

        if np.any(counts > 1):
            raise ValueError("Nodes should be unique!")

        self._u = np.log10(_x)
        self._z = np.log10(_y)

        # Sort the nodes
        sortindices = np.argsort(self._u)
        self._u = self._u[sortindices]
        self._z = self._z[sortindices]

        if normalize:
            self._z -= np.log10(self.norm)
            self._z += np.log10(normvalue)

    @classmethod
    def flat(cls, xmin, xmax, y=None):
        """classmethod to create a LLPLF that is a flat line
        in logscale between xmin and xmax

        Args:
            xmin: float left boundary
            xmax: float right boundary
            y: float height (if None, the height is set to 1.0/(xmax - xmin))

        Returns:
            instance of LLPLF
        """

        if xmax < xmin:
            xmin, xmax = xmax, xmin

        if not y:
            y = 1.0 / (xmax - xmin)

        return cls([xmin, xmax], [y, y])

    def __call__(self, x):
        """Evaluate the LLPLF in x

        Args:
            x (:obj: `list` of :obj:`float` or :obj:`np.ndarray`): the nodes

        Returns
            (:obj:`np.ndarray`): function evaluations in `x`. For values
            outside the range, the function value is set to zero.
        """
        xu = np.log10(np.atleast_1d(x))
        result = np.zeros_like(xu)

        # Calculate the slopes
        slope = np.diff(self._z) / np.diff(self._u)

        # Find acceptable indices, i.e. in the range of definition of LLPLF
        idx = np.where(np.logical_and(xu >= self._u[0], xu < self._u[-1]))

        # Find interval indices
        iidx = np.searchsorted(self._u, xu, side="right") - 1
        iidx = iidx[idx]

        # Calculate the llplf
        result[idx] = np.power(
            10.0, self._z[iidx] + slope[iidx] * (xu[idx] - self._u[iidx])
        )

        # Check for the endpoint
        idx_endp = np.where(xu == self._u[-1])
        result[idx_endp] = np.power(10, self._z[-1])

        return result

    @property
    def norm(self):
        """Calculates the integral of the piecewise function over its domain
        using a closed formula
        """

        # Calculate the slopes
        slope = np.diff(self._z) / np.diff(self._u)

        # Need to take care of m=-1 (or close)
        mask = np.isclose(slope, -1.0, rtol=1e-8)

        slopem = ma.MaskedArray(slope, mask, fill_value=np.nan)

        F = np.power(10, self._z)
        x = np.power(10, self._u)

        # Calculate the integral over each subinterval except the ones where
        # slope is -1 (using masked array). Fill the intervals where slope is
        # -1 with np.nan
        result = (
            F[:-1] / (slopem + 1) * (x[1:] * np.power(x[1:] / x[:-1], slopem) - x[:-1])
        ).filled()

        # Find the indices where slope was -1 and fill those intervals with ln
        # formula
        idx = np.where(mask)[0]
        result[idx] = F[idx] * x[idx] * np.log(x[idx + 1] / x[idx])

        norm = np.sum(result)

        return norm

    @property
    def x(self):
        """Returns the nodes"""
        return np.power(10, self._u)

    @x.setter
    def x(self, x):
        """No self-manipulation of the nodes. Makes things complicated.
        Use the insert_points function to add nodes. Deleting nodes is not
        permitted."""
        raise ValueError("Shouldn't set 'x'")

    @property
    def y(self):
        """Returns the function values at the nodes."""
        return np.power(10, self._z)

    @y.setter
    def y(self, y):
        """Sets new function values."""
        self._z = np.log10(np.atleast_1d(y))
        if self._u.shape != self._z.shape:
            raise ValueError("y has wrong shape")

    def __str__(self):
        """String representation for a piecewiselinear function."""
        retval = ""

        retval += np.array2string(
            np.power(10, self._u),
            separator=", ",
            sign="+",
            formatter={"float_kind": lambda x: "%+.4e" % x},
        )
        retval += "\n"
        retval += np.array2string(
            np.power(10, self._z),
            separator=", ",
            sign="+",
            formatter={"float_kind": lambda x: "%+.4e" % x},
        )
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
        newu = np.log10(newx)
        indices = np.searchsorted(self._u, newu)

        self._u = np.insert(self._u, indices, newu)
        self._z = np.insert(self._z, indices, np.log10(newy))

        return self

    def refine(self):
        """Refine nodes

        Args:

        Returns:
            (:obj: `PiecewiseLinearFunction`): self
        """

        newu = (self._u[:-1] + self._u[1:]) / 2
        newy = self(np.power(10, newu))

        self.insert_points(np.power(10, newu), newy)

        return self

    def __mul__(self, other):
        """Multiplication operator overloading."""

        # unionize the grid
        unionized_u = np.unique(np.concatenate((self._u, other._u)))

        # Since functions are considered to be zero outside their interval,
        # the product is only defined where both are non-zero (i.e. defined).
        min_u = np.max([self._u[0], other._u[0]])
        max_u = np.min([self._u[-1], other._u[-1]])

        unionized_u = unionized_u[unionized_u >= min_u]
        unionized_u = unionized_u[unionized_u <= max_u]

        unionized_x = np.power(10.0, unionized_u)
        return LLPLF(unionized_x, self(unionized_x) * other(unionized_x))

    def copy(self):
        """
        Return a full, deep copy of LLPLF.

        Args:
            none

        Returns:
            (:obj: `LLPLF`): a deep copy of itself
        """
        return LLPLF(
            np.power(10, self._u),
            np.power(10, self._z),
            normalize=True,
            normvalue=self.norm,
        )
