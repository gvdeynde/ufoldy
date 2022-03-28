# -*- coding: utf-8 -*-
"""
This module provides a class to work with loglog piecewiselinear functions
"""

import numpy as np


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
        self._u = np.log10(np.asarray(x, dtype=float))
        self._z = np.log10(np.asarray(y, dtype=float))

        shapeu = self._u.shape
        shapez = self._z.shape

        if len(shapeu) != 1:
            raise ValueError("x needs to be a one-dimensional array")

        if len(shapez) != 1:
            raise ValueError("y needs to be a one-dimensional array")

        if shapeu[0] != shapez[0]:
            raise ValueError("x and y should have the same length")

        # Check for unique nodes
        t, counts = np.unique(self._u, return_counts=True)

        if np.any(counts != np.ones_like(self._u)):
            raise ValueError("Nodes should be unique!")

        # Sort the nodes
        sortindices = np.argsort(self._u)
        self._u = self._u[sortindices]
        self._z = self._z[sortindices]

        if normalize:
            self._z /= self.norm
            self._z *= normvalue

    @classmethod
    def flat(cls, xmin, xmax, y=None):
        """classmethod to create a LLPLF that is a flat line
        in logscale between xmin and xmax

        Args:
            xmin: float left boundary
            xmax: float right boundary
            y: float height (if None, the height is set to 1.0/(xmax - xmin))

        Returns:
            instance of PiecewiseLinearFunction
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
        """
        xu = np.log10(x)
        result = np.zeros_like(x)

        # slope = np.zeros_like(self._u)
        slope = np.diff(self._z) / np.diff(self._u)
        print("slopes", slope)

        # Find acceptable indices, i.e. in the range of definition of LLPLF
        idx = np.where(np.logical_and(xu >= self._u[0], xu < self._u[-1]))
        print("here")
        print(self._u)
        print(xu)
        print("idx", idx, xu[idx])

        # Find interval indices
        iidx = np.searchsorted(self._u, xu, side="right") - 1
        iidx = iidx[idx]
        print("iidx", iidx)

        # Calculate the llplf
        result[idx] = np.power(
            10.0, self._z[iidx] + slope[iidx] * (xu[idx] - self._u[iidx])
        )

        # Check for the endpoint
        idx_endp = np.where(xu == self._u[-1])
        result[idx_endp] = np.power(10, self._z[-1])

        print("end point", idx_endp)

        return result

    @property
    def norm(self):
        """Calculates the integral of the piecewise function over its domain
        using a closed formula
        """
        m = np.log10(self._z[1:] / self._z[:-1]) / np.log10(self._u[1:] / self.u[:-1])

        m_minus_one = np.isclose(m, -1)

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
        self._z = np.log10(np.asarray(y))
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
        newy = self(newu)

        self.insert_points(np.power(10, newu), newy)

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
            self._u, self._z, normalize=True, normvalue=self.norm
        )
