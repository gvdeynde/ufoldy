# -*- coding: utf-8 -*-
"""
This module provides a class to work with piecewiselinear functions
"""

from abc import ABC, abstractmethod
import numpy as np


class PiecewiseFunction(ABC):

    """Class for piecewise functions, defined by x positions and corresponding y values.
    It's an abstract base class which needs specialisation for piecewise
    constant and piecewise linear functions.
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

        if normalize:
            self._y /= self.norm()
            self._y *= normvalue

    @classmethod
    def flat(cls, xmin, xmax, y=None):
        """classmethod to create a PLF that is a flat line
        between xmin and xmax

        Args:
            xmin: float left boundary
            xmax: float right boundary
            y: float height (if None, the height is set to 1.0/(xmax - xmin))

        Returns:
            instance of PiecewiseFunction
        """

        if xmax < xmin:
            xmin, xmax = xmax, xmin

        if not y:
            y = 1.0 / (xmax - xmin)

        return cls([xmin, xmax], [y, y])

    @abstractmethod
    def __call__(self, x):
        pass

    @property
    @abstractmethod
    def slopes(self):
        pass

    @slopes.setter
    def slopes(self, val):
        raise ValueError("slopes is a read-only property")

    @abstractmethod
    def norm(self, left, right):
        pass

    @property
    def nnodes(self):
        return len(self._x)

    @nnodes.setter
    def nnodes(self, val):
        raise ValueError("nnodes is a read-only property")

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

    def __str__(self):
        """String representation for a piecewiselinear function."""
        retval = ""

        retval += np.array2string(
            self._x,
            separator=", ",
            sign="+",
            formatter={"float_kind": lambda x: "%+.4e" % x},
        )
        retval += "\n"
        retval += np.array2string(
            self._y,
            separator=", ",
            sign="+",
            formatter={"float_kind": lambda x: "%+.4e" % x},
        )
        return retval

    def insert_nodes(self, newx, newy=None):
        """Insert new nodes and corresponding function values. Important: the
        function does not re-normalize the piecewiselinear function!

        Args:
            newx (:obj: `list` of :obj:`float` or :obj:`np.ndarray`): the nodes
            to be added
            newy (:obj: `list` of :obj:`float` or :obj:`np.ndarray`): the
            corresponding function values.

        Returns:
            (:obj: `PiecewiseFunction`): self
        """

        newx = np.atleast_1d(newx)

        # If new y values not given, calculate them with linear interpolation
        if newy is None:
            newy = np.atleast_1d(self(newx))
        else:
            # Make sure newy is array like
            newy = np.atleast_1d(newy)

        # Naive implementation
        # First overwrite the existing nodes
        existing_idx = {}
        for i, x in enumerate(newx):
            idx = np.where(self._x == x)[0]
            if len(idx) > 0:
                # the node is already present
                existing_idx[i] = idx[0]
                self._y[idx] = newy[i]

        # Now insert all new nodes
        if len(existing_idx) != len(newx):
            mask = np.ones(newx.size, dtype=bool)
            mask[list(existing_idx.keys())] = False

            idx = np.searchsorted(self._x, newx[mask])
            self._x = np.insert(self._x, idx, newx[mask])
            self._y = np.insert(self._y, idx, newy[mask])

        return self

    def refine_log(self):
        """
        Refine the nodes in a logarithmic way
        """

        logx = np.log10(self._x)
        newlogx = (logx[1:] + logx[:-1]) / 2

        newx = np.power(10.0, newlogx)

        self.insert_nodes(newx)

    def refine_lin(self):
        """
        Refine the nodes in a linear way
        """

        newx = (self._x[1:] + self._x[:-1]) / 2.0

        self.insert_nodes(newx)

    @abstractmethod
    def copy(self):
        pass


class PLF(PiecewiseFunction):
    """Class that implements Piecewise Linear Functions. Derived from
    PiecewiseFunction.
    """

    # def _calculate_slopes(self):
    # return np.diff(self._y) / np.diff(self._x)

    def __call__(self, x):
        # Initialise
        result = np.zeros_like(np.atleast_1d(x))

        # Left end-point
        result[x == self._x[0]] = self._y[0]

        # Right end-point
        result[x == self._x[-1]] = self._y[-1]

        # Interpolate
        idx = np.searchsorted(self._x, x)

        # this is a fixup to avoid problems with left/right endpoint. It just
        # replaces a too high index with arbitrary 0. idx for that position is
        # never needed in end result (where in next line) but is needed to
        # avoid an index error
        idx = np.where(np.logical_or(idx < 1, idx >= len(self._x)), 1, idx)

        result = np.where(
            np.logical_and(x > (self._x[0]), (x < self._x[-1])),
            self._y[idx - 1] + self.slopes[idx - 1] * (x - self._x[idx - 1]),
            result,
        )

        return result

    @PiecewiseFunction.slopes.getter
    def slopes(self):
        return np.diff(self._y) / np.diff(self._x)

    def norm(self, left=0, right=None):
        if not right:
            right = len(self._x) - 1

        return np.trapz(self._y[left : right + 1], self._x[left : right + 1])

    def copy(self):
        """
        Return a full, deep copy of PLF.

        Args:
            none

        Returns:
            (:obj: `PLF`): a deep copy of itself
        """
        return PLF(self._x, self._y, normalize=True, normvalue=self.norm())

    def partial_integrals(self, pcf):
        """
        Return the partial integrals of the PLF between the nodes of the PCF

        Args:
            pcf: (:obj: `PCF`): the piecewise constant function

        Returns:
            (:obj: nd.array(float)): numpy array containing the partial integrals
        """

        if not isinstance(pcf, PCF):
            raise ValueError("I can only convolute a PLF with a PCF")

        # Insert PCF nodes into a local copy of plf
        lplf = self.copy()
        lplf.insert_nodes(pcf.x)

        # Insert left and right end-point of plf in local copy of PCF
        lpcf = pcf.copy()
        lpcf.insert_nodes([self._x[0], self._x[-1]])
        # Set lpcf nodes left and right to zero
        lpcf.y = np.where(
            np.logical_or(lpcf.x < self._x[0], lpcf.x >= self._x[-1]), 0.0, lpcf.y
        )

        result = np.zeros(len(pcf.x) - 1)

        for i in range(len(lpcf.x) - 1):
            k = np.where(lplf.x == lpcf.x[i])[0][0]
            l = np.where(lplf.x == lpcf.x[i + 1])[0][0]
            result[i] = np.trapz(lplf.y[k : l + 1], lplf.x[k : l + 1])

        return result

    def convolute(self, pcf):
        """
        Return the convolution (integral) of the PLF with a PCF

        Args:
            pcf: (:obj: `PCF`): the piecewise constant function

        Returns:
            float: the convolution integral
        """

        if not isinstance(pcf, PCF):
            raise ValueError("I can only convolute a PLF with a PCF")

        # Insert PCF nodes into a local copy of plf
        lplf = self.copy()
        lplf.insert_nodes(pcf.x)

        # Insert left and right end-point of plf in local copy of PCF
        lpcf = pcf.copy()
        lpcf.insert_nodes([self._x[0], self._x[-1]])
        # Set lpcf nodes left and right to zero
        lpcf.y = np.where(
            np.logical_or(lpcf.x < self._x[0], lpcf.x >= self._x[-1]), 0.0, lpcf.y
        )

        result = 0.0
        for i in range(len(lpcf.x) - 1):
            k = np.where(lplf.x == lpcf.x[i])[0][0]
            l = np.where(lplf.x == lpcf.x[i + 1])[0][0]
            result += lpcf.y[i] * np.trapz(lplf.y[k : l + 1], lplf.x[k : l + 1])

        return result


class PCF(PiecewiseFunction):
    """Class that implements Piecewise Constant Functions. Derived from
    PiecewiseFunction.
    """

    def __init__(self, x, y, normalize=False, normvalue=1.0):
        super().__init__(x, y, normalize, normvalue)

        # make sure the last y value is zero
        self._y[-1] = 0.0

    def __call__(self, x):

        idx = np.searchsorted(self._x, x, side="right") - 1

        result = np.where(
            np.logical_and(idx >= 0, idx < len(self._x)), self._y[idx], 0.0
        )

        return result

    @PiecewiseFunction.slopes.getter
    def slopes(self):
        return np.ones_like(self._x)[:-1]

    def norm(self, left=0, right=None):
        if not right:
            right = len(self._x)

        return np.dot(self._y[left : right - 1], np.diff(self._x[left:right]))

    def copy(self):
        """
        Return a full, deep copy of PCF.

        Args:
            none

        Returns:
            (:obj: `PCF`): a deep copy of itself
        """
        return PCF(self._x, self._y, normalize=True, normvalue=self.norm())
