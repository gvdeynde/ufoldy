# -*- coding: utf-8 -*-
"""
This module provides a class for unfolding neutron spectra
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize, shgo
from ufoldy.llplf import LLPLF
from ufoldy.reactionrate import ReactionRate


class UFO(ABC):
    """Class that handles the generic approach to spectrum unfolding.
    Subclasses have to implement the minimization of a functional step.
    """

    def __init__(self, reaction_rates, initial_guess, verbosity=0):
        """
        Args:
            reaction_rates (:obj:`list` of :obj:`ReactionRate`): the reaction
            rates to use in the unfolding procedure
            initial_guess (:obj: `PiecewiseLinearFunction`): an initial guess
            for the neutron spectrum.
            verbose (int): level of verbosity
                            0: silent
                            1: only high-level
                            2: puts the optimizer algorithm also to verbose
        """

        self._reaction_rates = reaction_rates
        self._initial_guess = initial_guess.copy()
        self._verbosity = verbosity

        self._solutions = []

    @property
    def reaction_rates(self):
        return self._reaction_rates

    @reaction_rates.setter
    def reaction_rates(self, rr):
        raise ValueError(
            "Sorry. You can't change the reaction rates on the fly. Please instantiate a new object."
        )

    @property
    def initial_guess(self):
        return self._initial_guess

    @initial_guess.setter
    def initial_guess(self, ig):
        raise ValueError(
            "Sorry. You can't change the initial guess on the fly. Please instantiate a new object."
        )

    @property
    def solutions(self):
        return self._solutions

    @solutions.setter
    def solutions(self, sol):
        raise ValueError(
            "Sorry. You can't change the solution list on the fly. The code fills this list."
        )

    @property
    def reaction_rate_errors(self, idx=-1):
        """Property to get the errors in the reaction rates for a certain
        solution. If no idx is given, function returns the errors for the last
        solution in the list.
        """

        Nrr = len(self._reaction_rates)

        rr_errors = np.zeros(Nrr)

        for i in range(Nrr):
            rr_errors[i] = (
                self._reaction_rates[i].reaction_rate
                - (self._solutions[idx] * self._reaction_rates[i].cross_section).norm
            )

        return rr_errors

    @reaction_rate_errors.setter
    def reaction_rate_errors(self, val):
        raise ValueError("reaction_rate_errors is a read-only property")

    def unfold(self, Nrefine):
        """
        This method performs the whole unfolding, using `Nrefine` refinement
        steps using the logarithmic refiner method. It "clears" the whole solution
        list at the start!

         Args:
             Nrefine (int): number of refinement steps

         Returns:
             (list): self._solutions. A list of `LLPLF` objects
        """

        # Clean the solution list
        self._solutions = []

        # Start with initial guess and optimize
        if self._verbosity > 0:
            print("Start with optimizing initial guess.")

        self.unfold_initial()

        if self._verbosity > 1:
            print(self._solutions[-1])
            print("===========================")
            print("\n")

        iref = 0
        while iref < Nrefine:
            iref += 1

            if self._verbosity > 0:
                print("---------------------------")
                print(f"  - Refinement step #{iref:2d} -")
                print("---------------------------")

            self.unfold_singlestep()

            if self._verbosity > 1:
                print(self._solutions[-1])
                print("===========================")
                print("\n")

    def unfold_initial(self):
        """
        This method performs the intial step in the unfolding: starting
        from the user's initial guess, it creates the first optimized (but
        non-refined) estimate.
        """
        result = self.optimize(self._initial_guess, True)

        self._solutions.append(result)

    def unfold_singlestep(self):
        """This method performs one refinement in the unfolding."""

        if len(self._solutions) == 0:
            # No solutions present. I should start with the inital guess from
            # the user
            self.unfold_initial()

        else:
            # Get a copy of the previous estimate
            guess = self._solutions[-1].copy()

            # Refine it according to the refiner strategy
            guess.refine()

            # Optimize
            result = self.optimize(guess, False)

            # Append result
            self._solutions.append(result)

    @abstractmethod
    def optimize(self, guess, initial):
        """This method should be specified in the child classes"""
        pass


class Tikhonov(UFO):
    """Class that implements Tikhonov regularization"""

    def __init__(
        self,
        reaction_rates,
        initial_guess,
        weights=None,
        verbosity=0,
    ):
        """
        Args:
            reaction_rates (:obj: `list` of :obj:`ReactionRate`): the reaction
            rates to use in the unfolding procedure
            initial_guess (:obj: `PiecewiseLinearFunction`): an initial guess
            for the neutron spectrum.
            weights (:obj: `numpy.ndarray` or list) of length 4: weights for the Tikhonov
            functional. First weight is for the residual, second weight is for
            the deviation from the prior, third weight is for the derivative,
            fourth weight is for the curvature.
            verbose (int): level of verbosity
                            0: silent
                            1: only high-level
                            2: puts the optimizer algorithm also to verbose
        """
        super().__init__(reaction_rates, initial_guess, verbosity)

        if not weights:
            w = np.ones(4)
        else:
            w = np.asarray(weights)
            if w.size != 4:
                raise ValueError("The Tikhonov weights vector should have length 4!")

        self._weights = w

    def optimize(self, guess, initial):

        self._guess = guess.copy()
        self._initial = initial

        x0 = guess.z

        options = {}
        tol = 1e-8

        # Selection of minimizer could also be made more flexible
        # method = "Nelder-Mead"
        # options = {**options, **{"adaptive": True, "xatol": 1e-9, "fatol": 1e-7}}

        # method = "trust-constr"

        method = "L-BFGS-B"

        # method = "TNC"
        # options["ftol"] = 1e-8

        bounds = [(-12, -6) for xi in x0]

        optim_res = minimize(
            self.tikhonov_functional,
            x0=x0,
            method=method,
            options=options,
            bounds=bounds,
            tol=tol,
        )

        # options = {**options, "f_tol": 1e-8}
        # optim_res = shgo(self.tikhonov_functional, bounds, options=options)

        guess.z = optim_res.x

        self._guess = None
        self._initial = None

        return guess

    def tikhonov_functional(self, z):
        """This function calculates the Tikhonov functional"""

        # Update the guess with the new `z` value
        self._guess.z = z

        # Calculate reaction rate residual
        rr_res = np.zeros(len(self._reaction_rates))

        for i, rr in enumerate(self._reaction_rates):
            rr_res[i] = (
                (rr.cross_section * self._guess).norm - rr.reaction_rate
            ) ** 2 / rr.reaction_rate_error

        rr_residual = np.sqrt(np.sum(rr_res))

        # Calculate difference to prior
        # If self._initial is True, we need to compare to initial guess from
        # user. If self._initial is False, we need to compare to the last
        # solution in the solution list.
        # Comparison is done by evaluating the prior in the nodes and comparing
        # that value to the current `y` estimates.

        if self._initial:
            difference_prior = np.linalg.norm(
                z - np.log10(self._initial_guess(self._guess.x))
            )
        else:
            difference_prior = np.linalg.norm(
                z - np.log10(self._solutions[-1](self._guess.x))
            )

        # Calculate smoothness
        smoothness = np.linalg.norm(np.diff(z))

        # Calculate curvature
        curvature = np.linalg.norm(np.diff(z, 2))

        # Total residual
        total_residual = np.dot(
            np.array([rr_residual, difference_prior, smoothness, curvature]),
            self._weights,
        )

        if self._verbosity > 3:
            print(self._guess.y)
            print(rr_residual, difference_prior, smoothness, curvature, total_residual)
            print("--")

        return total_residual
