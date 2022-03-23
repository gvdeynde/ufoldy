# -*- coding: utf-8 -*-
"""
This module provides a class for unfolding neutron spectra
"""

from abc import ABC, abstractmethod
import numpy as np
from ufoldy.piecewiselinear import PieceWiseLinear as PLF
from ufoldy.Piecewiselinear import refine_log
from ufoldy.reactionrate import ReactionRate


class UFO(ABC):
    """Class that handles the generic approach to spectrum unfolding.
    Subclasses have to implement the minimization of a functional step.
    """

    def __init__(self, reaction_rates, initial_guess, refiner, verbosity):
        """
        Args:
            reaction_rates (:obj: `list` of :obj:`ReactionRate`): the reaction
            rates to use in the unfolding procedure
            initial_guess (:obj: `PiecewiseLinearFunction`): an initial guess
            for the neutron spectrum.
            refiner (:fun:) refiner function
            verbose (int): level of verbosity
                            0: silent
                            1: only high-level
                            2: puts the optimizer algorithm also to verbose
        """

        self._reaction_rates = reaction_rates
        self._initial_guess = initial_guess
        self._verbosity = verbosity
        self._refiner = refiner

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

    def unfold(self, Nrefine):
        """
        This method performs the whole unfolding, using `Nrefine` refinement
        steps using the refiner method. It "clears" the whole solution list at
        the start!

         Args:
             Nrefine (int): number of refinement steps

         Returns:
             (list): self._solutions. A list of `PiecewiseLinearFunction`
             objects
        """

        # Clean the solution list
        self._solutions = []

        # Start with initial guess and optimize
        if self._verbosity > 0:
            print("Start with optimizing initial guess.")

        self.unfold_initial()

        iref = 0
        while iref < Nrefine:
            iref += 1

            if self._verbosity > 0:
                print("\n")
                print(f" Refinement step #{iref}")

            self.unfold_singlestep()

        return self._solutions

    def unfold_initial(self):
        """
        This method performs the intial step in the unfolding: starting
        from the user's initial guess, it creates the first optimized (but
        non-refined) estimate.
        """
        result = self.optimize(self._initial_guess, True)

        self._solutions.append(result)

        return result

    def unfold_singlestep(self):
        """This method performs one refinement in the unfolding.
        Args:

        Returns:
        (:obj: `PiecewiseLinearFunction`): the next refined optimized
        solution
        """

        if len(self._solutions) == 0:
            # No solutions present. I should start with the inital guess from
            # the user
            self.unfold_initial()

        else:
            # Get a copy of the previous estimate
            guess = self._solutions[-1].copy()

            # Refine it according to the refiner strategy
            guess.refine(self._refiner)

            # Optimize
            result = self.optimize(guess, False)

            # Append result
            self._solutions.append(result)

        return self._solutions[-1]

    @abstractmethod
    def optimize(self, guess):
        """This method should be specified in the child classes"""
        pass


class Tikhonov(UFO):
    """Class that implements Tikhonov regularization"""

    def __init__(
        self,
        reaction_rates,
        initial_guess,
        weights=None,
        refiner=refiner_log,
        verbosity=0,
    ):
        """
        Args:
            reaction_rates (:obj: `list` of :obj:`ReactionRate`): the reaction
            rates to use in the unfolding procedure
            initial_guess (:obj: `PiecewiseLinearFunction`): an initial guess
            for the neutron spectrum.
            weigts (:obj: `numpy.ndarray` or list) of length 4: weights for the Tikhonov
            functional. First weight is for the residual, second weight is for
            the deviation from the prior, third weight is for the derivative,
            fourth weight is for the curvature.
            refiner (:fun:) refiner function
            verbose (int): level of verbosity
                            0: silent
                            1: only high-level
                            2: puts the optimizer algorithm also to verbose
        """
        super().__init__(reaction_rates, initial_guess, refiner, verbosity)

        if not weights:
            w = np.ones(4)
        else:
            w = np.asarray(weights)
            if w.size != 4:
                raise ValueError("The Tikhonov weights vector should have length 4!")

        self._weigts = w

        def optimize(self, guess, initial):

            self._guess = guess.copy()
            self._initial = initial

            x0 = np.log10(guess.y)

            # This could also be made more flexible
            method = "Nelder-Mead"
            options = {"adaptive": True, "xatol": 1e-4, "fatol": 1e-6}
            bounds = [(0, None) for i in x0.size]

            optim_res = minimize(
                self.tikhonov_functional,
                x0=x0,
                method=method,
                options=options,
                bounds=bounds,
            )

            guess.y = np.power(10, optim_res.x)

            self._guess = None
            self._initial = None

            return guess

        def tikhonov_functional(self, y):
            """This function calculates the Tikhonov functional"""

            # Update the guess with the new `y` value
            self._guess.y = np.power(10, y)

            # Calculate reaction rate residual
            rr_res = np.zeros(len(self._reaction_rates))

            for i, rr in enumerate(self._reaction_rates):
                rr_res[i] = (
                    rr.cross_section.convolute(self._guess) - rr.reaction_rate
                ) ** 2 / rr.reaction_rate_error

            rr_residual = np.sqrt(np.sum(rr_res))

            # Calculate difference to prior
            # If self._initial is True, we need to compare to initial guess from
            # user. If self._initial is False, we need to compare to the last
            # solution in the solution list.

            if self._initial:
                difference_prior = np.linalg.norm(y - np.log10(self._initial_guess.y))
            else:
                difference_prior = np.linalg.norm(y - np.log10(self._solutions[-1].y))

            # Calculate smoothness
            smoothness = np.linalg.norm(np.diff(y))

            # Calculate curvature
            curvature = np.lingalg.norm(np.diff(y, 2))

            # Total residual
            total_residual = np.dot(
                np.array([rr_residual, difference_prior, smoothness, curvature]),
                self._weigts,
            )

            if self._verbose > 1:
                print(self._guess)
                print(rr_residual, difference_prior, smoothness, curvature)
                print(self._weigts[0] * rr_residual)
                print(self._weigts[1] * difference_prior)
                print(self._weigts[2] * smoothness)
                print(self._weigts[3] * curvature)
                print(total_residual)

            return total_residual


