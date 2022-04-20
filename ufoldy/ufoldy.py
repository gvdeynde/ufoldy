# -*- coding: utf-8 -*-
"""
This module provides a class for unfolding neutron spectra
"""

import numpy as np
from scipy.optimize import lsq_linear
from ufoldy.piecewisefunction import PLF, PCF
from ufoldy.reactionrate import ReactionRate


def ufoldy(
    reactionrates,
    initial_guess=PCF.flat(1e-6, 2e7),
    max_prior_factor=10.0,
    gradient_weight=0.1,
    refinements=5,
    iters=5,
    lbound=1e-10,
    ubound=1.0,
):

    # Create list for flux estimates
    fluxes = [initial_guess]
    mu = np.sqrt(gradient_weight)
    NRR = len(reactionrates)

    # Start iterative procedure
    for i_refine in range(refinements):
        print(f'Refinement {i_refine:2d}')
        # Refine previous estimate
        flux_estimate = fluxes[-1].copy()
        flux_estimate.refine_log()

        for i_iter in range(iters):
            # Create matrix A
            A = np.zeros((NRR + flux_estimate.nnodes - 2, flux_estimate.nnodes - 1))

            # LS error reaction rates
            for i, rr in enumerate(reactionrates):
                for j in range(flux_estimate.nnodes - 1):
                    A[i, :] = rr.cross_section.partial_integrals(flux_estimate)

            # Contribution gradient
            for i in range(flux_estimate.nnodes - 2):
                A[NRR + i, i] = -mu
                A[NRR + i, i + 1] = +mu

            lbounds = np.maximum(lbound, flux_estimate.y[:-1] / max_prior_factor)
            ubounds = np.minimum(ubound, flux_estimate.y[:-1] * max_prior_factor)

            # Fill RHS
            b = np.zeros(NRR + flux_estimate.nnodes - 2)
            for i, rr in enumerate(reactionrates):
                b[i] = rr.reaction_rate

            solution = lsq_linear(
                A, b, bounds=(lbounds, ubounds), method="bvls", tol=1e-20
            )
            nfluxy = np.zeros_like(flux_estimate.x)
            nfluxy[:-1] = solution.x

            fluxes.append(PCF(flux_estimate.x, nfluxy))

    return fluxes
