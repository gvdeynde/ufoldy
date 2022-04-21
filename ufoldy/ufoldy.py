# -*- coding: utf-8 -*-
"""
This module provides a class for unfolding neutron spectra
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import lsq_linear
from tqdm.auto import trange
from ufoldy.piecewisefunction import PLF, PCF
from ufoldy.reactionrate import ReactionRate


def ufoldy(
    reactionrates,
    initial_guess=PCF.flat(1e-6, 2e7),
    max_prior_factor=10.0,
    gradient_weight=0.1,
    refinements=5,
    max_iter=5,
    tol_iter=1e-2,
    lbound=1e-10,
    ubound=1.0,
    verbose=False,
):

    # Check verbosity level for progress bar
    if verbose:
        ranger = trange
    else:
        ranger = range

    # Create list for flux estimates
    fluxes = []
    mu = np.sqrt(gradient_weight)
    NRR = len(reactionrates)

    flux_estimate = initial_guess.copy()
    indices = []

    # Start iterative procedure
    for i_refine in ranger(refinements, leave=False):
        # print(f"Refinement {i_refine:2d}")
        # Refine previous estimate
        flux_estimate.refine_log()

        i_iter = 0
        converged = False

        while i_iter < max_iter and not converged:
            i_iter += 1
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
                A,
                b,
                bounds=(lbounds, ubounds),
                max_iter=200,
                method="bvls",
                # lsq_solver="lsmr",
                verbose=0,
            )

            nfluxy = np.zeros_like(flux_estimate.x)
            nfluxy[:-1] = solution.x

            fluxes.append(PCF(flux_estimate.x, nfluxy))
            flux_estimate = fluxes[-1].copy()

            if i_iter > 1:
                converged = (
                    np.linalg.norm(
                        np.abs(fluxes[-2].y[:-1] - fluxes[-1].y[:-1])
                        / fluxes[-1].y[:-1]
                    )
                    < tol_iter
                )

        indices.append(i_iter)

    return fluxes, indices
