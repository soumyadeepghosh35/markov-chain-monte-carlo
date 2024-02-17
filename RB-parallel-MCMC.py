#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:10:35 2019

@author: Tobias Schwedes

!/usr/bin/env python3

Script to implement Bayesian logistic regression using using Rao-Blackwellised
parallel MCMC.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from Data import DataLoad
from Seed import SeedGen


class RB_BayesianLogReg:

    def __init__(
        self,
        N,
        StepSize,
        PowerOfTwo,
        x0,
        InitMean,
        InitCov,
        Case,
        alpha=100.0,
        Stream="iid",
    ):
        """
        Implements the Bayesian Logistic Regression based on the
        Data sets in ./Data/ by using Rao-Blackwellised
        parallel MCMC.

        Inputs:
        -------
        N               - int
                        number of proposals per iteration
        StepSize        - float
                        step size for proposed jump in mean
        PowerOfTwo      - int
                        Defines size S of seed by S=2**PowerOfTwo-1
        x0              - array_like
                        d-dimensional array; starting value
        InitMean        - array_like
                        d-dimensional initial proposal mean
        InitCov         - array_like
                        dxd-dimensional initial proposal covariance
        Case            - string
                        determines the data used
        alpha           - float
                        1./alpha scales prior covariance
        Stream          - string
                        either 'cud' or 'iid'; defining what seed is used
        """

        #############
        # Load Data #
        #############

        Data = DataLoad(Case)
        d = Data.GetDimension()
        XX = Data.GetDesignMatrix()
        t = Data.GetResponses()

        ##################################
        # Choose stream for Markoc Chain #
        ##################################

        xs = SeedGen(d + 1, PowerOfTwo, Stream)

        ##################
        # Initialisation #
        ##################

        # List of samples to be collected
        self.xVals = list()
        self.xVals.append(x0)

        # Iteration number
        NumOfIter = int(int((2**PowerOfTwo - 1) / (d + 1)) * (d + 1) / N)
        print("Total number of Iterations = ", NumOfIter)

        # Set up acceptance rate array
        self.AcceptVals = list()

        # Initialise
        xI = self.xVals[0]
        I = 0

        # Weighted Sum and Covariance Arrays
        self.WeightedSum = np.zeros((NumOfIter, d))
        self.WeightedCov = np.zeros((NumOfIter, d, d))

        # Approximate Posterior Mean and Covariance as initial estimates
        self.ApprPostMean = InitMean
        self.ApprPostCov = InitCov

        # Cholesky decomposition of initial Approximate Posterior Covariance
        CholApprPostCov = np.linalg.cholesky(self.ApprPostCov)
        InvApprPostCov = np.linalg.inv(self.ApprPostCov)

        ####################
        # Start Simulation #
        ####################

        for n in range(NumOfIter):

            ######################
            # Generate proposals #
            ######################

            # Load stream of points in [0,1]^(d+1)
            U = xs[n * N : (n + 1) * N, :]

            # Sample new proposed States according to multivariate Gaussian
            y = self.ApprPostMean + np.dot(
                norm.ppf(U[:, :d], loc=np.zeros(d), scale=StepSize), CholApprPostCov
            )

            # Add current state xI to proposals
            Proposals = np.insert(y, 0, xI, axis=0)

            ########################################################
            # Compute probability ratios = weights of RB-estimator #
            ########################################################

            # Compute Log-posterior probabilities
            LogPriors = -0.5 * np.dot(
                np.dot(Proposals, np.identity(d) / alpha), (Proposals).T
            ).diagonal(0)
            fs = np.dot(XX, Proposals.T)
            LogLikelihoods = np.dot(t, fs) - np.sum(np.log(1.0 + np.exp(fs)), axis=0)
            LogPosteriors = LogPriors + LogLikelihoods

            # Compute Log of transition probabilities
            LogK_ni = -0.5 * np.dot(
                np.dot(Proposals - self.ApprPostMean, InvApprPostCov / (StepSize**2)),
                (Proposals - self.ApprPostMean).T,
            ).diagonal(0)
            LogKs = np.sum(LogK_ni) - LogK_ni  # from any state to all others

            # Compute weights
            LogPstates = LogPosteriors + LogKs
            Sorted_LogPstates = np.sort(LogPstates)
            LogPstates = LogPstates - (
                Sorted_LogPstates[-1]
                + np.log(
                    1 + np.sum(np.exp(Sorted_LogPstates[:-1] - Sorted_LogPstates[-1]))
                )
            )
            Pstates = np.exp(LogPstates)

            ########################
            # Compute RB-estimates #
            ########################

            # Compute weighted sum as posterior mean estimate
            WeightedStates = np.tile(Pstates, (d, 1)) * Proposals.T
            self.WeightedSum[n, :] = np.sum(WeightedStates, axis=1).copy()

            ##################################
            # Sample according to RB-weights #
            ##################################

            # Sample N new states
            PstatesSum = np.cumsum(Pstates)
            Is = np.searchsorted(PstatesSum, U[:, d:].flatten())
            xValsNew = Proposals[Is]
            self.xVals.append(xValsNew.copy())

            # Compute approximate acceptance rate
            AcceptValsNew = 1.0 - Pstates[Is]
            self.AcceptVals.append(AcceptValsNew)

            # Update current state
            I = Is[-1]
            xI = Proposals[I, :]

    def GetSamples(self, BurnIn=0):
        """
        Compute samples from posterior

        Inputs:
        ------
        BurnIn  - int
                Burn-In period

        Outputs:
        -------
        Samples - array_like
                (Number of samples) x d-dimensional arrayof Samples
        """

        Samples = np.concatenate(self.xVals[1:], axis=0)[BurnIn:, :]

        return Samples

    def GetAcceptRate(self, BurnIn=0):
        """
        Compute acceptance rate

        Inputs:
        ------
        BurnIn  - int
                Burn-In period

        Outputs:
        -------
        AcceptRate - float
                    average acceptance rate
        """

        AcceptVals = np.concatenate(self.AcceptVals)[BurnIn:]
        AcceptRate = np.mean(AcceptVals)

        return AcceptRate

    def GetIS_MeanEstimate(self, N, BurnIn=0):
        """
        Compute RB estimate

        Outputs:
        -------
        WeightedMean    - array_like
                        d-dimensional array
        """

        WeightedMean = np.mean(self.WeightedSum[int(BurnIn / N) :, :], axis=0)

        return WeightedMean

    def GetIS_FunMeanEstimate(self, N, BurnIn=0):
        """
        Compute RB estimate

        Outputs:
        -------
        WeightedMean    - array_like
                        d-dimensional array
        """

        WeightedMean = np.mean(self.WeightedFunSum[int(BurnIn / N) :, :], axis=0)

        return WeightedMean

    def GetIS_CovEstimate(self, N, BurnIn=0):
        """
        Compute RB covariance estimate


        Outputs:
        -------
        WeightedCov - d-dimensional array
        """

        WeightedCov = np.mean(self.WeightedCov[int(BurnIn / N) :, :, :], axis=0)

        return WeightedCov

    def GetMarginalHistogram(self, Index=0, BarNum=100, BurnIn=0):
        """
        Plot histogram of marginal distribution for posterior

        Inputs:
        ------
        Index   - int
                index of dimension for marginal distribution
        BurnIn  - int
                Burn-In period

        Outputs:
        -------
        Plot
        """

        Fig = plt.figure()
        SubPlot = Fig.add_subplot(111)
        SubPlot.hist(
            self.GetSamples(BurnIn)[:, Index],
            BarNum,
            label="PDF Histogram",
            density=True,
        )

        return Fig
