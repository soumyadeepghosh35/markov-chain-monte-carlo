#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:31:25 2018

@author: Tobias Schwedes

Script to implement the sampling from a standard Normal Gaussian target
using the multiple proposal algorithm introduced in Yang (2018).
Proposals are generated according to a random walk Gaussian kernel.
"""

import numpy as np
import matplotlib.pyplot as plt


class Sampling_Yang:
    
    def __init__(self, d, alpha, x0, L, M, N, StepSize, Cov):
    
        """
        Implements sampling from a standard Normal Gaussian target
	    using the multiple proposal algorithm introduced in Yang (2018).
    
        Inputs:
        -------   
        d               - int 
                        dimension of posterior    
        alpha           - float
                        Standard deviation for Observation noise
        x0              - array_like
                        d-dimensional array; starting value
        L               - int
                			number of iterations
        M               - int
                        number of subsamples per iteration
        N               - int 
                        number of proposals per iteration                                          
        StepSize        - float 
                        step size for proposed jump in mean
        Cov         	   - array_like
                        dxd-dimensional proposal covariance             
        """
    
         
        ##################
        # Initialisation #
        ##################
    
        # List of samples to be collected
        self.xVals = list()
        self.xVals.append(x0)
       
        # Set up acceptance rate array
        self.AcceptVals = list()
    
        # Cholesky decomposition of Proposal Covariance
        CholCov = np.linalg.cholesky(Cov)
        InvCov = np.linalg.inv(Cov)
        
        # Target covariance
        TarCov = np.identity(d)
        
        
        ####################
        # Start Simulation #
        ####################
    
        for n in range(L):
            
            ######################
            # Generate proposals #
            ######################
              
            
            # Sample new proposed States according to multivariate t-distribution               
            y = x0 + np.dot(np.random.normal(0,StepSize,(N,d)), CholCov)
            
            # Add current state x0 to proposals    
            Proposals = np.insert(y, 0, x0, axis=0)
    
    
            ########################################################
            # Compute probability ratios = weights of IS-estimator #
            ########################################################
    
            # Compute Log-posterior probabilities
            LogTargets = -0.5*np.dot(np.dot(Proposals, TarCov), Proposals.T).diagonal(0)
    
            # Compute Log of transition probabilities
            LogK_ni = -0.5*np.dot(np.dot(Proposals-x0, InvCov/(StepSize**2)), \
                                 (Proposals - x0).T).diagonal(0)
            LogKs = np.sum(LogK_ni) - LogK_ni # from any state to all others
            

            # Compute weights
            LogPstates          = LogTargets + LogKs
            Pstates             = 1/N * np.minimum(1., np.exp(LogPstates - LogPstates[0]))
            Pstates[0]          = 1 - np.sum(np.delete(Pstates,0))
    
   
            ###############################
            # Sample according to weights #
            ###############################
    
            # Sample M new states 
            PstatesSum = np.cumsum(Pstates)
            Is = np.searchsorted(PstatesSum, np.random.uniform(0,1,M))
            xVals_new = Proposals[Is]
            self.xVals.append(xVals_new)
    
            # Compute acceptance rate
            AcceptValsNew = (np.count_nonzero(xVals_new[:-1]-xVals_new[1:]) + np.count_nonzero(xVals_new[0]-x0))/M
            self.AcceptVals.append(AcceptValsNew)
   
            # Update current state
            I = np.searchsorted(PstatesSum, np.random.uniform(0,1,1))
            x0  = Proposals[I]
#            x0 = xVals_new[-1].copy()


    def getSamples(self, BurnIn=0):
        
        """
        Compute samples from posterior from MP-QMCMC
        
        Inputs:
        ------
        BurnIn  - int 
                Burn-In period
        
        Outputs:
        -------
        Samples - array_like
                (Number of samples) x d-dimensional arrayof Samples      
        """
        
        Samples = np.concatenate(self.xVals[1:], axis=0)[BurnIn:,:]
                
        return Samples
       
        
    def getAcceptRate(self, BurnIn=0):
        
        """
        Compute acceptance rate of MP-QMCMC
        
        Inputs:
        ------
        BurnIn  - int
                Burn-In period
        
        Outputs:
        -------
        AcceptRate - float
                    average acceptance rate of MP-QMCMC 
        """    
        
        AcceptRate = np.mean(self.AcceptVals[BurnIn:])
        
        return AcceptRate

    
      
    def getMarginalHistogram(self, Index=0, BarNum=100, BurnIn=0):
        
        """
        Plot histogram of marginal distribution for posterior samples using 
        MP-QMCMC
        
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
        SubPlot.hist(self.getSamples(BurnIn)[:,Index], BarNum, label = "PDF Histogram", density = True)
        
        return Fig


