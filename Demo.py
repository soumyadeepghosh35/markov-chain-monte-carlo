#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:11:15 2018

@author: Tobias Schwedes
"""

import time
import numpy as np
from Gauss_Calderhead import Sampling_Calderhead
from Gauss_Yang import Sampling_Yang
from Gauss_Yang_auxiliary import Sampling_Yang_auxiliary
import matplotlib.pyplot as plt
from scipy.stats import norm

if __name__ == '__main__':

    #############################
    # Parameters for simulation #
    #############################
    
    d           = 1                 # Dimension of posterior
    alpha       = 0.5               # Standard deviation of observation noise
    x0          = np.zeros(d)       # Starting value
    N           = 100              # Number of proposed states
    L           = 10000
    M           = 100
    StepSize    = 1.                 # Proposal step size
    Cov         = np.identity(d)
    
    # Define optional Burn-In
    BurnIn = 0
    
    # Target Parameters
    TarMean = np.zeros(d)
    TarCov = np.identity(d)
    
    ##################
    # Run simulation #
    ##################

    # Starting time of simulation
    StartTime = time.time()
    


    # Run simulation
#    Output = Sampling_Calderhead(d, alpha, x0, L, M, N, StepSize, Cov)
#    Output = Sampling_Yang(d, alpha, x0, L, M, N, StepSize, Cov)
    Output = Sampling_Yang_auxiliary(d, alpha, x0, L, M, N, StepSize, Cov)  
    
    # Stopping time
    EndTime = time.time()

    print ("CPU time needed =", EndTime - StartTime)



    ###################
    # Analyse results #
    ###################

    # Samples
    Samples = Output.getSamples(BurnIn)

    # Plot marginal PDF histogram in Index-th coordinate
    Index = 0
    BarNum = 100
    x = np.linspace(TarMean-4,TarMean+4,100)
    y = norm.pdf(x, TarMean, np.sqrt(TarCov[0]))
    Samples = Output.getSamples(0)
    Fig = plt.figure()
    SubPlot = Fig.add_subplot(111)
    SubPlot.hist(Samples[:,Index], BarNum, label = "Algo Output", density = True)
    SubPlot.plot(x,y, label = 'True PDF')
    SubPlot.legend()
    plt.savefig('./Yang_etal_M1.eps')


    # Compute average acceptance rate 
    AcceptRate = Output.getAcceptRate(BurnIn)
    print ("Acceptance rate = ", AcceptRate)
