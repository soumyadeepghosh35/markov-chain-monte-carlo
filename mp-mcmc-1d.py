import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, rv_discrete

plt.ion()


def normal_dens1d(x, mu=np.array([0.0]), sigma=1.0):
    """
    Function to compute the probability density according to a 2
    dimensional normal distribution with uncorrelated components.

    Inputs:
    x - value at which Gaussian density is to be evaluated
    mu - mean value
    sigma - std deviation

    Outputs:
    val - density value
    """

    # covariance matrix
    cov = sigma**2 * np.matrix(np.identity(1))

    val = multivariate_normal.pdf(x, mean=mu, cov=cov)

    return val


def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    ra = np.correlate(x, x, mode="full")[-n:]
    assert np.allclose(
        ra, np.array([(x[: n - k] * x[-(n - k) :]).sum() for k in range(n)])
    )
    res = ra / (variance * (np.arange(n, 0, -1)))

    return res


def Ess(xvals, autocor):

    N = xvals.shape[0]

    if N % 2 == 0:
        auto = autocor[2:][::2] + autocor[3:][::2]
    elif N % 2 == 1:
        auto = autocor[2:-1][::2] + autocor[3:][::2]

    if any(auto < 0):
        n_neg = min(np.where(auto < 0)[0])
    else:
        print(
            "Increase sample size to have sufficiently many autocorrelations for estimation"
        )
        n_neg = len(auto)

    diff = np.diff(auto)
    if any(diff > 0):
        n_mon = np.min(np.where(diff > 0))
    else:
        n_mon = n_neg

    n = min(n_neg, n_mon) + 1
    K = 2 * n + 2

    tau = 1 + 2 * np.sum(autocor[1:K])
    res = N * tau ** (-1)

    return res


def proposal_sample(x_old, N, sigma_proposal=1.0):
    """
    Function to draw proposal samples according to a 1-dimensional
    normal distribution with idendity covariance matrix.

    inputs:
    x_old - the last accepted MC sample
    N - number of proposals
    sigma_proposal - standard deviation of proposals
    outputs:
    x_val - the proposed sample
    """

    cov = sigma_proposal**2 * np.matrix(np.identity(1))
    x_vals = np.random.multivariate_normal(x_old, cov, N)

    return x_vals


def proposal_density(x, y, sigma_proposal=1.0):
    """
    Function to compute the proposal probability density according to
    a 1-dimensional normal distribution with uncorrelated components.

    Inputs:
    x - mean value of Gaussian
    y - array of values at which Gaussian density is to be evaluated
    sigma_proposal - std deviation
    Outputs:
    val - density value
    """

    # covariance matrix
    cov = sigma_proposal**2 * np.matrix(np.identity(1))

    val = multivariate_normal.pdf(y, mean=x, cov=cov)

    return val


def fun_Y(xvals):
    if xvals.ndim == 1:
        val = xvals[1]
    elif xvals.ndim == 2:
        val = xvals[:, 1]
    return val


def mp_mcmc(x0, sample_size, N, epsilon):
    """
    Function to implement the MP-MCMC algorithm.

    inputs:
    x0 - the initial condition for the Markov chain
    sample_size - number of samples
    N - number of proposals per iteration
    epsilon - proposal distribution standard deviation

    outputs:
    xvals - the array of sample values
    avals - the array of acceptance probabilities
    """

    # set up sample array and transition array
    xvals = list()
    xvals.append(x0)

    # set up acceptance rate array
    acpt = list()

    # initialise
    x_I = xvals[0]
    I = 0

    # number of iterations
    n_iter = int(np.ceil(sample_size / N))

    # Weighted Sum and Covariance Arrays
    WeightedSum = np.zeros((n_iter, 1))

    # set up quantities for weight computations
    Is = np.zeros(n_iter + 1)
    Is[0] = I
    x_Is = np.zeros((n_iter + 1, 1))
    x_Is[0, :] = x_I

    for n in range(n_iter):

        # define proposal distribution according to Langevin
        grad_log = np.array([0.0 - x_I[0]])
        mu_I = x_I + epsilon**2 / 2 * grad_log

        # draw sample from auxiliary distribution
        z = proposal_sample(mu_I, 1, epsilon)[0]
        mu_z = z + epsilon**2 / 2 * np.array([0.0 - z[0]])

        # draw proposal samples
        proposals = proposal_sample(mu_z, N, epsilon)

        # add state x_I to proposals
        states = np.insert(proposals, I, x_I, axis=0)

        # compute densitiy according to stationary distribution
        LogPosteriors = np.log(multivariate_normal.pdf(states, mean=0, cov=1))

        # computing the transition probabilities
        InvFisherInfo_xi = 1.0
        GradLog_states = 1 * (0.0 - states)  # np.dot(InvPostCov,(post_mean-states).T)
        mu_Proposals = states + epsilon**2 / 2.0 * InvFisherInfo_xi * GradLog_states
        LogKiz = np.log(
            multivariate_normal.pdf(
                mu_Proposals, mean=z, cov=epsilon**2 * InvFisherInfo_xi
            )
        )

        FisherInfo_z = 1.0
        LogKz_ni = -0.5 * np.dot(
            np.dot(states - mu_z, FisherInfo_z / (epsilon**2)), (states - mu_z).T
        ).diagonal(0)
        LogKs = np.sum(LogKz_ni) - LogKz_ni + LogKiz

        LogPstates = LogPosteriors + LogKs
        Sorted_LogPstates = np.sort(LogPstates)
        LogPstates = LogPstates - (
            Sorted_LogPstates[0]
            + np.log(1 + np.sum(np.exp(Sorted_LogPstates[1:] - Sorted_LogPstates[0])))
        )
        Pstates = np.exp(LogPstates)

        # do sampling
        Is = rv_discrete(values=(range(N + 1), Pstates)).rvs(size=N)
        xvals_new = states[Is]
        xvals.append(xvals_new)

        # compute acceptance rate
        acpt_new = 1.0 - Pstates[Is]
        acpt.append(acpt_new)

        ### Add weighting proposals here
        WeightedStates = np.tile(Pstates, (1, 1)) * states.T
        WeightedSum[n, :] = np.sum(WeightedStates, axis=1).copy()

        x_I = xvals_new[-1]

    ApprPostMean = np.mean(WeightedSum, axis=0)[0]

    return xvals, acpt, ApprPostMean, WeightedSum


if __name__ == "__main__":

    # system parameters
    sample_size = 10_000
    startVal = np.array([0.0])
    propNum = 10
    burnIn_val = 0
    epsilon = np.sqrt(1.8)

    # stop time
    start_time = time.time()

    # run MP-MCMC
    res = mp_mcmc(startVal, sample_size, propNum, epsilon)

    end_time = time.time()

    print("CPU time =", end_time - start_time)

    print("Number of proposals = ", propNum)

    # outcomes of MP-MCMC
    xvals = np.concatenate(res[0][1:], axis=0)[burnIn_val:, :]

    # compute autocorrelations and effective samples sizes
    autocorx = estimated_autocorrelation(xvals[:, 0])
    ess_x = Ess(xvals[:, 0], autocorx)

    print("Effective sampling size in x = ", ess_x)

    # compute estimated mean of stationary distribution (f=Id) and estimator variance
    estim = np.mean(xvals, axis=0)
    estimVar = np.var(xvals, axis=0) / ess_x

    print("estimated mean =", estim)
    print("estimator variance =", estimVar)
    print("estimated standard deviation =", np.sqrt(estimVar))

    # plot PDF histogram vs true density in x-direction
    plt.clf()
    plt.hist(xvals, 50, label="PDF Histogram in x-direction", density=1)
    x_0 = np.linspace(-3, 3, 500)
    y_0 = multivariate_normal.pdf(x_0, mean=0.0, cov=1.0)
    plt.plot(x_0, y_0)
    plt.legend(loc="best", fontsize=14)
    plt.show()

    print("RB mean estimate = ", res[2])

    # used average acceptance rate
    acpt = np.concatenate(res[1])[burnIn_val:]

    print("Acceptance rate = ", np.sum(acpt) / acpt.shape[0])
