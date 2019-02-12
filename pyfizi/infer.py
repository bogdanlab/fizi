import numpy as np

from numpy.linalg import multi_dot as mdot, norm
from scipy.optimize import minimize
from scipy.linalg import pinv, pinvh, lstsq, cholesky
from scipy.stats import multivariate_normal as mvn


__all__ = ["infer_taus"]


def _infer_taus_hereg(zscores, LD, A):
    """
    Infer the variance parameters (tau) using HE-Regression over the summary statistics

    :param zscores:  numpy.ndarray of zscores
    :param LD: numpy.ndarray LD matrix representing correlation structure among `zscores`
    :param A: numpy.ndarray functional category annotation matrix

    :return: (numpy.ndarray taus, float sigma2e) tuple of the estimate taus and the residual variance
    """

    k, t = A.shape

    # the errors aren't independent due to LD; we need to weight the regression by LD estimates
    def get_weights(LD):
        # LD estimate is likely to be rank deficient so bump up eigenvalues to enable cholesky
        # Maybe look into optimal selection of offset (see Nocedal, Wright, Numerical Optimization)
        L = cholesky(LD + np.eye(k) * 0.1, lower=True)
        W = pinv(L)
        return W[np.tril_indices(k)]

    # each column in our multiple regression is the lower triangular values of the Vdiag(A_i)V matrix where V = LD
    def get_component(LD, A, jdx):
        return np.dot(LD * A.T[jdx], LD)[np.tril_indices(k)]

    # build regression matrix
    X = np.stack((get_component(LD, A, jdx) for jdx in range(t)), axis=1)

    # compute LD weights for weighted least squares
    W = get_weights(LD)

    # weighted matrix
    Xw = (X.T * W).T

    # compute the weighted sum of squared diff between zscores
    tmp = zscores.values * np.ones((k, k))
    yw = ((tmp - tmp.T) ** 2)[np.tril_indices(k)] * W

    # compute weighted least squares estimates of taus
    taus, sumsqerr, rank, svals = lstsq(Xw, yw)

    # estimate of the residual variance
    # TODO: if this is greater than one its likely we're at a GWAS region and indicates that the model is a poor fit
    # need to come up with heuristics...
    sigma2e = sumsqerr / len(yw)

    return taus, sigma2e


def _infer_taus_reml(zscores, LD, A):
    """
    Infer the variance parameters (tau) using maximum likelihood over the summary statistics

    :param zscores:  numpy.ndarray of zscores
    :param LD: numpy.ndarray LD matrix representing correlation structure among `zscores`
    :param A: numpy.ndarray functional category annotation matrix

    :return: (numpy.ndarray taus, float sigma2e) tuple of the estimate taus and the residual variance
    """

    k, t = A.shape
    r = t + 1

    def get_component(LD, A, jdx):
        return np.dot(LD * A.T[jdx], LD)

    As = [get_component(LD, A, jdx) for jdx in range(t)] + [LD]

    # start from the null of all variance explained by LD/finite-sample
    init = np.zeros(r)
    sigma2e = mdot([zscores, pinvh(LD), zscores]) / k
    init[-1] = sigma2e

    # TODO: replace V and P terms as globals that the closures have access to
    # it should speed things up by only needing to compute V and P once per iteration

    # negative log-likelihood (NLL) of the zscores given the variance parameters
    def obj(vars):
        V = sum(As[i] * vars[i] for i in range(r))
        logL = -mvn.logpdf(zscores, cov=V, allow_singular=True)
        print("NLL({}) = {}".format(",".join(map(str, vars)), logL))
        return logL

    # gradient of the NLL
    def grad(vars):
        g = np.zeros(r)
        V = sum(As[i] * vars[i] for i in range(r))
        P = pinvh(V)
        ztP = np.dot(zscores, P)
        Pz = ztP.T
        for i in range(r):
            Ai = As[i]
            g[i] = np.trace(P.dot(Ai)) - mdot([ztP, Ai, Pz])

        g = 0.5 * g
        print("||g|| = {}".format(norm(g)))
        return g

    # average-information matrix of the NLL; not really the hessian...
    def hess(vars):
        AI = np.zeros((r, r))
        V = sum(As[i] * vars[i] for i in range(r))
        P = pinvh(V)
        ztP = np.dot(zscores.T, P)
        Pz = ztP.T
        for i in range(r):
            ztPAsi = np.dot(ztP, As[i])
            for j in range(i + 1):
                AI[i, j] = mdot([ztPAsi, P, As[j], Pz])
                AI[j, i] = AI[i, j]

        AI = 0.5 * AI
        return AI

    try:
        # trust-ncg should be more robust compared with ncg
        res = minimize(obj, init, method="trust-ncg", jac=grad, hess=hess, options={"gtol": 1e-3})
        if res.success:
            result = (res.x[:-1], res.x[-1])
        else:
            result = None
    except Exception as exc:
        result = None

    return result


def infer_taus(zscores, LD, A):
    """
    Infer the variance parameters (tau) using summary GWAS statistics

    :param zscores:  numpy.ndarray of zscores
    :param LD: numpy.ndarray LD matrix representing correlation structure among `zscores`
    :param A: numpy.ndarray functional category annotation matrix

    :return: (numpy.ndarray taus, float sigma2e) tuple of the estimate taus and the residual variance
    """
    return _infer_taus_hereg(zscores, LD, A)


