import logging

import numpy as np
import numpy.linalg as lin
import pandas as pd
import scipy.stats as stats


__all__ = ['partition_data', 'impute_gwas']


def partition_data(gwas, ref, window_size, loc):
    pass


def impute_gwas(gwas, obsRef, unobsRef, sigmas=None, epsilon=1e-6):
    log = logging.getLogger("fimpg")
    log.info("Imputing region {}".format(region))

    results = dict()

    # Get observed Z-scores
    obsZ = gwas.Zs

    # Get covariance matrices
    if sigmas is not None:
        # we have annotations. use random-effects model (ie FIMPg)
        pass
    else:
        # no annotations. use fixed-effect model (ie IMPg)
        obsV = obsRef.V
        unobsV = unobsRef.V

    obsVinv = lin.pinv(obsV)

    # predict the Z-scores
    impZs = lin.multi_dot([unobsV.T, obsVinv, obsZ])
    # compute two-sided z-test for p-value
    pvals = stats.chi2.isf(impZs ** 2, 1)

    # compute r2-pred
    r2pred = np.diag(lin.multi_dot([unobsV.T, obsVinv, unobsV]))

    # compute r2-pred adjusted for effective number of markers used for inference
    n_ref = obsRef.sample_size
    p_eff = obsRef.effective_snp_size
    r2pred_adj = max(1 - (1 - r2pred) * (n_ref - 1) / (n_ref - p_eff - 1), epsilon)

    neff = np.max(gwas.Ns) * r2pred
    nimp = len(unobsRef)
    nobs = len(gwas)

    results["CHR"] = gwas.CHRs[0] * (nimp + nobs)
    results["SNP"] = gwas.SNPs + unobsRef.SNPs
    results["BP"] = gwas.BPs + unobsRef.BPs
    results["A1"] = gwas.A1s + unobsRef.A1s
    results["A2"] = gwas.A2s + unobsRef.A2s
    results["Type"] = (["GWAS"] * nobs) + (["imputed"] * nimp)
    results["Z"] = gwas.Zs + impZs
    results["R2pred"] = ([1.0] * nobs) + r2pred
    results["Neff"] = gwas.Ns + neff
    results["P"] = gwas.Ps + pvals

    df = pd.DataFrame(data=results)
    df.sort_values(by=["BP"])

    return df
