import logging

import pyfizi
import numpy as np
import pandas as pd


__all__ = ['create_output', 'impute_gwas']


def create_output(obs_snps, imp_snps=None, gwas_n=None, impZs=None, r2blup=None, pvals=None, start=None, stop=None):
    """
    Create an output pandas.DataFrame containing the original GWAS results and imputed results (if provided)

    :param obs_snps: pyfizi.GWAS object containing the original GWAS data
    :param imp_snps: pyfizi.RefPanel object containing data at imputed SNPs (default None)
    :param gwas_n: int GWAS sample size (default None)
    :param impZs: numpy.ndarray containing imputed Z-scores (default None)
    :param r2blup: numpy.ndarray containing R2-blup values for imputed Z-scores (default None)
    :param pvals: numpy.ndarray containing p-values for imputed Z-scores (default None)
    :param start: int Starting base-pair position for imputed data (default None)
    :param stop: int Stopping base-pair position for imputed data (default None)

    :return: pandas.DataFrame of formatted, sorted observed and (optionally) imputed GWAS data
    """

    # aliases
    GWAS = pyfizi.GWAS
    RefPanel = pyfizi.RefPanel

    nall = len(obs_snps)
    nimp = len(imp_snps) if imp_snps is not None else 0

    results = dict()
    if imp_snps is not None:
        results[GWAS.CHRCOL] = [obs_snps[GWAS.CHRCOL].iloc[0]] * (nimp + nall)
        results[GWAS.SNPCOL] = obs_snps[GWAS.SNPCOL].tolist() + imp_snps[RefPanel.SNPCOL].tolist()
        results[GWAS.BPCOL] = obs_snps[GWAS.BPCOL].tolist() + imp_snps[RefPanel.BPCOL].tolist()
        results[GWAS.A1COL] = obs_snps[GWAS.A1COL].tolist() + imp_snps[RefPanel.A1COL].tolist()
        results[GWAS.A2COL] = obs_snps[GWAS.A2COL].tolist() + imp_snps[RefPanel.A2COL].tolist()
        results[GWAS.TYPECOL] = (["gwas"] * nall) + (["imputed"] * nimp)
        results[GWAS.ZCOL] = obs_snps[GWAS.ZCOL].tolist() + list(impZs)
        results[GWAS.R2COL] = ([1.0] * nall) + list(r2blup)
        if GWAS.NCOL in obs_snps:
            neff = np.median(obs_snps[GWAS.NCOL]) * r2blup
            results[GWAS.NEFFCOL] = obs_snps[GWAS.NCOL].tolist() + list(neff)
        elif gwas_n is not None:
            neff = gwas_n * r2blup
            results[GWAS.NEFFCOL] = ([gwas_n] * nall) + list(neff)
        results[GWAS.PCOL] = obs_snps[GWAS.PCOL].tolist() + list(pvals)
    else:
        results[GWAS.CHRCOL] = [obs_snps[GWAS.CHRCOL].iloc[0]] * nall
        results[GWAS.SNPCOL] = obs_snps[GWAS.SNPCOL].tolist()
        results[GWAS.BPCOL] = obs_snps[GWAS.BPCOL].tolist()
        results[GWAS.A1COL] = obs_snps[GWAS.A1COL].tolist()
        results[GWAS.A2COL] = obs_snps[GWAS.A2COL].tolist()
        results[GWAS.TYPECOL] = (["gwas"] * nall)
        results[GWAS.ZCOL] = obs_snps[GWAS.ZCOL].tolist()
        results[GWAS.R2COL] = ([1.0] * nall)
        if GWAS.NCOL in obs_snps:
            results[GWAS.NEFFCOL] = obs_snps[GWAS.NCOL].tolist()
        elif gwas_n is not None:
            results[GWAS.NEFFCOL] = [gwas_n] * nall

        results[GWAS.PCOL] = obs_snps[GWAS.PCOL].tolist()

    df = pd.DataFrame(data=results)

    # re-order columns to sensible format
    if GWAS.NCOL in obs_snps or gwas_n is not None:
        df = df[[GWAS.CHRCOL, GWAS.SNPCOL, GWAS.BPCOL, GWAS.A1COL, GWAS.A2COL, GWAS.TYPECOL, GWAS.ZCOL, GWAS.R2COL,
                GWAS.NEFFCOL, GWAS.PCOL]]
        df[GWAS.NEFFCOL] = df[GWAS.NEFFCOL].astype(int)
    else:
        df = df[[GWAS.CHRCOL, GWAS.SNPCOL, GWAS.BPCOL, GWAS.A1COL, GWAS.A2COL, GWAS.TYPECOL, GWAS.ZCOL, GWAS.R2COL,
                GWAS.PCOL]]

    # order the data by position
    df[GWAS.BPCOL] = df[GWAS.BPCOL].astype(int)
    df = df.sort_values(by=[GWAS.BPCOL])

    # make effect allele explicit
    df = df.rename(index=str, columns={GWAS.A1COL: "A_EFFECT", GWAS.A2COL: "A_ALT"})

    # imputation relies on data outside the window (buffer parameter)
    # prune down to actual imputation window here
    if start is not None and stop is not None:
        df = df.loc[(df[GWAS.BPCOL] >= start) & (df[GWAS.BPCOL] <= stop)]
    elif start is not None:
        df = df.loc[(df[GWAS.BPCOL] >= start)]
    elif stop is not None:
        df = df.loc[(df[GWAS.BPCOL] <= stop)]

    return df


def impute_gwas(gwas, ref, gwas_n=None, annot=None, taus=None, start=None, stop=None, prop=0.4, ridge=0.1):
    """
    Impute missing Z-scores using reference panel LD and optionally functional information.

    :param gwas: pyfizi.GWAS object for the region
    :param ref:  pyfizi.RefPanel object for reference genotype data at the region
    :param gwas_n: numpy.ndarray or int GWAS sample size. If int assumes sample size is uniform at each SNP.
                    Not required if 'N' is column in GWAS data (default: None)
    :param annot: pyfizi.Annot object representing the functional annotations at the region (default: None)
    :param taus: pyfizi.Tau object representing the prior variance terms for functional categories (default: None)
    :param start: int Starting base-pair position for GWAS data
    :param stop: int Stoping base-pair position for GWAS data
    :param prop: float Minimum proportion of GWAS SNPs to total data that must be present for imputation (default=0.4)
    :param ridge: float Ridge term to regularize LD estimation (default=0.1)

    :return: pandas.DataFrame containing observed and imputed GWAS results
    """
    log = logging.getLogger(pyfizi.LOG)
    log.info("Starting imputation at region {}".format(ref))

    # fizi or impg
    run_fizi = annot is not None

    # merge gwas with local-reference panel
    merged_snps = ref.overlap_gwas(gwas)

    # TODO: filter on large effect sizes, MAF, etc?
    # Some stats may break normality assumption and we can improve results by dropping/pruning them

    obs = merged_snps.are_observations()
    to_impute = merged_snps.are_imputations()

    imp_snps = merged_snps[to_impute]
    obs_snps = merged_snps[obs]
    nobs = len(obs_snps)
    nimp = len(imp_snps)

    if nimp == 0:
        log.info("Skipping region {}. No SNPs require imputation".format(ref))
        return pyfizi.create_output(gwas, start=start, stop=stop)

    mprop = nobs / float(nobs + nimp)
    log.debug("Proportion of observed-SNPs / total-SNPs = {:.3g}".format(mprop))
    if mprop < prop:
        log.warning("Skipping region {}. Too few SNPs for imputation {:.3g}%".format(ref, mprop))
        return pyfizi.create_output(gwas, start=start, stop=stop)

    # flip zscores at SNPs with diff ref allele between GWAS and RefPanel
    obsZ = pyfizi.flip_alleles(obs_snps.zscores,
                               obs_snps.gwas_a1_alleles, obs_snps.gwas_a2_alleles,
                               obs_snps.ref_a1_alleles, obs_snps.ref_a2_alleles)

    if run_fizi and gwas_n is None and gwas.has_n():
        gwas_n = np.median(gwas.ns)

    # TODO: other statistics to report? estimated taus? residual variance?
    # impute the missing zscores
    impZs, pvals, r2blup = _impute(merged_snps, ref, annot, taus, gwas_n, obs, to_impute, obsZ, ridge, run_fizi)

    df = pyfizi.create_output(gwas, imp_snps, gwas_n, impZs, r2blup, pvals, start, stop)
    log.info("Completed imputation at region {}".format(ref))

    return df


def _impute(merged_snps, ref, annot, taus, gwas_n, obs, to_impute, obsZ, ridge, run_fizi):
    """
    this is the internal logic for the imputation

    I refactored this into diff function to improve flexibility for any changes downstream
    (e.g., MI, sampling, sketching, etc)

    testing out multiple imputation (MI) for the functional part of fizi
    we could incorporate MI into the estimation of LD as well but it might come with a big computational hit
    one cool trick might be to use sketching to speed up LD estimation to maintain performance for MI

    :param merged_snps: pyfizi.MergedPanel object containing merged GWAS and LDRef data
    :param ref: pyfizi.RefPanel object for reference genotype data at the region
    :param annot: pyfizi.Annot object representing the functional annotations at the region (default: None)
    :param taus: pyfizi.Tau object representing the prior variance terms for functional categories (default: None)
    :param gwas_n: numpy.ndarray or int GWAS sample size. If int assumes sample size is uniform at each SNP.
                    Not required if 'N' is column in GWAS data (default: None)
    :param obsZ: numpy.ndarray vector of observed Z-scores that have been flipped to match ref panel
    :param obs: numpy.ndarray boolean vector marking which rows in `merged_snps` have observed Z-scores
    :param to_impute: numpy.ndarray boolean vector marking which rows in `merged_snps` need to be imputed
    :param ridge: float Ridge term to regularize LD estimation (default=0.1)
    :param run_fizi: bool indicating if fizi or impg is run

    :return: (numpy.ndarray imputed_z, numpy.ndarray pvalues, numpy.ndarray r2blups)
    """

    from numpy.linalg import multi_dot as mdot
    from scipy.linalg import pinvh
    from scipy.stats import chi2

    log = logging.getLogger(pyfizi.LOG)
    nobs = np.sum(obs)
    nimp = np.sum(to_impute)

    # compute linkage-disequilibrium estimate
    log.debug("Estimating LD for {} SNPs".format(len(merged_snps)))
    LD = ref.estimate_ld(merged_snps, adjust=ridge)

    log.debug("Partitioning LD into quadrants")
    Voo_ld = LD[obs].T[obs].T
    Vuo_ld = LD[to_impute].T[obs].T
    Vou_ld = Vuo_ld.T
    Vuu_ld = LD[to_impute].T[to_impute].T

    if run_fizi:
        if taus is not None:
            A = annot.get_matrix(merged_snps, taus.names)
            estimates = taus.estimates
            D = np.diag(gwas_n * np.dot(A, estimates)) #/ np.power(np.median(merged_snps.SE.values[obs]), 2)
            Do = D.T[obs].T[obs]
            Du = D.T[to_impute].T[to_impute]
            uoV = Vuo_ld + mdot([Vuu_ld, Du, Vuo_ld]) + mdot([Vuo_ld, Do, Voo_ld])
            ooV = Voo_ld + mdot([Voo_ld, Do, Voo_ld]) + mdot([Vou_ld, Du, Vuo_ld])
            uuV = Vuu_ld + mdot([Vuu_ld, Du, Vuu_ld]) + mdot([Vuo_ld, Do, Vou_ld])
        else:
            A = annot.get_matrix(merged_snps)
            names = annot.names
            Ao = A[obs]
            flag = np.mean(Ao != 0, axis=0) > 0
            Ao = Ao.T[flag].T
            A = A.T[flag].T
            names = names[flag]

            log.debug("Starting inference for variance parameters")
            estimates = pyfizi.infer_taus(obsZ, Voo_ld, Ao)
            if estimates is not None:
                log.debug("Finished variance parameter inference")

                estimates, sigma2e = estimates
                # rescale estimates
                estimates = estimates * np.sum(Ao != 0, axis=0) / np.sum(A != 0, axis=0)

                # N gets inferred as part of the parameter
                D = np.diag(np.dot(A, estimates))
                Do = D.T[obs].T[obs]
                Du = D.T[to_impute].T[to_impute]
                uoV = Vuo_ld + mdot([Vuu_ld, Du, Vuo_ld]) + mdot([Vuo_ld, Do, Voo_ld])
                ooV = Voo_ld + mdot([Voo_ld, Do, Voo_ld]) + mdot([Vou_ld, Du, Vuo_ld])
                uuV = Vuu_ld + mdot([Vuu_ld, Du, Vuu_ld]) + mdot([Vuo_ld, Do, Vou_ld])
            else:
                log.warning("Variance parameter optimization failed. Defaulting to ImpG")
                # estimation failed... default to ImpG
                uoV = Vuo_ld
                ooV = Voo_ld
                uuV = Vuu_ld
    else:
        uoV = Vuo_ld
        ooV = Voo_ld
        uuV = Vuu_ld

    """
    TODO: consider replacing with the following more numerically stable and efficient code
    this is low priority but might be useful to explore at some point

    # method 1; no extra overhead, but addtl solve cost
    ooL = cholesky(ooV, lower=True)
    uoVLinv = triangular_solve(ooL, uoV.T, lower=True)
    LinvZ = triangular_solve(ooL, obsZ, lower=True)

    impZs = uoVLinv.T @ LinvZ
    r2blup = np.sum(uoVLinv ** 2, axis=0) / np.diag(uuV)

    # method 2; extra memory, but cheaper solve cost
    ooL = cholesky(ooV, lower=True)
    tmp = triangular_solve(ooL, np.concatenate((obsZ[:,np.newaxis], uoV.T), axis=1), lower=True)
    uoVLinv = tmp.T[1:]
    LinvZ = tmp.T[0]

    impZs = uoVLinv.T @ LinvZ
    r2blup = np.sum(uoVLinv ** 2, axis=0) / np.diag(uuV)

    # method 3; use conjugate gradient with vec matrix ops over 'raw' LD and diagonal offset,
    # instead of adding offset to LD matrix and solving
    # or even just use raw genotype data vec matrix ops (if N_ref < SNP_obs)
    #    (X'X + Ilambda) @ candidate = X' @ (X @ candidate) + lambda * candidate
    uoVinv = cg(linear_op, uoV.T)
    impZs = uoVinv @ obsZ
    r2blup = np.diag(uoVinv @ uoV.T) / np.diag(uuV) # this can likely be further optimized with a product/sum op
    """
    log.debug("Computing inverse of variance-covariance matrix for {} observed SNPs".format(nobs))
    ooVinv = pinvh(ooV, check_finite=False)

    log.debug("Imputing {} SNPs from {} observed scores".format(nimp, nobs))
    impZs = mdot([uoV, ooVinv, obsZ])

    # compute r2-pred scores
    r2blup = np.diag(mdot([uoV, ooVinv, uoV.T])) / np.diag(uuV)

    # compute two-sided z-test for p-value
    pvals = chi2.sf(impZs ** 2, 1)

    return impZs, pvals, r2blup
