import logging

import pyfizi
import numpy as np
import pandas as pd
import scipy.linalg as lin
import scipy.stats as stats

from numpy.linalg import multi_dot as mdot


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

    GWAS = pyfizi.GWAS
    RefPanel = pyfizi.RefPanel

    nall = len(obs_snps)
    nimp = len(imp_snps) if imp_snps is not None else 0

    # this needs to be cleaned up. at some point just switch to 'standard' columns
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
            neff = np.max(obs_snps[GWAS.NCOL]) * r2blup
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

    if GWAS.NCOL in obs_snps or gwas_n is not None:
        df = df[[GWAS.CHRCOL, GWAS.SNPCOL, GWAS.BPCOL, GWAS.A1COL, GWAS.A2COL, GWAS.TYPECOL, GWAS.ZCOL, GWAS.R2COL,
                GWAS.NEFFCOL, GWAS.PCOL]]
    else:
        df = df[[GWAS.CHRCOL, GWAS.SNPCOL, GWAS.BPCOL, GWAS.A1COL, GWAS.A2COL, GWAS.TYPECOL, GWAS.ZCOL, GWAS.R2COL,
                GWAS.PCOL]]

    # order the data by position
    df[GWAS.BPCOL] = df[GWAS.BPCOL].astype(int)
    df = df.sort_values(by=[GWAS.BPCOL])

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
    run_fizi = annot is not None and taus is not None

    # cut down on typing
    GWAS = pyfizi.GWAS
    RefPanel = pyfizi.RefPanel
    Annot = pyfizi.Annot
    Taus = pyfizi.Taus

    # merge gwas with local-reference panel
    merged_snps = ref.overlap_gwas(gwas)

    # TODO: filter on large effect sizes, MAF, etc?

    # compute linkage-disequilibrium estimate
    log.debug("Estimating LD for {} SNPs".format(len(merged_snps)))
    LD = ref.estimate_ld(merged_snps, adjust=ridge)

    obs_flag = ~pd.isna(merged_snps.Z)
    to_impute = (~obs_flag).values
    obs = obs_flag.values

    nobs = np.sum(obs)
    nimp = np.sum(to_impute)
    if nimp == 0:
        log.info("Skipping region {}. No SNPs require imputation".format(ref))
        return pyfizi.create_output(gwas, start=start, stop=stop)

    mprop = nobs / float(nobs + nimp)
    log.debug("Proportion of observed-SNPs / total-SNPs = {}".format(mprop))
    if mprop < prop:
        log.info("Skipping region {}. Too few SNPs for imputation {:.3}%".format(ref, mprop))
        return pyfizi.create_output(gwas, start=start, stop=stop)

    imp_snps = merged_snps[to_impute]

    # flip zscores at SNPs with diff ref allele between GWAS and RefPanel
    sset = merged_snps[obs_flag]
    obsZ = sset.Z.values
    obsZ = pyfizi.flip_alleles(obsZ, sset[GWAS.A1COL], sset[GWAS.A2COL], sset[RefPanel.A1COL], sset[RefPanel.A2COL])

    log.debug("Partitioning LD into quadrants")
    Voo_ld = LD[obs].T[obs].T
    Vuo_ld = LD[to_impute].T[obs].T
    Vou_ld = Vuo_ld.T
    Vuu_ld = LD[to_impute].T[to_impute].T

    if run_fizi:
        # merge annotations with merged reference snps
        merged_snps = pd.merge(merged_snps, annot, how="left", left_on=RefPanel.SNPCOL, right_on=Annot.SNPCOL)

        # this assumes all taus columns are in the annotations
        # we should perform a sanity check when the program launches...
        sigma_values = taus[Taus.TAUCOL]
        annot_names = taus[Taus.NAMECOL].values.flatten()
        A = merged_snps[annot_names].values

        # this assumes intercept is first...
        # fine for now but we should fix this...
        A.T[0] = 1
        A[np.isnan(A)] = 0

        if gwas_n is None and GWAS.NCOL in gwas:
            gwas_n = np.median(gwas[GWAS.NCOL])

        D = np.diag(gwas_n * np.dot(A, sigma_values))
        Do = D.T[obs].T[obs]
        Du = D.T[to_impute].T[to_impute]
        uoV = Vuo_ld + mdot([Vuu_ld, Du, Vuo_ld]) + mdot([Vuo_ld, Do, Voo_ld])
        ooV = Voo_ld + mdot([Voo_ld, Do, Voo_ld]) + mdot([Vou_ld, Du, Vuo_ld])
        uuV = Vuu_ld + mdot([Vuu_ld, Du, Vuu_ld]) + mdot([Vuo_ld, Do, Vou_ld])
    else:
        uoV = Vuo_ld
        ooV = Voo_ld
        uuV = Vuu_ld

    log.debug("Computing inverse of variance-covariance matrix for {} observed SNPs".format(sum(obs)))
    try:
        ooVinv = lin.inv(ooV, check_finite=False)
    except lin.LinAlgError:
        log.debug("Inverse failed. Falling back to psuedo-inverse")
        ooVinv = lin.pinvh(ooV, check_finite=False)

    log.debug("Imputing {} SNPs from {} observed scores".format(sum(to_impute), sum(obs)))
    # predict the Z-scores
    impZs = mdot([uoV, ooVinv, obsZ])

    # compute r2-pred scores
    r2blup = np.diag(mdot([uoV, ooVinv, uoV.T])) / np.diag(uuV)

    # compute two-sided z-test for p-value
    pvals = stats.chi2.sf(impZs ** 2, 1)

    df = pyfizi.create_output(gwas, imp_snps, gwas_n, impZs, r2blup, pvals, start, stop)
    log.info("Completed imputation at region {}".format(ref))

    return df
