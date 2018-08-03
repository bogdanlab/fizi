import itertools as it
import logging

import fimpg
import numpy as np
import numpy.linalg as lin
import pandas as pd
import scipy.stats as stats

from scipy.linalg import svdvals


__all__ = ['create_output', 'impute_gwas', 'VALID_SNPS', 'MATCH_ALLELES']


# Base-handling code is from LDSC...
# complementary bases
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
# bases
BASES = COMPLEMENT.keys()
# true iff strand ambiguous
STRAND_AMBIGUOUS = {''.join(x): x[0] == COMPLEMENT[x[1]]
                    for x in it.product(BASES, BASES)
                    if x[0] != x[1]}
# SNPS we want to keep (pairs of alleles)
VALID_SNPS = {x for x in map(lambda y: ''.join(y), it.product(BASES, BASES))
              if x[0] != x[1] and not STRAND_AMBIGUOUS[x]}

# T iff SNP 1 has the same alleles as SNP 2 (allowing for strand or ref allele flip).
MATCH_ALLELES = {x for x in map(lambda y: ''.join(y), it.product(VALID_SNPS, VALID_SNPS))
                 # strand and ref match
                 if ((x[0] == x[2]) and (x[1] == x[3])) or
                 # ref match, strand flip
                 ((x[0] == COMPLEMENT[x[2]]) and (x[1] == COMPLEMENT[x[3]])) or
                 # ref flip, strand match
                 ((x[0] == x[3]) and (x[1] == x[2])) or
                 ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))}  # strand and ref flip
# T iff SNP 1 has the same alleles as SNP 2 w/ ref allele flip.
FLIP_ALLELES = {''.join(x):
                ((x[0] == x[3]) and (x[1] == x[2])) or  # strand match
                # strand flip
                ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))
                for x in MATCH_ALLELES}


def create_output(obs_snps, imp_snps=None, gwas_n=None, impZs=None, r2blup=None, pvals=None, start=None, stop=None):

    GWAS = fimpg.GWAS
    RefPanel = fimpg.RefPanel

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

    df[GWAS.BPCOL] = df[GWAS.BPCOL].astype(int)
    df = df.sort_values(by=[GWAS.BPCOL])

    if start is not None and stop is not None:
        df = df.loc[(df[GWAS.BPCOL] >= start) & (df[GWAS.BPCOL] <= stop)]
    elif start is not None:
        df = df.loc[(df[GWAS.BPCOL] >= start)]
    elif stop is not None:
        df = df.loc[(df[GWAS.BPCOL] <= stop)]

    return df


def impute_gwas(gwas, ref, gwas_n=None, annot=None, sigmas=None, start=None, stop=None, prop=0.4, ridge=0.1, epsilon=1e-6):
    log = logging.getLogger(fimpg.LOG)
    log.info("Starting imputation at region {}".format(ref))

    # cut down on typing
    GWAS = fimpg.GWAS
    RefPanel = fimpg.RefPanel
    Annot = fimpg.Annot
    Sigmas = fimpg.Sigmas

    # merge gwas with local-reference panel
    merged_snps = ref.overlap_gwas(gwas)
    ref_snps = merged_snps.loc[~pd.isna(merged_snps.i)]

    if annot is not None and sigmas is not None:
        # merge annotations with merged reference snps
        ref_snps = pd.merge(ref_snps, annot, how="left", left_on=RefPanel.SNPCOL, right_on=fimpg.Annot.SNPCOL)

        # this assumes all sigmas columns are in the annotations
        # we should perform a sanity check when the program launches...
        sigma_values = sigmas[Sigmas.SIGMACOL]
        annot_names = sigmas[Sigmas.NAMECOL].values.flatten()
        A = ref_snps[annot_names].values

        # this assumes intercept is first...
        # fine for now but we should fix this...
        A.T[0] = 1
        A[np.isnan(A)] = 0

        if gwas_n is None and GWAS.NCOL in gwas:
            gwas_n = np.median(gwas[GWAS.NCOL])

        D = np.diag(gwas_n * np.dot(A, sigma_values))

    # compute linkage-disequilibrium estimate
    LD = ref.estimate_LD(ref_snps, adjust=ridge)
    obs_flag = ~pd.isna(ref_snps.Z)
    to_impute = (~obs_flag).values
    obs = obs_flag.values

    nobs = np.sum(obs)
    nimp = np.sum(to_impute)
    if nimp == 0:
        log.info("Skipping region {}. No SNPs require imputation".format(ref))
        return fimpg.create_output(gwas, start=start, stop=stop)

    mprop = nobs / float(nobs + nimp)
    if mprop < prop:
        log.info("Skipping region {}. Too few SNPs for imputation {}%".format(ref, mprop))
        return fimpg.create_output(gwas, start=start, stop=stop)

    obs_snps = ref_snps[obs]
    imp_snps = ref_snps[to_impute]

    # check for allele flips
    sset = ref_snps[obs_flag]
    obsZ = sset.Z.values
    alleles = sset[GWAS.A1COL] + sset[GWAS.A2COL] + sset[RefPanel.A1COL] + sset[RefPanel.A2COL]

    # from LDSC...
    try:
        flip_flags = alleles.apply(lambda y: FLIP_ALLELES[y])
        obsZ *= (-1) ** flip_flags
    except KeyError as e:
        msg = 'Incompatible alleles in .sumstats files: %s. ' % e.args
        msg += 'Did you forget to use --merge-alleles with LDScore munge_sumstats.py?'
        raise KeyError(msg)

    Voo_ld = LD[obs].T[obs].T
    Vuo_ld = LD[to_impute].T[obs].T
    Vou_ld = Vuo_ld.T
    Vuu_ld = LD[to_impute].T[to_impute].T
    if sigmas is not None and annot is not None:
        Do = D.T[obs].T[obs]
        Du = D.T[to_impute].T[to_impute]
        uoV = Vuo_ld + lin.multi_dot([Vuu_ld, Du, Vuo_ld]) + lin.multi_dot([Vuo_ld, Do, Voo_ld])
        ooV = Voo_ld + lin.multi_dot([Voo_ld, Do, Voo_ld]) + lin.multi_dot([Vou_ld, Du, Vuo_ld])
        uuV = Vuu_ld + lin.multi_dot([Vuu_ld, Du, Vuu_ld]) + lin.multi_dot([Vuo_ld, Do, Vou_ld])
    else:
        uoV = Vuo_ld
        ooV = Voo_ld
        uuV = Vuu_ld

    ooVinv = lin.pinv(ooV)

    # predict the Z-scores
    impZs = lin.multi_dot([uoV, ooVinv, obsZ])

    # compute two-sided z-test for p-value
    pvals = stats.chi2.sf(impZs ** 2, 1)

    # compute marginal r2-pred scores
    # we re-scale by the marginal prior to account for inflation due to prior info
    # and adjustment to LD diagonal
    # this ensures that r2pred will always be between 0-1 (if D is PSD)
    r2blup = np.diag(lin.multi_dot([uoV, ooVinv, uoV.T])) / np.diag(uuV)

    df = fimpg.create_output(gwas, imp_snps, gwas_n, impZs, r2blup, pvals, start, stop)
    log.info("Completed imputation at region {}".format(ref))

    return df
