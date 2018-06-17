import itertools as it
import logging

import fimpg
import numpy as np
import numpy.linalg as lin
import pandas as pd
import scipy.stats as stats

from scipy.linalg import svdvals


__all__ = ['effective_size', 'partition_data', 'impute_gwas']


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

def _iter_loc(loc):
    """
    Helper function to enumerate in same fashion as window-size approach
    """
    for idx, row in loc.iterrows():
        yield row.tolist()

    return

def effective_size(mat, eps=1e-6):
    svals = svdvals(mat)
    return np.sum(svals > eps)


def partition_data(gwas, ref, window_size=250e3, loc=None):
    # for each chromosome
    if loc is None:
        partitions = ref.partitions_by_chr(window_size)
    else:
        partitions = _iter_loc(loc)

    return partitions


def impute_gwas(gwas, ref, sigmas=None, prop=0.75, epsilon=1e-6):
    #log = logging.getLogger(fimpg.LOG)
    log = logging.getLogger()
    log.info("Imputing region {}".format(ref))

    merged_snps = ref.overlap_gwas(gwas)
    ref_snps = merged_snps.loc[~pd.isna(merged_snps.i)]

    # compute linkage-disequilibrium estimate
    LD = ref.estimate_LD(ref_snps)

    obsZ = merged_snps.loc[~pd.isna(merged_snps.Z)].Z.values
    to_impute = pd.isna(ref_snps.Z).values
    obs = (~pd.isna(ref_snps.Z)).values

    nobs = np.sum(obs)
    nimp = np.sum(to_impute)
    if nimp == 0:
        log.info("No missing SNPs at region {}".format(ref))
        return None

    # this needs tweaking... we need to check so there exist enough
    # snps for imputation per user spec
    mprop = (nobs - nimp) / float(nobs)
    if mprop < prop:
        log.info("Skipping region {}. Too few SNPs for imputation {}%".format(ref, mprop))
        return None

    if sigmas is not None:
        pass # TBD
    else:
        obsV = LD.T[obs].T[obs]
        unobsV = LD.T[to_impute].T[obs]

    obsVinv = lin.pinv(obsV)

    # predict the Z-scores
    impZs = lin.multi_dot([unobsV.T, obsVinv, obsZ])
    # compute two-sided z-test for p-value
    pvals = stats.chi2.sf(impZs ** 2, 1)

    # compute r2-pred
    r2pred = np.diag(lin.multi_dot([unobsV.T, obsVinv, unobsV]))

    # compute r2-pred adjusted for effective number of markers used for inference
    n_ref = ref.sample_size
    p_eff = fimpg.effective_size(obsV)
    def _r2adj(r2p):
        return max(1 - (1 - r2p) * ((n_ref - 1) / float(n_ref - p_eff - 1)), epsilon)

    r2adj = np.vectorize(_r2adj)
    r2pred_adj = r2adj(r2pred)

    obs_snps = merged_snps[obs]
    imp_snps = merged_snps[to_impute]

    nall = len(gwas)

    results = dict()
    results[fimpg.GWAS.CHRCOL] = [gwas[fimpg.GWAS.CHRCOL].iloc[0]] * (nimp + nall)
    results[fimpg.GWAS.SNPCOL] = gwas[fimpg.GWAS.SNPCOL].tolist() + imp_snps[fimpg.RefPanel.SNPCOL].tolist()
    results[fimpg.GWAS.BPCOL] = gwas[fimpg.GWAS.BPCOL].tolist() + imp_snps[fimpg.RefPanel.BPCOL].tolist()
    results[fimpg.GWAS.A1COL] = gwas[fimpg.GWAS.A1COL].tolist() + imp_snps[fimpg.RefPanel.A1COL].tolist()
    results[fimpg.GWAS.A2COL] = gwas[fimpg.GWAS.A2COL].tolist() + imp_snps[fimpg.RefPanel.A2COL].tolist()
    results[fimpg.GWAS.TYPECOL] = (["gwas"] * nall) + (["imputed"] * nimp)
    results[fimpg.GWAS.ZCOL] = gwas[fimpg.GWAS.ZCOL].tolist() + list(impZs)
    results[fimpg.GWAS.ADJR2COL] = ([1.0] * nall) + list(r2pred_adj)
    if fimpg.GWAS.NCOL in gwas:
        neff = np.max(gwas[fimpg.GWAS.NCOL]) * r2pred
        results[fimpg.GWAS.NEFFCOL] = gwas[fimpg.GWAS.NCOL].tolist() + list(neff)
    else:
        results[fimpg.GWAS.NEFFCOL] = ["NA"] * (nall + nimp)
    results[fimpg.GWAS.PCOL] = gwas[fimpg.GWAS.PCOL].tolist() + list(pvals)

    df = pd.DataFrame(data=results)
    df = df[[fimpg.GWAS.CHRCOL, fimpg.GWAS.SNPCOL, fimpg.GWAS.BPCOL, fimpg.GWAS.A1COL, fimpg.GWAS.A2COL,
             fimpg.GWAS.TYPECOL, fimpg.GWAS.ZCOL, fimpg.GWAS.ADJR2COL, fimpg.GWAS.NEFFCOL, fimpg.GWAS.PCOL]]
    df = df.sort_values(by=[fimpg.GWAS.BPCOL])
    log.info("Completed imputation at region {}".format(ref))

    return df
