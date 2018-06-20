import itertools as it
import logging

import fimpg
import numpy as np
import numpy.linalg as lin
import pandas as pd
import scipy.stats as stats

from scipy.linalg import svdvals


__all__ = ['impute_gwas']


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


def impute_gwas(gwas, ref, sigmas=None, prop=0.75, epsilon=1e-6):
    log = logging.getLogger(fimpg.LOG)
    log.info("Starting imputation at region {}".format(ref))

    # cut down on typing
    GWAS = fimpg.GWAS
    RefPanel = fimpg.RefPanel

    # merge gwas with local-reference panel
    merged_snps = ref.overlap_gwas(gwas)
    ref_snps = merged_snps.loc[~pd.isna(merged_snps.i)]

    # compute linkage-disequilibrium estimate
    LD = ref.estimate_LD(ref_snps)
    obs_flag = ~pd.isna(merged_snps.Z)
    to_impute = (~obs_flag).values
    obs = obs_flag.values

    # check for allele flips
    sset = merged_snps[obs_flag]
    obsZ = sset.Z.values
    alleles = sset[GWAS.A1COL] + sset[GWAS.A2COL] + sset[RefPanel.A1COL] + sset[RefPanel.A2COL]

    # from LDSC...
    try:
        obsZ *= (-1) ** alleles.apply(lambda y: FLIP_ALLELES[y])
    except KeyError as e:
        msg = 'Incompatible alleles in .sumstats files: %s. ' % e.args
        msg += 'Did you forget to use --merge-alleles with LDScore munge_sumstats.py?'
        raise KeyError(msg)

    nobs = np.sum(obs)
    nimp = np.sum(to_impute)
    if nimp == 0:
        log.info("Skipping region {}. No SNPs require imputation".format(ref))
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

    # compute r2-pred adjusted for effective number of markers used in inference
    n_ref = ref.sample_size

    # compute effective size
    svals = svdvals(obsV)
    p_eff = np.sum(svals > epsilon)

    def _r2adj(r2p):
        return max(1 - (1 - r2p) * ((n_ref - 1) / float(n_ref - p_eff - 1)), epsilon)

    r2adj = np.vectorize(_r2adj)
    r2pred_adj = r2adj(r2pred)

    obs_snps = merged_snps[obs]
    imp_snps = merged_snps[to_impute]

    nall = len(gwas)

    # this needs to be cleaned up. at some point just switch to 'standard' columns
    results = dict()
    results[GWAS.CHRCOL] = [gwas[GWAS.CHRCOL].iloc[0]] * (nimp + nall)
    results[GWAS.SNPCOL] = gwas[GWAS.SNPCOL].tolist() + imp_snps[RefPanel.SNPCOL].tolist()
    results[GWAS.BPCOL] = gwas[GWAS.BPCOL].tolist() + imp_snps[RefPanel.BPCOL].tolist()
    results[GWAS.A1COL] = gwas[GWAS.A1COL].tolist() + imp_snps[RefPanel.A1COL].tolist()
    results[GWAS.A2COL] = gwas[GWAS.A2COL].tolist() + imp_snps[RefPanel.A2COL].tolist()
    results[GWAS.TYPECOL] = (["gwas"] * nall) + (["imputed"] * nimp)
    results[GWAS.ZCOL] = gwas[GWAS.ZCOL].tolist() + list(impZs)
    results[GWAS.ADJR2COL] = ([1.0] * nall) + list(r2pred_adj)
    if GWAS.NCOL in gwas:
        neff = np.max(gwas[GWAS.NCOL]) * r2pred
        results[GWAS.NEFFCOL] = gwas[GWAS.NCOL].tolist() + list(neff)
    results[GWAS.PCOL] = gwas[GWAS.PCOL].tolist() + list(pvals)

    df = pd.DataFrame(data=results)
    if GWAS.NCOL in gwas:
        df = df[[GWAS.CHRCOL, GWAS.SNPCOL, GWAS.BPCOL, GWAS.A1COL, GWAS.A2COL, GWAS.TYPECOL, GWAS.ZCOL, GWAS.ADJR2COL,
                GWAS.NEFFCOL, GWAS.PCOL]]
    else:
        df = df[[GWAS.CHRCOL, GWAS.SNPCOL, GWAS.BPCOL, GWAS.A1COL, GWAS.A2COL, GWAS.TYPECOL, GWAS.ZCOL, GWAS.ADJR2COL,
                GWAS.PCOL]]
                
    df = df.sort_values(by=[GWAS.BPCOL])
    log.info("Completed imputation at region {}".format(ref))

    return df
