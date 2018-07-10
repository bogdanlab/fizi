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
elements = [(''.join(x), x[0] == COMPLEMENT[x[1]])
                    for x in it.product(BASES, BASES)
                    if x[0] != x[1]]
STRAND_AMBIGUOUS = dict((key, value) for (key, value) in elements)
# SNPS we want to keep (pairs of alleles)
VALID_SNPS = [x for x in map(lambda y: ''.join(y), it.product(BASES, BASES))
              if x[0] != x[1] and not STRAND_AMBIGUOUS[x]]
#VALID_SNPS = dict((x[0], x[1]) for x in elements)
# T iff SNP 1 has the same alleles as SNP 2 (allowing for strand or ref allele flip).
MATCH_ALLELES = [x for x in map(lambda y: ''.join(y), it.product(VALID_SNPS, VALID_SNPS))
                 # strand and ref match
                 if ((x[0] == x[2]) and (x[1] == x[3])) or
                 # ref match, strand flip
                 ((x[0] == COMPLEMENT[x[2]]) and (x[1] == COMPLEMENT[x[3]])) or
                 # ref flip, strand match
                 ((x[0] == x[3]) and (x[1] == x[2])) or
                 ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))]  # strand and ref flip
#MATCH_ALLELES = dict((x[0], x[1]) for x in elements)
# T iff SNP 1 has the same alleles as SNP 2 w/ ref allele flip.
FLIP_ALLELES = [(''.join(x),
                ((x[0] == x[3]) and (x[1] == x[2])) or  # strand match
                # strand flip
                ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]])))
                for x in MATCH_ALLELES]
FLIP_ALLELES = dict((key, value) for (key,value) in elements)

def impute_gwas(gwas, ref, annot=None, sigmas=None, prop=0.4, epsilon=1e-6):
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
    n = gwas.iloc[0][GWAS.NCOL]
 
    # compute linkage-disequilibrium estimate
    LD = ref.estimate_LD(ref_snps)
    obs_flag = ~pd.isna(ref_snps.Z)
    to_impute = (~obs_flag).values
    obs = obs_flag.values

    # check for allele flips
    sset = ref_snps[obs_flag]
    obsZ = sset.Z.values
    alleles = sset[GWAS.A1COL] + sset[GWAS.A2COL] + sset[RefPanel.A1COL] + sset[RefPanel.A2COL]

    if annot is not None and sigmas is not None:
        
        # filter out annotations with low tau Z-scores
        sigmas = sigmas.loc[(sigmas[Sigmas.NAMECOL]=="baseL2_0") | (sigmas[Sigmas.SIGMAZCOL].astype(float) < -2.0) | (sigmas[Sigmas.SIGMAZCOL].astype(float) > 2.0)]
        
        sigma_values = sigmas[Sigmas.SIGMACOL]
        annot_names = sigmas[Sigmas.NAMECOL].values.flatten()
        
        # remove "L2_0" from annot names in sigmas file so they match with names in annotation file
        for a in range(len(annot_names)):
            if annot_names[a][len(annot_names[a])-4:len(annot_names[a])] == "L2_0":
                annot_names[a] = annot_names[a][0:len(annot_names[a])-4]
        annot_names = annot_names.tolist()

        m = len(obs)
        D = np.zeros(shape=(m, m))

        # merge annotations with merged reference snps
        merged_annot = pd.merge(ref_snps, annot, how="outer", left_on=GWAS.SNPCOL, right_on=fimpg.Annot.SNPCOL)
        merged_annot.drop_duplicates(subset=fimpg.RefPanel.SNPCOL)
        
        for i in range(m):
            col_info = merged_annot[annot_names]
            annot_info = col_info.iloc[i,:].values.flatten()
            
            # if annotations are not available for this SNP
            if np.isnan(annot_info[0]):
                annot_data = np.zeros(shape=(1,len(annot_names)))
                annot_data[0,0] = 1
            
            # if annotations are available
            else:
                annot_data = annot_info
            
            D[i,i] = np.dot(annot_data, np.multiply(sigma_values,n))


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
    mprop = nobs / float(nobs+nimp)
    if mprop < prop:
        log.info("Skipping region {}. Too few SNPs for imputation {}%".format(ref, mprop))
        return None

    Voo_ld = LD[obs].T[obs].T
    Vuo_ld = LD[to_impute].T[obs].T
    Vou_ld = LD[obs].T[to_impute].T
    Vuu_ld = LD[to_impute].T[to_impute].T
    if sigmas is not None and annot is not None:
        Do = D.T[obs].T[obs]
        Du = D.T[to_impute].T[to_impute]
        unobsV = Vuo_ld + lin.multi_dot([Vuu_ld, Du, Vuo_ld]) + lin.multi_dot([Vuo_ld, Do, Voo_ld])
        obsV = Voo_ld + lin.multi_dot([Voo_ld, Do, Voo_ld]) + lin.multi_dot([Vou_ld, Du, Vuo_ld])
    else:
        obsV = Voo_ld
        unobsV = Vuo_ld

    obsVinv = lin.pinv(obsV)

    # predict the Z-scores
    impZs = lin.multi_dot([unobsV, obsVinv, obsZ])

    # compute two-sided z-test for p-value
    pvals = stats.chi2.sf(impZs ** 2, 1)

    # compute r2-pred

    r2pred = np.diag(lin.multi_dot([unobsV, obsVinv, unobsV.T]))
    # compute r2-pred adjusted for effective number of markers used in inference
    n_ref = ref.sample_size

    # compute effective size
    svals = svdvals(obsV)
    p_eff = np.sum(svals > epsilon)

    def _r2adj(r2p):
        return max(1 - (1 - r2p) * ((n_ref - 1) / float(n_ref - p_eff - 1)), epsilon)

    r2adj = np.vectorize(_r2adj)
    r2pred_adj = r2adj(r2pred)

    obs_snps = ref_snps[obs]
    imp_snps = ref_snps[to_impute]

    nall = len(obs_snps)

    # this needs to be cleaned up. at some point just switch to 'standard' columns
    results = dict()
    results[GWAS.CHRCOL] = [obs_snps[GWAS.CHRCOL].iloc[0]] * (nimp + nall)
    results[GWAS.SNPCOL] = obs_snps[GWAS.SNPCOL].tolist() + imp_snps[RefPanel.SNPCOL].tolist()
    results[GWAS.BPCOL] = obs_snps[GWAS.BPCOL].tolist() + imp_snps[RefPanel.BPCOL].tolist()
    results[GWAS.A1COL] = obs_snps[GWAS.A1COL].tolist() + imp_snps[RefPanel.A1COL].tolist()
    results[GWAS.A2COL] = obs_snps[GWAS.A2COL].tolist() + imp_snps[RefPanel.A2COL].tolist()
    results[GWAS.TYPECOL] = (["gwas"] * nall) + (["imputed"] * nimp)
    results[GWAS.ZCOL] = obs_snps[GWAS.ZCOL].tolist() + list(impZs)
    results[GWAS.ADJR2COL] = ([1.0] * nall) + list(r2pred_adj)
    if GWAS.NCOL in obs_snps:
        neff = np.max(obs_snps[GWAS.NCOL]) * r2pred
        results[GWAS.NEFFCOL] = obs_snps[GWAS.NCOL].tolist() + list(neff)
    results[GWAS.PCOL] = obs_snps[GWAS.PCOL].tolist() + list(pvals)

    df = pd.DataFrame(data=results)
    if GWAS.NCOL in obs_snps:
        df = df[[GWAS.CHRCOL, GWAS.SNPCOL, GWAS.BPCOL, GWAS.A1COL, GWAS.A2COL, GWAS.TYPECOL, GWAS.ZCOL, GWAS.ADJR2COL,
                GWAS.NEFFCOL, GWAS.PCOL]]
    else:
        df = df[[GWAS.CHRCOL, GWAS.SNPCOL, GWAS.BPCOL, GWAS.A1COL, GWAS.A2COL, GWAS.TYPECOL, GWAS.ZCOL, GWAS.ADJR2COL,
                GWAS.PCOL]]
                
    #df = df.sort_values(by=[GWAS.BPCOL])
    log.info("Completed imputation at region {}".format(ref))

    df = df.sort_values(by=[GWAS.BPCOL])

    return df
