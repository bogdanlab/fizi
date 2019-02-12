import logging
import warnings

import pyfizi
import numpy as np
import pandas as pd

import scipy.linalg as lin

from numpy.linalg import multi_dot as mdot
from pandas_plink import read_plink


class RefPanel(object):
    CHRCOL = "chrom"
    SNPCOL = "snp"
    BPCOL = "pos"
    A1COL = "a1"
    A2COL = "a0"

    # chrom snp  cm pos a0 a1 i

    def __init__(self, snp_info, sample_info, geno):
        self._snp_info = snp_info
        if pd.api.types.is_categorical_dtype(self._snp_info[RefPanel.A1COL]):
            self._snp_info.loc[:, RefPanel.A1COL] = self._snp_info[RefPanel.A1COL].astype('str')
            self._snp_info.loc[:, RefPanel.A2COL] = self._snp_info[RefPanel.A2COL].astype('str')
        self._sample_info = sample_info
        self._geno = geno
        return

    def __len__(self):
        return len(self._snp_info)

    def __str__(self):
        start_chr = self._snp_info[RefPanel.CHRCOL].iloc[0]
        stop_chr = self._snp_info[RefPanel.CHRCOL].iloc[-1]

        start_bp = self._snp_info[RefPanel.BPCOL].iloc[0]
        stop_bp = self._snp_info[RefPanel.BPCOL].iloc[-1]
        return "{}:{} - {}:{}".format(start_chr, int(start_bp), stop_chr, int(stop_bp))

    def get_partitions(self, window_size, chrom=None, start=None, stop=None):
        """
        Lazily iterate over location partitions
        """
        log = logging.getLogger(pyfizi.LOG)

        chroms = self._snp_info[RefPanel.CHRCOL].unique()

        if chrom is not None:
            if chrom not in chroms:
                msg = "User supplied chromosome {} is not found in data".format(chrom)
                log.error(msg)
                return

        for chrm in chroms:
            if chrom is not None and chrom != chrm:
                continue

            snps = self._snp_info.loc[self._snp_info[RefPanel.CHRCOL] == chrm]

            min_pos_indata = snps[RefPanel.BPCOL].min()
            max_pos_indata = snps[RefPanel.BPCOL].max()

            # check against user arguments
            if start is not None:
                min_pos = int(start)
                if min_pos < min_pos_indata and min_pos < max_pos_indata:
                    msg = "User supplied start {} is less than min start found in data {}. Switching to data start"
                    msg = msg.format(min_pos, min_pos_indata)
                    log.warning(msg)
                    min_pos = min_pos_indata
            else:
                min_pos = min_pos_indata

            if stop is not None:
                max_pos = int(stop)
                if max_pos > max_pos_indata and max_pos > min_pos_indata:
                    msg = "User supplied stop {} is greater than max stop found in data {}. Switching to data stop"
                    msg = msg.format(max_pos, max_pos_indata)
                    log.warning(msg)
                    max_pos = max_pos_indata
            else:
                max_pos = max_pos_indata

            nwin = int(np.ceil((max_pos - min_pos + 1) / window_size))
            yield [chrm, min_pos, min(min_pos + window_size, max_pos)]

            last_stop = min_pos + window_size
            for i in range(1, nwin):
                start = last_stop + 1
                stop = min(start + window_size, max_pos)
                yield [chrm, start, stop]
                last_stop = stop

        return

    def subset_by_pos(self, chrom, start=None, stop=None, clean_snps=True):
        df = self._snp_info
        if start is not None and stop is not None:
            snps = df.loc[(df[RefPanel.CHRCOL] == chrom) & (df[RefPanel.BPCOL] >= start) & (df[RefPanel.BPCOL] <= stop)]
        elif start is not None and stop is None:
            snps = df.loc[(df[RefPanel.CHRCOL] == chrom) & (df[RefPanel.BPCOL] >= start)]
        elif start is None and stop is not None:
            snps = df.loc[(df[RefPanel.CHRCOL] == chrom) & (df[RefPanel.BPCOL] <= stop)]
        else:
            snps = df.loc[(df[RefPanel.CHRCOL] == chrom)]

        if clean_snps:
            valid = pyfizi.check_valid_snp(snps[RefPanel.A1COL], snps[RefPanel.A2COL])
            snps = snps.loc[valid].drop_duplicates(subset=pyfizi.RefPanel.SNPCOL)

        return RefPanel(snps, self._sample_info, self._geno)

    def overlap_gwas(self, gwas):
        df = self._snp_info

        # we only need to perform right join to get all matching RefPanel SNPs
        # this is because we can re-use the original, unmatched, unverified GWAS data in output
        merged_snps = pd.merge(gwas, df, how="right", left_on=pyfizi.GWAS.SNPCOL, right_on=pyfizi.RefPanel.SNPCOL)

        gwas_a1 = merged_snps[pyfizi.GWAS.A1COL]
        gwas_a2 = merged_snps[pyfizi.GWAS.A2COL]
        ref_a1 = merged_snps[pyfizi.RefPanel.A1COL]
        ref_a2 = merged_snps[pyfizi.RefPanel.A2COL]

        # RefPanel-only SNPs will be NA for GWAS; keep those
        valid_ref = pd.isna(gwas_a1)

        # Alleles for GWAS SNPs must be valid pairs with RefPanel alleles
        valid_match = pyfizi.check_valid_alleles(gwas_a1, gwas_a2, ref_a1, ref_a2)

        # final valid is: valid RefPanel SNPs or non-ambiguous GWAS SNPs that match RefPanel SNPs
        merged = pyfizi.MergedPanel(merged_snps.loc[valid_ref | valid_match])

        return merged

    def get_geno(self, snps=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            if snps is None:
                return self._geno.compute().T
            else:
                return self._geno[snps.i.values, :].compute().T

    @property
    def sample_size(self):
        return float(len(self._sample_info))

    def estimate_ld(self, snps=None, adjust=0.1, return_eigvals=False):
        G = self.get_geno(snps)
        n, p = G.shape
        col_mean = np.nanmean(G, axis=0)

        # impute missing with column mean
        inds = np.where(np.isnan(G))
        G[inds] = np.take(col_mean, inds[1])
        G = (G - np.mean(G, axis=0)) / np.std(G, axis=0)

        if return_eigvals:
            _, S, V = lin.svd(G, full_matrices=True)

            # adjust 
            D = np.full(p, adjust)
            D[:len(S)] = D[:len(S)] + (S**2 / n)

            # same as `mdot([V.T, np.diag(D), V]), D)`
            return np.dot(V.T * D, V), D
        else:
            return (np.dot(G.T, G) / self.sample_size) + (np.eye(p) * adjust)

    @classmethod
    def parse_plink(cls, path):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', 'FutureWarning')
            bim, fam, bed = read_plink(path, verbose=False)
        return RefPanel(bim, fam, bed)


class MergedPanelSeries(pd.Series):
    @property
    def _constructor(self):
        return pyfizi.MergedPanelSeries

    @property
    def _constructor_expanddim(self):
        return pyfizi.MergedPanel


class MergedPanel(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        super(MergedPanel, self).__init__(*args, **kwargs)
        return

    @property
    def _constructor(self):
        return pyfizi.MergedPanel

    @property
    def _constructor_sliced(self):
        return pyfizi.MergedPanelSeries

    @property
    def zscores(self):
        return self["Z"].values

    @property
    def gwas_a1_alleles(self):
        return self[pyfizi.GWAS.A1COL]

    @property
    def gwas_a2_alleles(self):
        return self[pyfizi.GWAS.A2COL]

    @property
    def ref_a1_alleles(self):
        return self[pyfizi.RefPanel.A1COL]

    @property
    def ref_a2_alleles(self):
        return self[pyfizi.RefPanel.A2COL]

    def are_observations(self):
        obs_flag = ~pd.isna(self.zscores)
        return obs_flag

    def are_imputations(self):
        to_impute = pd.isna(self.zscores)
        return to_impute
