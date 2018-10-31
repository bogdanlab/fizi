import logging
import warnings

import fizi
import numpy as np
import pandas as pd

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
        log = logging.getLogger(fizi.LOG)

        chroms = self._snp_info[RefPanel.CHRCOL].unique()

        if chrom is not None:
            chrom = self.clean_chrom(chrom)
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

    def subset_by_pos(self, chrom, start=None, stop=None, filter_ambig=True):
        AMBIG = ["AT", "TA", "CG", "GC"]
        df = self._snp_info
        chrom = self.clean_chrom(chrom)
        if start is not None and stop is not None:
            snps = df.loc[(df[RefPanel.CHRCOL] == chrom) & (df[RefPanel.BPCOL] >= start) & (df[RefPanel.BPCOL] <= stop)]
        elif start is not None and stop is None:
            snps = df.loc[(df[RefPanel.CHRCOL] == chrom) & (df[RefPanel.BPCOL] >= start)]
        elif start is None and stop is not None:
            snps = df.loc[(df[RefPanel.CHRCOL] == chrom) & (df[RefPanel.BPCOL] <= stop)]
        else:
            snps = df.loc[(df[RefPanel.CHRCOL] == chrom)]

        if filter_ambig:
            alleles = snps[RefPanel.A1COL] + snps[RefPanel.A2COL]
            non_ambig = alleles.apply(lambda y: y.upper() not in AMBIG)
            snps = snps[non_ambig]

        return RefPanel(snps, self._sample_info, self._geno)

    def overlap_gwas(self, gwas):
        df = self._snp_info
        merged_snps = pd.merge(gwas, df, how="outer", left_on=fizi.GWAS.SNPCOL, right_on=fizi.RefPanel.SNPCOL)
        merged_snps.drop_duplicates(subset=fizi.RefPanel.SNPCOL, inplace=True)
        return merged_snps

    def get_geno(self, snps=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            if snps is None:
                return self._geno.compute().T
            else:
                return self._geno[snps.i.values, :].compute().T

    @property
    def sample_size(self):
        return len(self._sample_info)

    def estimate_LD(self, snps=None, adjust=0.1):
        G = self.get_geno(snps)
        n, p = G.shape
        col_mean = np.nanmean(G, axis=0)

        # impute missing with column mean
        inds = np.where(np.isnan(G))
        G[inds] = np.take(col_mean, inds[1])

        LD = np.corrcoef(G.T) + np.eye(p) * adjust
        return LD

    def clean_chrom(self, chrom):
        df = self._snp_info
        ret_val = None
        if pd.api.types.is_string_dtype(df[RefPanel.CHRCOL]) or pd.api.types.is_categorical_dtype(df[RefPanel.CHRCOL]):
            ret_val = str(chrom)
        else:
            ret_val = int(chrom)

        return ret_val

    @classmethod
    def parse_plink(cls, path):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', 'FutureWarning')
            bim, fam, bed = read_plink(path, verbose=False)
        return RefPanel(bim, fam, bed)
