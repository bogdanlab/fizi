import numpy as np
import numpy.linalg as lin

import fimpg
import pandas as pd
from pandas_plink import read_plink


class RefPanel(object):
    CHRCOL = "chrom"
    SNPCOL = "snp"
    BPCOL = "pos"
    A1COL = "a1"
    A2COL = "a0"

    #chrom         snp   cm       pos a0 a1    i

    def __init__(self, snp_info, sample_info, geno):
        self._snp_info = snp_info
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

    @property
    def CHRs(self):
        return self._snp_info[RefPanel.CHRCOL].unique()

    def filter_snps_by_chr(self, chr):
        return self._snp_info.loc[self._snp_info[RefPanel.CHRCOL] == chr]

    def partitions_by_chr(self, window_size):
        """
        Lazily iterate over location partitions
        """
        for chrom in self.CHRs:
            snps = self.filter_snps_by_chr(chrom)
            min_pos = snps[RefPanel.BPCOL].min()
            max_pos = snps[RefPanel.BPCOL].max()

            nwin = int(np.ceil((max_pos - min_pos + 1) / window_size))

            yield [chrom, min_pos, min(min_pos + window_size, max_pos)]
            last_stop = min_pos + window_size

            for i in range(1, nwin):
                start = last_stop + 1
                stop = min(start + window_size, max_pos)
                yield [chrom, start, stop]
                last_stop = stop

        return

    def subset_by_pos(self, chrom, start, stop):
        df = self._snp_info
        if pd.api.types.is_string_dtype(df[RefPanel.CHRCOL]) or pd.api.types.is_categorical_dtype(df[RefPanel.CHRCOL]):
            chrom = str(chrom)
        else:
            chrom = int(chrom)

        if pd.api.types.is_string_dtype(df[RefPanel.BPCOL]):
            start = str(start)
            stop = str(stop)
        else:
            start = int(start)
            stop = int(stop)

        snps = df.loc[(df[RefPanel.CHRCOL] == chrom) & (df[RefPanel.BPCOL] >= start) & (df[RefPanel.BPCOL] <= stop)]

        return RefPanel(snps, self._sample_info, self._geno)

    def overlap_gwas(self, gwas):
        df = self._snp_info
        merged_snps = pd.merge(gwas, df, how="outer", left_on=fimpg.GWAS.SNPCOL, right_on=fimpg.RefPanel.SNPCOL)
        return merged_snps

    def get_geno(self, snps=None):
        if snps is None:
            return self._geno.compute().T
        else:
            return self._geno[snps.i.values, :].compute().T

    @property
    def sample_size(self):
        return len(self._sample_info)

    @property
    def effective_snp_size(self):
        pass

    def estimate_LD(self, snps=None, lmbda=0.1):
        G = self.get_geno(snps)

        n, p = G.shape
        col_mean = np.nanmean(G, axis=0)

        # unlikely, but drop SNPs that are completely missing
        #inds = np.where(!np.isnan(col_mean))
        #G = G.T[inds].T
        #col_mean = col_mean[inds]

        # impute missing with column mean
        inds = np.where(np.isnan(G))
        G[inds] = np.take(col_mean, inds[1])

        LD = np.corrcoef(G.T) + np.eye(p) * lmbda
        return LD

    @classmethod
    def parse_plink(cls, path):
        bim, fam, bed = read_plink(path, verbose=False)
        return RefPanel(bim, fam, bed)
