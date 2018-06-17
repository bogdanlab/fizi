import pandas as pd
import scipy.stats as stats

import fimpg

__all__ = ["GWAS", "GWASSeries"]


class GWASSeries(pd.Series):
    @property
    def _constructor(self):
        return fimpg.GWASSeries

    @property
    def _constructor_expanddim(self):
        return fimpg.GWAS


class GWAS(pd.DataFrame):
    """
    Thin wrapper for a pandas DataFrame object containing GWAS summary data.
    Assumes the GWAS data have gone through LDSC munge-sumstat
    """

    CHRCOL = "CHR"
    SNPCOL = "SNP"
    BPCOL = "BP"
    A1COL = "A1"
    A2COL = "A2"
    ZCOL = "Z"

    PCOL = "P"
    NCOL = "N"

    NEFFCOL = "NEFF"
    TYPECOL = "TYPE"
    ADJR2COL = "ADJ.R2.PRED"

    REQ_COLS = [CHRCOL, SNPCOL, BPCOL, A1COL, A2COL, ZCOL]

    def __init__(self, *args, **kwargs):
        super(GWAS, self).__init__(*args, **kwargs)
        return

    @property
    def _constructor(self):
        return fimpg.GWAS

    @property
    def _constructor_sliced(self):
        return fimpg.GWASSeries

    def subset_by_pos(self, chrom, start, stop):
        if pd.api.types.is_string_dtype(self[GWAS.CHRCOL]) or pd.api.types.is_categorical_dtype(self[GWAS.CHRCOL]):
            chrom = str(chrom)
        else:
            chrom = int(chrom)

        if pd.api.types.is_string_dtype(self[GWAS.BPCOL]):
            start = str(start)
            stop = str(stop)
        else:
            start = int(start)
            stop = int(stop)

        snps = self.loc[(self[GWAS.CHRCOL] == chrom) & (self[GWAS.BPCOL] >= start) & (self[GWAS.BPCOL] <= stop)]

        return GWAS(snps)

    @classmethod
    def parse_gwas(cls, stream):
        df = pd.read_csv(stream, delim_whitespace=True)
        for column in GWAS.REQ_COLS:
            if column not in df:
                raise ValueError("{}-column not found in summary statistics".format(column))

        if GWAS.PCOL not in df:
            df[GWAS.PCOL] = stats.chi2.sf(df[GWAS.ZCOL] ** 2, 1)

        return cls(df)
