import pandas as pd

__all__ = ["GWAS"]

class GWAS(object):
    """
    Thin wrapper for a pandas DataFrame object containing GWAS summary data.
    Assumes the GWAS data have gone through LDSC munge-sumstat
    """

    CHR = "CHR"
    SNP = "SNP"
    BP = "BP"
    A1 = "A1"
    A2 = "A2"
    Z = "Z"
    P = "P"
    N = "N"

    REQ_COLS = [CHR, SNP, BP, A1, A2, Z, P, N]

    def __init__(self, gwas_df):
        self._df = gwas_df
        for column in GWAS.REQ_COLS:
            if column not in self._df:
                raise ValueError("{}-column not found in summary statistics".format(column))
        return

    def __len__(self):
        return len(self._df)

    def __contains__(self, name):
        return name in self._df

    @property
    def SNPs(self):
        return self._df[GWAS.SNP].tolist()

    @property
    def BPs(self):
        return self._df[GWAS.BP].tolist()

    @property
    def CHRs(self):
        return self._df[GWAS.CHR].tolist()

    @property
    def Zs(self):
        return self._df[GWAS.Z].tolist()

    @property
    def Pvals(self):
        return self._df[GWAS.P].tolist()

    @property
    def Ns(self):
        return self._df[GWAS.N].tolist()

    @classmethod
    def parse_gwas(cls, stream):
        df = pd.read_csv(stream, delim_whitespace=True)
        return cls(df)
